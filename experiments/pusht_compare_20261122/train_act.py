"""Train the existing PiPER relative-action ACT on the matched Push-T window population."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch

from accelerate import Accelerator
import numpy as np
import torch
from torch.utils.data import Subset

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
import lerobot.scripts.lerobot_train as upstream
from lerobot.utils.constants import IMAGENET_STATS

from piper_outcome_stack.smolvla_evaluate import evaluate
from piper_outcome_stack.smolvla_training import make_views, write_json
from piper_outcome_stack.training import select_policy_inputs
from piper_outcome_stack.training_windows import CompleteWindows, sampler_class
from piper_policy.positive_act import PositiveACTConfig
from piper_policy.smolvla import IMAGE_FEATURE
from piper_policy.statistics import numeric_stats
from piper_policy.training import act_processors, register_act_factory


def run(spec):
    root, output = Path(spec["run_root"]), Path(spec["output_dir"])
    if output.exists() or (root / "training-status.json").exists():
        raise FileExistsError(output)
    raw, development = make_views(spec)
    counts = spec["demonstrations"]
    training = CompleteWindows(raw, counts["train"], counts["task_layout_cells"])
    dev_ids = {(r["source_session"], r["source_episode"]) for r in development.lineage}
    if len(dev_ids) != counts["development"]:
        raise ValueError("original development demonstration count differs")
    # Normalize each future absolute target relative to the state at its generation anchor.
    # The retained view guarantees that every training horizon stays inside one valid segment.
    horizon = spec["input_contract"]["chunk_size"]
    starts = training.indices
    relative = raw.actions[starts[:, None] + np.arange(horizon)] - raw.states[starts, None, :]
    relative_stats = numeric_stats(relative.reshape(-1, 7))
    del relative
    stats = deepcopy(raw.meta.stats)
    stats[IMAGE_FEATURE].update(
        {k: np.asarray(v, dtype=np.float32) for k, v in IMAGENET_STATS.items()}
    )
    opt, budget = spec["optimization"], spec["budget"]
    pcfg = PositiveACTConfig(
        device="cuda",
        push_to_hub=False,
        use_amp=False,
        chunk_size=horizon,
        n_action_steps=horizon,
        input_features=select_policy_inputs(raw.meta.features, [IMAGE_FEATURE]),
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        vision_backbone="resnet18",
        pretrained_backbone_weights=spec["initialization"]["backbone_weights"],
        optimizer_lr=opt["learning_rate"],
        optimizer_lr_backbone=opt["backbone_lr"],
        gripper_state_mean=float(stats["observation.state"]["mean"][6]),
        gripper_state_std=float(stats["observation.state"]["std"][6]),
        gripper_action_mean=float(relative_stats["mean"][6]),
        gripper_action_std=float(relative_stats["std"][6]),
        positive_scale_m=spec["positive_scale_m"],
        positive_floor_m=spec["positive_floor_m"],
    )
    pcfg.drop_n_last_frames = 0
    subset = Subset(
        development, development.evaluation_indices(spec["evaluation"]["samples_per_episode"])
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=training.repo_id,
            root=Path(spec["train_sources"][0]["root"]),
            eval_split=development.num_episodes
            / (training.num_episodes + development.num_episodes),
        ),
        policy=pcfg,
        output_dir=output,
        job_name=spec["run_id"],
        seed=opt["seed"],
        batch_size=opt["batch_size"],
        num_workers=opt["num_workers"],
        steps=budget["max_steps"],
        log_freq=50,
        save_freq=budget["save_interval_steps"],
        eval_steps=budget["eval_interval_steps"],
        env_eval_freq=0,
    )
    cfg.dataset.image_transforms.enable = False
    cfg.wandb.enable = False
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "dataset-view.json",
        dict(
            train=training.summary,
            train_state_statistics=stats["observation.state"],
            relative_action_statistics=relative_stats,
            development_demos=len(dev_ids),
            development_segments=development.num_episodes,
            data_manifest_sha256=spec["data_manifest_sha256"],
        ),
    )
    started = time.monotonic()
    status = dict(
        status="starting",
        updates=0,
        start_step=0,
        hardware_io=False,
        fresh_optimizer=True,
        training_windows=len(training),
        run_id=spec["run_id"],
    )

    def save_status():
        status["wall_seconds"] = time.monotonic() - started
        write_json(root / "training-status.json", status)

    def processors(policy_cfg, **kwargs):
        return act_processors(
            policy_cfg,
            dataset_stats=stats,
            relative_stats=relative_stats,
            joint_state_std_floor_rad=spec["joint_state_std_floor_rad"],
        )

    official_update = upstream.update_policy

    def update(*args, **kwargs):
        if time.monotonic() - started > budget["max_wallclock_seconds"]:
            raise TimeoutError("ACT training exceeded its fixed wall-clock ceiling")
        model, batch, optimizer = args[1:4]
        if not status["updates"]:
            if optimizer.state or model.config.pretrained_path is not None:
                raise ValueError("fresh ACT requires a new policy and optimizer")
            cache = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
            backbone = torch.load(cache, map_location="cpu", weights_only=True)
            if not torch.equal(
                model.model.backbone.conv1.weight.detach().cpu(), backbone["conv1.weight"]
            ):
                raise ValueError("ImageNet ResNet18 initialization differs")
            write_json(
                root / "initialization.json",
                dict(
                    step=0,
                    policy_checkpoint=None,
                    backbone=str(cache),
                    verified_backbone_conv1=True,
                    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                ),
            )
            status["first_batch_indices"] = batch["index"].detach().cpu().tolist()
        updated = official_update(*args, **kwargs)
        if not all(np.isfinite(v) for v in updated[1].values() if isinstance(v, (float, int))):
            raise ValueError("nonfinite ACT objective")
        status["updates"] += 1
        if status["updates"] == 1 or status["updates"] % 50 == 0:
            with (root / "objective.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=status["updates"], **updated[1])) + "\n")
            status.update(status="running", last_step=status["updates"])
            save_status()
        return updated

    save_status()
    try:
        torch.set_num_threads(4)
        with ExitStack() as stack:
            stack.enter_context(register_act_factory())
            for name, fn in [
                ("make_train_eval_datasets", lambda cfg: (training, subset)),
                ("make_pre_post_processors", processors),
                ("EpisodeAwareSampler", sampler_class(training)),
                ("update_policy", update),
            ]:
                stack.enter_context(patch.object(upstream, name, fn))
            upstream.train(
                cfg,
                accelerator=Accelerator(mixed_precision="no", step_scheduler_with_optimizer=False),
            )
        if status["updates"] != cfg.steps:
            raise RuntimeError("ACT update count differs from the requested budget")
        cp = output / "checkpoints" / f"{cfg.steps:06d}" / "pretrained_model"
        write_json(
            cp / "representation.json",
            dict(
                joint_representation="relative",
                horizon=horizon,
                internal_action="absolute seven-axis target at t+k minus observed state at t",
                physical_output="decoded full absolute rad/m chunk at the same generation anchor",
                state_std_floor_rad=spec["joint_state_std_floor_rad"],
                gripper="positive softplus",
            ),
        )
        status.update(status="evaluating", final_checkpoint=str(cp))
        save_status()
        evaluate(spec, cp, root / "evaluation", "act")
        status.update(status="complete", last_step=cfg.steps)
    except BaseException as err:
        status.update(status="failed", error=repr(err))
        raise
    finally:
        save_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()))
