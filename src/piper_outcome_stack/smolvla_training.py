"""Build PiPER data/processors and delegate fine-tuning to official LeRobot."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import json
import time
from unittest.mock import patch

import numpy as np
import torch

from .retained_dataset import RetainedDataset, require_disjoint_sessions
from piper_policy.smolvla_gripper import make_processors
from piper_policy.smolvla import IMAGE_FEATURE, SmolVLAChunkPredictor, finetune_config


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")


def validate_task(spec, task):
    contract = spec["input_contract"]
    expected = contract["tasks"] if "tasks" in contract else [contract["task"]]
    if not isinstance(task, str) or not task.strip() or task not in expected:
        raise ValueError(f"source task is not in the frozen input contract: {task!r}")
    return task


def make_views(spec):
    train = RetainedDataset(spec["train_sources"], repo_id="local/smolvla-retained-train")
    development = RetainedDataset(
        spec["development_sources"], repo_id="local/smolvla-retained-development"
    )
    require_disjoint_sessions(train, development)
    for view in (train, development):
        for row in view.lineage:
            if "task" in row:
                validate_task(spec, row["task"])
    if (len(train), train.num_episodes, development.num_episodes) != (
        spec["expected_train_frames"],
        spec["expected_train_episodes"],
        spec["expected_development_episodes"],
    ):
        raise ValueError("loaded data differs from the frozen experiment counts")
    return train, development


def view_report(train, development, spec):
    return {
        "train_frames": len(train),
        "train_episodes": train.num_episodes,
        "development_frames": len(development),
        "development_episodes": development.num_episodes,
        "train_lineage": train.lineage,
        "development_lineage": development.lineage,
        "train_sources": spec["train_sources"],
        "development_sources": spec["development_sources"],
        "normalization_source": "retained training frames only; processors store encoded action statistics",
        "action_encoding": spec.get("action_encoding", "absolute"),
        "reported_statistics": "raw physical rad/m; encoded statistics are in saved processors",
        "state_statistics": train.meta.stats["observation.state"],
        "action_statistics": train.meta.stats["action"],
        "development_sample_indices": development.evaluation_indices(
            spec["evaluation"]["samples_per_episode"]
        ),
    }


def parameter_report(model):
    groups = {}
    for name, value in model.named_parameters():
        key = (
            "vlm"
            if ".vlm." in name
            else "action_expert"
            if ".lm_expert." in name
            else "projections"
        )
        entry = groups.setdefault(key, {"total": 0, "trainable": 0})
        entry["total"] += value.numel()
        entry["trainable"] += value.numel() if value.requires_grad else 0
    return {
        "total": sum(v.numel() for v in model.parameters()),
        "trainable": sum(v.numel() for v in model.parameters() if v.requires_grad),
        "groups": groups,
    }


def configured_policy(spec):
    cfg = finetune_config(spec["source_checkpoint"], spec["vlm_assets"])
    cfg.optimizer_lr = spec["optimization"]["learning_rate"]
    cfg.scheduler_warmup_steps = spec["optimization"]["warmup_steps"]
    cfg.scheduler_decay_steps = spec["budget"]["max_steps"]
    return cfg


def normalized_batch(batch, preprocessor):
    if batch[IMAGE_FEATURE].dtype != torch.uint8:
        raise ValueError("source training RGB must remain uint8 until the official scaling step")
    batch = dict(batch)
    batch[IMAGE_FEATURE] = batch[IMAGE_FEATURE].float() / 255.0
    return preprocessor(batch)


def smoke(spec, output):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "started", "hardware_io": False}
    started = time.monotonic()
    try:
        torch.manual_seed(spec["optimization"]["seed"])
        torch.set_num_threads(4)
        train, dev = make_views(spec)
        write_json(output / "data-view.json", view_report(train, dev, spec))
        cfg = configured_policy(spec)
        model = SmolVLAPolicy.from_pretrained(spec["source_checkpoint"], config=cfg, strict=True)
        model = model.to("cuda")
        pre, post = make_processors(cfg, train, spec.get("action_encoding", "absolute"))
        result["parameters"] = parameter_report(model)
        initial = output / "initial/pretrained_model"
        model.save_pretrained(initial)
        pre.save_pretrained(initial)
        post.save_pretrained(initial)
        item = train[0]
        task = validate_task(spec, item["task"])
        batch = next(
            iter(
                torch.utils.data.DataLoader(
                    train, batch_size=spec["optimization"]["batch_size"], num_workers=0
                )
            )
        )
        batch = normalized_batch(batch, pre)
        model.train()
        optimizer = torch.optim.AdamW(
            [v for v in model.parameters() if v.requires_grad],
            lr=cfg.optimizer_lr,
            betas=cfg.optimizer_betas,
            eps=cfg.optimizer_eps,
            weight_decay=cfg.optimizer_weight_decay,
        )
        watched = model.model.state_proj.weight
        before = watched.detach().clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model.forward(batch)
        if not torch.isfinite(loss):
            raise ValueError("non-finite pretrained training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer_grad_clip_norm)
        if not torch.isfinite(grad_norm):
            raise ValueError("non-finite pretrained training gradient")
        optimizer.step()
        change = float((watched.detach() - before).abs().max())
        if change == 0:
            raise ValueError("state projection did not update")
        checkpoint = output / "updated/pretrained_model"
        model.save_pretrained(checkpoint)
        pre.save_pretrained(checkpoint)
        post.save_pretrained(checkpoint)
        predictor = SmolVLAChunkPredictor(model.eval(), pre, post, task)
        rgb = item[IMAGE_FEATURE].permute(1, 2, 0).numpy()
        state = item["observation.state"].numpy()
        torch.manual_seed(123)
        expected = predictor.predict(rgb, state)
        reloaded = SmolVLAChunkPredictor.from_checkpoint(checkpoint, task=task)
        torch.manual_seed(123)
        actual = reloaded.predict(rgb, state)
        error = float(abs(actual - expected).max())
        if error > 1e-5:
            raise ValueError(f"saved/reloaded prediction differs: {error}")
        np.savez_compressed(output / "reload-predictions.npz", expected=expected, actual=actual)
        result.update(
            status="passed",
            loss=float(loss.detach()),
            grad_norm=float(grad_norm),
            state_projection_update_max=change,
            decoded_shape=list(actual.shape),
            reload_max_abs_difference=error,
            max_allocated_cuda_gb=torch.cuda.max_memory_allocated() / 1e9,
            initial_checkpoint=str(initial),
            updated_checkpoint=str(checkpoint),
        )
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", result)
    print("SMOLVLA_WITNESS", json.dumps(result))


def training_config(spec, training, development):
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import DatasetConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=training.repo_id,
            root=Path(spec["train_sources"][0]["root"]),
            # Official config validation requires a declared eval fraction. The
            # factory below still returns our existing, session-disjoint views.
            eval_split=development.num_episodes
            / (training.num_episodes + development.num_episodes),
        ),
        policy=configured_policy(spec),
        output_dir=Path(spec["output_dir"]),
        job_name=spec["run_id"],
        seed=spec["optimization"]["seed"],
        batch_size=spec["optimization"]["batch_size"],
        num_workers=spec["optimization"]["num_workers"],
        steps=spec["budget"]["max_steps"],
        log_freq=50,
        save_freq=spec["budget"]["save_interval_steps"],
        eval_steps=spec["budget"]["eval_interval_steps"],
        env_eval_freq=0,
    )
    cfg.dataset.image_transforms.enable = False
    cfg.wandb.enable = False
    cfg.policy.push_to_hub = False
    return cfg


def train(spec):
    from accelerate import Accelerator
    import lerobot.scripts.lerobot_train as upstream

    output = Path(spec["output_dir"])
    root = output.parent
    if output.exists() or (root / "training-status.json").exists():
        raise FileExistsError(
            f"training already has results/state: {root}; use a new run ID and output"
        )
    torch.set_num_threads(4)
    training, development = make_views(spec)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "dataset-view.json", view_report(training, development, spec))
    subset = torch.utils.data.Subset(
        development, development.evaluation_indices(spec["evaluation"]["samples_per_episode"])
    )
    cfg = training_config(spec, training, development)
    started = time.monotonic()
    original_update, original_save, original_policy = (
        upstream.update_policy,
        upstream.save_checkpoint,
        upstream.make_policy,
    )

    def update(*args, **kwargs):
        if time.monotonic() - started >= spec["budget"]["max_wallclock_seconds"]:
            raise TimeoutError("SmolVLA frozen wall-clock training budget reached")
        return original_update(*args, **kwargs)

    def processors(policy_cfg, **kwargs):
        return make_processors(policy_cfg, training, spec.get("action_encoding", "absolute"))

    def make_policy(*args, **kwargs):
        policy = original_policy(*args, **kwargs)
        write_json(root / "trainable-parameters.json", parameter_report(policy))
        return policy

    def save(*args, **kwargs):
        original_save(*args, **kwargs)
        directory = Path(kwargs["checkpoint_dir"] if "checkpoint_dir" in kwargs else args[0])
        write_json(directory / "piper-spec.json", spec)

    result = {"status": "running", "hardware_io": False, "spec": spec}
    write_json(root / "training-status.json", result)
    try:
        with ExitStack() as stack:
            for name, replacement in [
                ("make_train_eval_datasets", lambda cfg: (training, subset)),
                ("make_pre_post_processors", processors),
                ("update_policy", update),
                ("save_checkpoint", save),
                ("make_policy", make_policy),
            ]:
                stack.enter_context(patch.object(upstream, name, replacement))
            accelerator = Accelerator(mixed_precision="bf16", step_scheduler_with_optimizer=False)
            upstream.train(cfg, accelerator=accelerator)
        result.update(status="complete", stop_reason="fixed_step_budget_completed")
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_json(root / "training-status.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "smoke", "train"])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.command == "train":
        train(spec)
    elif args.output is None:
        parser.error("prepare/smoke require --output")
    elif args.command == "smoke":
        smoke(spec, args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        training, development = make_views(spec)
        write_json(args.output / "data-view.json", view_report(training, development, spec))
        print(json.dumps({"train_frames": len(training), "development_frames": len(development)}))


if __name__ == "__main__":
    main()
