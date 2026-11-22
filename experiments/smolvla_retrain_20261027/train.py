# ruff: noqa: E402
# This archived runner binds its frozen source directory before importing policy code.
"""Fresh pretrained SmolVLA pair: native objective versus fixed flat temporal FM."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "source"))

import numpy as np
import torch
from torch.utils.data import Subset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from piper_outcome_stack.smolvla_training import (
    make_views,
    training_config,
    parameter_report,
    write_json,
)
from piper_outcome_stack.training_windows import CompleteWindows, sampler_class
from piper_policy.smolvla_gripper import make_processors
from piper_policy.smolvla import IMAGE_FEATURE, SmolVLAChunkPredictor
from temporal_loss import ORIGINAL_FORWARD, objective, training_forward


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def initial_weights(model, source):
    from safetensors import safe_open

    actual = model.state_dict()
    matched = 0
    with safe_open(
        str(Path(source) / "model.safetensors"), framework="pt", device="cpu"
    ) as tensors:
        for name in tensors.keys():
            require(name in actual, f"pretrained tensor not loaded: {name}")
            expected = tensors.get_tensor(name)
            require(
                torch.equal(actual[name].detach().cpu(), expected.to(actual[name].dtype)),
                f"initial tensor differs from official pretrained source: {name}",
            )
            matched += 1
    require(matched > 0, "empty pretrained source")
    return matched


def objective_witness(model, batch, weight):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(123)
        actions = model.prepare_action(batch)
        noise = model.model.sample_noise(actions.shape, actions.device)
        flow_time = model.model.sample_time(actions.shape[0], actions.device)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            native, _ = ORIGINAL_FORWARD(model, batch, noise, flow_time)
            tail_gradient = torch.autograd.grad(native, model.model.action_out_proj.weight)[0]
            require(
                torch.count_nonzero(tail_gradient[7:]) == 0,
                "native flow loss supervises structural output padding",
            )
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            total, _, base, _, auxiliary = objective(
                model, batch, noise, flow_time, weight=weight, weighting="flat"
            )
            require(
                torch.allclose(native.detach(), base.detach(), atol=1e-6, rtol=1e-6),
                "flat wrapper changed the original flow term",
            )
            grad = torch.autograd.grad(auxiliary, model.model.action_out_proj.weight)[0]
            require(
                torch.isfinite(grad).all()
                and grad[:6].abs().max() > 0
                and torch.count_nonzero(grad[6:]) == 0,
                "flat auxiliary gradient reaches gripper/padding or is absent",
            )
        return dict(
            native_flow=float(native.detach()),
            flat_base=float(base.detach()),
            flat_auxiliary=float(auxiliary.detach()),
            flat_total=float(total.detach()),
            native_output_tail_gradient_zero=True,
            auxiliary_only_first_six_outputs=True,
        )


def padding_and_reload_witness(spec, checkpoint, training, output, live_model, pre, post):
    predictor = SmolVLAChunkPredictor.from_checkpoint(checkpoint, task=training[0]["task"])
    item = training[0]
    image = item[IMAGE_FEATURE].permute(1, 2, 0).numpy()
    state = item["observation.state"].numpy()
    epsilon = np.random.default_rng(123).standard_normal((1, 50, 32)).astype(np.float32)
    live = SmolVLAChunkPredictor(live_model.eval(), pre, post, item["task"])
    expected_prediction = live.predict_chunk(image, state, noise=epsilon).actions
    flow = predictor.model.model
    original = flow.denoise_step
    checks = []

    def checked(prefix_pad_masks, past_key_values, x_t, timestep):
        expected = timestep[:, None, None] * torch.as_tensor(epsilon, device=x_t.device)[..., 7:]
        error = float((x_t[..., 7:] - expected).abs().max())
        require(error == 0, f"inference padding left the t*epsilon path: {error}")
        checks.append(error)
        return original(prefix_pad_masks, past_key_values, x_t, timestep)

    flow.denoise_step = checked
    try:
        first = predictor.predict_chunk(image, state, noise=epsilon).actions
    finally:
        flow.denoise_step = original
    second = predictor.predict_chunk(image, state, noise=epsilon).actions
    reload_error = float(abs(first - expected_prediction).max())
    repeat_error = float(abs(first - second).max())
    require(
        first.shape == (50, 7)
        and np.isfinite(first).all()
        and reload_error <= 1e-5
        and repeat_error <= 1e-5,
        "saved checkpoint inference/repeated explicit-noise prediction differs",
    )
    require(
        len(checks) == predictor.model.config.num_steps == 10,
        "padding checks did not cover ten steps",
    )
    np.savez_compressed(
        output / "witness-predictions.npz", live=expected_prediction, first=first, second=second
    )
    return dict(
        shape=list(first.shape),
        padding_steps_checked=len(checks),
        padding_max_error=max(checks),
        checkpoint_reload_prediction_difference=reload_error,
        repeated_explicit_noise_prediction_difference=repeat_error,
    )


def run(spec):
    from accelerate import Accelerator
    import lerobot.scripts.lerobot_train as upstream

    root = Path(spec["run_root"])
    output = Path(spec["output_dir"])
    require(
        not output.exists() and not (root / "training-status.json").exists(),
        f"run already started: {root}",
    )
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    raw_training, development = make_views(spec)
    # The archived B run uses 239/39 demonstrations in eight task-layout cells.
    counts = spec.get("demonstrations", {"train": 239, "development": 39, "task_layout_cells": 8})
    training = CompleteWindows(raw_training, counts["train"], counts["task_layout_cells"])
    require(
        len({(x["source_session"], x["source_episode"]) for x in development.lineage})
        == counts["development"],
        "development original-demo count differs",
    )
    subset = Subset(development, development.evaluation_indices(4))
    cfg = training_config(spec, training, development)
    cfg.resume = False
    cfg.policy.drop_n_last_frames = 0
    require(
        cfg.policy.max_state_dim == cfg.policy.max_action_dim == 32
        and cfg.policy.action_feature.shape == (7,),
        "physical/padded dimensions differ",
    )
    cfg.policy.num_steps = 10
    original_policy, original_update, original_save = (
        upstream.make_policy,
        upstream.update_policy,
        upstream.save_checkpoint,
    )
    created_models = []
    result = dict(
        status="starting",
        model=spec["arm"],
        start_step=0,
        target_step=cfg.steps,
        updates=0,
        source_checkpoint=spec["source_checkpoint"],
        source_revision=spec["source_revision"],
        fresh_optimizer=True,
        resume=False,
        hardware_io=False,
        dataset=training.summary,
    )
    write_json(
        root / "dataset-view.json",
        dict(
            train=training.summary,
            train_state_statistics=raw_training.meta.stats["observation.state"],
            train_action_statistics=raw_training.meta.stats["action"],
            development_demos=counts["development"],
            development_segments=development.num_episodes,
            data_manifest_sha256=spec["data_manifest_sha256"],
        ),
    )
    started = time.monotonic()

    def save_status():
        result["wall_seconds"] = time.monotonic() - started
        write_json(root / "training-status.json", result)

    def make_policy(*args, **kwargs):
        model = original_policy(*args, **kwargs)
        created_models.append(model)
        count = initial_weights(model, spec["source_checkpoint"])
        write_json(
            root / "initialization.json",
            dict(
                pretrained_tensors_equal=count,
                step=0,
                source_checkpoint=spec["source_checkpoint"],
                source_revision=spec["source_revision"],
                inherited_piper_weights=False,
                inherited_optimizer=False,
                parameters=parameter_report(model),
            ),
        )
        return model

    def processors(policy_cfg, **kwargs):
        return make_processors(
            policy_cfg,
            raw_training,
            "absolute",
            joint_state_std_floor_rad=spec.get("joint_state_std_floor_rad"),
        )

    def update(*args, **kwargs):
        if time.monotonic() - started > spec["budget"]["max_wallclock_seconds"]:
            raise TimeoutError("fixed training wall-clock ceiling reached")
        model, batch, optimizer = args[1], args[2], args[3]
        before = None
        if result["updates"] == 0:
            require(not optimizer.state, "optimizer moments were inherited")
            actions, state = model.prepare_action(batch), model.prepare_state(batch)
            require(
                actions.shape[-1] == state.shape[-1] == 32
                and torch.count_nonzero(actions[..., 7:])
                == torch.count_nonzero(state[..., 7:])
                == 0,
                "nonzero structural state/action padding",
            )
            result["first_batch_indices"] = batch["index"].detach().cpu().tolist()
            result["training_padding"] = dict(
                real_dim=7,
                padded_dim=32,
                zero_state_tail=True,
                zero_action_target_tail=True,
                loss_dimensions=7,
            )
            if spec["witness"]:
                result["objective_witness"] = objective_witness(
                    model, batch, spec["temporal_weight"]
                )
            before = model.model.state_proj.weight.detach().clone()
        updated = original_update(*args, **kwargs)
        result["updates"] += 1
        require(
            all(np.isfinite(float(v)) for v in updated[1].values() if isinstance(v, (int, float))),
            "nonfinite training metric",
        )
        if before is not None:
            change = float((model.model.state_proj.weight.detach() - before).abs().max())
            require(np.isfinite(change) and change > 0, "state projection did not update")
            result["first_state_projection_update_max"] = change
            steps = [float(v["step"]) for v in optimizer.state.values() if "step" in v]
            require(steps and min(steps) == max(steps) == 1, "first optimizer step is not one")
        if result["updates"] == 1 or result["updates"] % 50 == 0:
            with (root / "objective.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=result["updates"], **updated[1])) + "\n")
            result.update(
                status="running",
                last_step=result["updates"],
                learning_rates=[g["lr"] for g in optimizer.param_groups],
            )
            save_status()
        return updated

    def save(*args, **kwargs):
        original_save(*args, **kwargs)
        directory = Path(kwargs["checkpoint_dir"] if "checkpoint_dir" in kwargs else args[0])
        write_json(directory / "piper-spec.json", spec)

    save_status()
    try:
        with ExitStack() as stack:
            if spec["arm"] == "flat":
                stack.enter_context(
                    patch.object(
                        SmolVLAPolicy, "forward", training_forward(spec["temporal_weight"], "flat")
                    )
                )
            for name, replacement in [
                ("make_train_eval_datasets", lambda config: (training, subset)),
                ("make_pre_post_processors", processors),
                ("make_policy", make_policy),
                ("update_policy", update),
                ("save_checkpoint", save),
                ("EpisodeAwareSampler", sampler_class(training)),
            ]:
                stack.enter_context(patch.object(upstream, name, replacement))
            upstream.train(
                cfg,
                accelerator=Accelerator(
                    mixed_precision="bf16", step_scheduler_with_optimizer=False
                ),
            )
        require(result["updates"] == cfg.steps, "fixed update budget not completed")
        checkpoint = output / "checkpoints" / f"{cfg.steps:06d}" / "pretrained_model"
        if spec["witness"]:
            pre, post = processors(cfg.policy)
            result["witness"] = padding_and_reload_witness(
                spec, checkpoint, training, root, created_models[0], pre, post
            )
        else:
            result["training_complete"] = True
            result["status"] = "evaluating"
            save_status()
            from piper_outcome_stack.smolvla_evaluate import evaluate

            evaluate(spec, checkpoint, root / "evaluation", "smolvla")
        result.update(status="complete", last_step=cfg.steps, final_checkpoint=str(checkpoint))
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()))
