"""Offline physical-unit action errors on the frozen development view."""

import argparse
from pathlib import Path
import json
import time

import numpy as np
import torch

from piper_policy.smolvla import IMAGE_FEATURE, SmolVLAChunkPredictor
from .smolvla_training import make_views, validate_task, write_json


def action_errors(predictions, targets, valid):
    error = np.asarray(predictions) - np.asarray(targets)
    valid = np.asarray(valid, dtype=bool)
    if error.ndim != 3 or error.shape[-1] != 7 or valid.shape != error.shape[:2]:
        raise ValueError(
            "action/error masks must have matching sample, horizon and seven-axis dimensions"
        )
    if not np.isfinite(error).all() or not valid[:, 0].all():
        raise ValueError("invalid action values or missing first-step supervision")
    first = abs(error[:, 0])
    chunk = abs(error)[valid]
    return {
        "samples": len(error),
        "valid_horizon_targets": int(valid.sum()),
        "first_joint_mae_deg": np.rad2deg(first[:, :6].mean(0)).tolist(),
        "first_gripper_mae_mm": float(first[:, 6].mean() * 1000),
        "chunk_joint_mae_deg": np.rad2deg(chunk[:, :6].mean(0)).tolist(),
        "chunk_gripper_mae_mm": float(chunk[:, 6].mean() * 1000),
    }


def command_envelope_counts(states, predictions, limits):
    target = predictions[:, 0]
    joints = (target[:, :6] < limits["joint_lower"]) | (target[:, :6] > limits["joint_upper"])
    grip = (target[:, 6] < limits["gripper_lower"]) | (target[:, 6] > limits["gripper_upper"])
    step = (abs(target[:, :6] - states[:, :6]) > limits["max_joint_step"]).any(1)
    step |= abs(target[:, 6] - states[:, 6]) > limits["max_gripper_step"]
    return {
        "samples": len(target),
        "joint_box_rejections": int(joints.any(1).sum()),
        "gripper_box_rejections": int(grip.sum()),
        "state_delta_rejections": int(step.sum()),
        "any_rejection": int((joints.any(1) | grip | step).sum()),
        "scope": "recorded numerical envelope only; not Cartesian/workspace or hardware acceptance",
    }


def grouped_errors(predictions, targets, valid, provenance):
    """Report each original demonstration once, then macro-average within task/layout."""
    groups = {}
    for index, row in enumerate(provenance):
        lineage = row["lineage"]
        key = (
            lineage["source_session"],
            lineage["source_episode"],
            row["task"],
            lineage["layout"],
        )
        groups.setdefault(key, []).append(index)
    sources = []
    for (session, episode, task, layout), ids in groups.items():
        sources.append(
            {
                "source_session": session,
                "source_episode": episode,
                "task": task,
                "layout": layout,
                **action_errors(predictions[ids], targets[ids], valid[ids]),
            }
        )
    macro = []
    metrics = (
        "first_joint_mae_deg",
        "first_gripper_mae_mm",
        "chunk_joint_mae_deg",
        "chunk_gripper_mae_mm",
    )
    for task, layout in sorted({(r["task"], r["layout"]) for r in sources}):
        rows = [r for r in sources if (r["task"], r["layout"]) == (task, layout)]
        macro.append(
            {
                "task": task,
                "layout": layout,
                "demonstrations": len(rows),
                **{k: np.asarray([r[k] for r in rows]).mean(axis=0).tolist() for k in metrics},
            }
        )
    return {"per_source": sources, "macro_task_layout": macro}


def evaluate(spec, checkpoint, output, policy_type):
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "started",
        "evaluation_type": "offline_action_imitation",
        "hardware_io": False,
        "policy_type": policy_type,
        "checkpoint": str(checkpoint),
        "task_success_evaluated": False,
        "temporal_ensemble": False,
    }
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        _, development = make_views(spec)
        indices = development.evaluation_indices(spec["evaluation"]["samples_per_episode"])
        first = development[indices[0]]
        first_task = validate_task(spec, first["task"])
        if policy_type == "smolvla":
            predictor = SmolVLAChunkPredictor.from_checkpoint(checkpoint, task=first_task)
        else:
            from piper_policy.policy_execution import ACTChunkPredictor

            predictor = ACTChunkPredictor.from_checkpoint(checkpoint)
        first = development[indices[0]]
        rgb = first[IMAGE_FEATURE].permute(1, 2, 0).numpy()
        state = first["observation.state"].numpy()
        for _ in range(5):
            predictor.predict(rgb, state)
        predictions, targets, valid, states, episodes, seconds = [], [], [], [], [], []
        provenance = []
        for index in indices:
            item = development[index]
            rgb = item[IMAGE_FEATURE].permute(1, 2, 0).numpy()
            state = item["observation.state"].numpy()
            task = validate_task(spec, item["task"])
            if policy_type == "smolvla":
                predictor.task = task
            torch.manual_seed(spec["optimization"]["seed"] + index)
            torch.cuda.synchronize()
            before = time.perf_counter()
            prediction = predictor.predict(rgb, state)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - before)
            predictions.append(prediction)
            targets.append(item["action"].numpy())
            valid.append(~item["action_is_pad"].numpy())
            states.append(state)
            episode = int(item["episode_index"])
            episodes.append(episode)
            part = int(np.searchsorted(development.part_ends, index, side="right"))
            local = index - (development.part_ends[part - 1] if part else 0)
            provenance.append(
                {
                    "task": task,
                    "retained_index": index,
                    "episode": episode,
                    "source_frame_index": int(development.parts[part][1][local]),
                    "source": spec["development_sources"][part],
                    "lineage": development.lineage[episode],
                }
            )
        predictions, targets, valid, states, episodes = map(
            np.asarray, [predictions, targets, valid, states, episodes]
        )
        per_episode = []
        for episode in sorted(set(episodes.tolist())):
            mask = episodes == episode
            per_episode.append(
                {"episode": episode, **action_errors(predictions[mask], targets[mask], valid[mask])}
            )
        hold = np.repeat(states[:, None, :], predictions.shape[1], axis=1)
        if "tasks" in spec["input_contract"]:
            result["grouped_metrics"] = grouped_errors(predictions, targets, valid, provenance)
        result.update(
            status="complete",
            metrics=action_errors(predictions, targets, valid),
            per_episode=per_episode,
            hold_reference=action_errors(hold, targets, valid),
            command_envelope=command_envelope_counts(
                states, predictions, spec["evaluation"]["command_limits"]
            ),
            inference_ms_p50_p95_p99=(np.percentile(seconds, [50, 95, 99]) * 1000).tolist(),
            inference_device=torch.cuda.get_device_name(),
            sampling=spec["evaluation"],
            interpretation="sampled development imitation errors, not closed-loop or real-robot success",
        )
        np.savez_compressed(
            output / "predictions.npz",
            predictions=predictions,
            targets=targets,
            valid=valid,
            states=states,
            episodes=episodes,
            retained_indices=np.array(indices),
            inference_seconds=np.asarray(seconds),
        )
        write_json(output / "samples.json", provenance)
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", result)
    print(json.dumps(result["metrics"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-type", choices=["smolvla", "act"], default="smolvla")
    args = parser.parse_args()
    evaluate(json.loads(args.spec.read_text()), args.checkpoint, args.output, args.policy_type)


if __name__ == "__main__":
    main()
