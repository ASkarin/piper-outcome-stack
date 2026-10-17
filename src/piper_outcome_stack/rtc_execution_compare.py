"""Three-way offline comparison on saved real observations; no closed-loop claims."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from piper_policy.smolvla import SmolVLAChunkPredictor, TASK_A
from piper_policy.execution_constraints import project_gripper, projection_stats
from .smolvla_training import write_json
from .smolvla_evaluate import action_errors
from .rtc_experiment import timed

METHODS = ("naive", "identity", "full-vjp")


def configure(predictor, method):
    predictor.configure_rtc(
        method != "naive", num_steps=5, guidance="identity" if method == "identity" else "full-vjp"
    )


def run(spec_path):
    spec = json.loads(spec_path.read_text())
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + spec["budget_seconds"]
    status = {
        "status": "running",
        "domain": "offline_recorded_observations",
        "hardware_io": False,
        "task_success_evaluated": False,
        "spec": spec,
    }
    write_json(root / "status.json", status)

    def budget():
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed offline comparison budget reached")

    try:
        torch.set_num_threads(4)
        predictor = SmolVLAChunkPredictor.from_checkpoint(spec["checkpoint"], task=TASK_A)
        versions = [p._version for p in predictor.model.parameters()]
        source = Path(spec["source_results"])
        files = sorted((source / "pairs").glob("*.npz"))
        if len(files) != 104:
            raise ValueError("expected the frozen104 real-observation pairs")
        delay = spec["delay_steps"]
        # Warm each method using the same real pair and same already-projected old prefix.
        first = np.load(files[0])
        previous = project_gripper(first["old_physical"], "zero")
        prefix = predictor.encode_execution_actions(previous)[25:]
        for method in METHODS:
            configure(predictor, method)
            for _ in range(5):
                budget()
                options = (
                    {}
                    if method == "naive"
                    else dict(
                        previous_model_actions=prefix, inference_delay=delay, overlap_steps=25
                    )
                )
                predictor.predict_chunk(
                    first["second_image"],
                    first["second_state"],
                    noise=first["new_noise"],
                    gripper_lower_bound="zero",
                    **options,
                )
        records = {
            m: {"raw": [], "execution": [], "model": [], "committed": [], "seconds": []}
            for m in METHODS
        }
        old_raw = []
        old_execution = []
        old_committed = []
        targets = []
        for i, file in enumerate(files):
            budget()
            item = np.load(file)
            previous = project_gripper(item["old_physical"], "zero")
            committed = predictor.encode_execution_actions(previous)
            # Rotation avoids always measuring one method first/last; weights/data/noise unchanged.
            order = METHODS[i % 3 :] + METHODS[: i % 3]
            for method in order:
                configure(predictor, method)
                options = (
                    {}
                    if method == "naive"
                    else dict(
                        previous_model_actions=committed[25:],
                        inference_delay=delay,
                        overlap_steps=25,
                    )
                )
                result, elapsed = timed(
                    predictor,
                    item["second_image"],
                    item["second_state"],
                    noise=item["new_noise"],
                    gripper_lower_bound="zero",
                    **options,
                )
                for key, value in [
                    ("raw", result.actions),
                    ("execution", result.execution_actions),
                    ("model", result.model_actions),
                    ("committed", result.committed_model_actions),
                    ("seconds", elapsed),
                ]:
                    records[method][key].append(value)
            old_raw.append(item["old_physical"])
            old_execution.append(previous)
            old_committed.append(committed)
            targets.append(item["targets"])
        gt = np.asarray(targets)
        valid = np.ones(gt.shape[:2], bool)
        previous = np.asarray(old_execution)
        arrays = {
            "targets": gt,
            "valid": valid,
            "old_raw": old_raw,
            "old_execution": previous,
            "old_committed": old_committed,
        }
        summary = {
            "pairs": 104,
            "delay_steps": delay,
            "num_steps": 5,
            "gripper_lower_bound": "zero",
            "checkpoint": spec["checkpoint"],
            "domain": "offline_recorded_observations",
            "closed_loop_or_success_evaluated": False,
            "actual_hardware_commands_sent": 0,
            "continuity_reference": "projected old model predictions; proxy, not GT",
            "identity_scope": "LeRobot-style g=e; same EXP mask, beta5 and seven-dimensional constraints as full VJP; not unmodified upstream defaults",
            "methods": {},
        }
        for method in METHODS:
            rec = {k: np.asarray(v) for k, v in records[method].items()}
            for k, v in rec.items():
                arrays[method + "_" + k] = v
            raw, execute = rec["raw"], rec["execution"]
            jump = execute[:, delay] - previous[:, 25 + delay - 1]
            raw_jump = raw[:, delay] - np.asarray(old_raw)[:, 25 + delay - 1]
            summary["methods"][method] = {
                "raw_gt_full": action_errors(raw, gt, valid),
                "raw_gt_available_suffix": action_errors(
                    raw[:, delay:], gt[:, delay:], valid[:, delay:]
                ),
                "execution_overlap_proxy": action_errors(
                    execute[:, :25], previous[:, 25:], valid[:, :25]
                ),
                "all_predicted_positions_projection": projection_stats(raw, execute),
                "first_available_projection": projection_stats(raw[:, delay], execute[:, delay]),
                "raw_boundary_joint_p50_p95_max_deg": np.percentile(
                    np.rad2deg(abs(raw_jump[:, :6])).max(1), [50, 95, 100]
                ).tolist(),
                "execution_boundary_joint_p50_p95_max_deg": np.percentile(
                    np.rad2deg(abs(jump[:, :6])).max(1), [50, 95, 100]
                ).tolist(),
                "raw_boundary_gripper_p50_p95_max_mm": np.percentile(
                    abs(raw_jump[:, 6]) * 1000, [50, 95, 100]
                ).tolist(),
                "execution_boundary_gripper_p50_p95_max_mm": np.percentile(
                    abs(jump[:, 6]) * 1000, [50, 95, 100]
                ).tolist(),
                "inference_ms_p50_p95_p99": (
                    np.percentile(rec["seconds"], [50, 95, 99]) * 1000
                ).tolist(),
                "inference_above_common_delay_count": int((rec["seconds"] > delay * 0.02).sum()),
                "execution_negative_gripper_positions": int((execute[:, :, 6] < 0).sum()),
                "execution_above_120mm_positions": int((execute[:, :, 6] > 0.12).sum()),
            }
        np.savez_compressed(root / "predictions.npz", **arrays)
        write_json(root / "summary.json", summary)
        write_json(
            root / "sources.json",
            {
                "parent_results": str(source),
                "pairs": json.loads((source / "pairs.json").read_text()),
                "old_predictions": "frozen parent predictions; projected and re-encoded, never used as GT",
                "noise_and_images": "same per-pair raw NPZ as parent; no synthetic input or new labels",
            },
        )
        if any(p.requires_grad or p.grad is not None for p in predictor.model.parameters()):
            raise ValueError("parameter gradients acquired during inference")
        if versions != [p._version for p in predictor.model.parameters()]:
            raise ValueError("parameter versions changed")
        status.update(status="complete", parameter_versions_unchanged=True)
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        status["wall_seconds"] = time.monotonic() - start
        write_json(root / "status.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    run(parser.parse_args().spec)


if __name__ == "__main__":
    main()
