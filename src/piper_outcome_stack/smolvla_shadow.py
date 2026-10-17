"""Post-capture shadow inference on the controller; no robot or command sender."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from piper_policy.smolvla import SmolVLAChunkPredictor, TASK_A
from piper_policy.execution_constraints import projection_stats

METHODS = ("naive", "identity", "full-vjp")


def compare_pair(predictor, pair, *, seed, delay=19):
    rng = np.random.default_rng(seed)
    old_noise = rng.standard_normal((1, 50, 32)).astype(np.float32)
    new_noise = rng.standard_normal((1, 50, 32)).astype(np.float32)
    predictor.configure_rtc(False, num_steps=5)
    old = predictor.predict_chunk(
        pair["first_image"], pair["first_state"], noise=old_noise, gripper_lower_bound="zero"
    )
    result = {
        "old_raw": old.actions,
        "old_execution": old.execution_actions,
        "old_committed": old.committed_model_actions,
        "old_noise": old_noise,
        "new_noise": new_noise,
        "first_image": pair["first_image"],
        "first_state": pair["first_state"],
        "second_image": pair["second_image"],
        "second_state": pair["second_state"],
    }
    for method in METHODS:
        predictor.configure_rtc(
            method != "naive",
            num_steps=5,
            guidance="identity" if method == "identity" else "full-vjp",
        )
        opts = (
            {}
            if method == "naive"
            else {
                "previous_model_actions": old.committed_model_actions[25:],
                "inference_delay": delay,
                "overlap_steps": 25,
            }
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        chunk = predictor.predict_chunk(
            pair["second_image"],
            pair["second_state"],
            noise=new_noise,
            gripper_lower_bound="zero",
            **opts,
        )
        torch.cuda.synchronize()
        result[method + "_seconds"] = time.perf_counter() - start
        result[method + "_raw"] = chunk.actions
        result[method + "_execution"] = chunk.execution_actions
        result[method + "_model"] = chunk.model_actions
        result[method + "_committed"] = chunk.committed_model_actions
    return result


def run(args):
    if args.mode == "captured":
        capture = json.loads((args.input / "summary.json").read_text())
        if capture.get("status") != "captured" or capture.get("motion_commands_sent") != 0:
            raise ValueError("shadow inference requires a completed read-only capture")
        files = sorted((args.input / "pairs").glob("*.npz"))
        if len(files) != capture["pairs"]:
            raise ValueError("captured pair count mismatch")
        source_scope = "new_live_sensor_capture_then_offline_inference"
    else:
        files = [args.input]
        source_scope = "recorded_input_preflight_not_live"
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "started",
        "mode": args.mode,
        "source_scope": source_scope,
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "vlm_assets": str(args.vlm_assets),
        "hardware_io": False,
        "motion_commands_sent": 0,
        "motor_enable_executed": False,
        "ground_truth_actions_available": False,
        "task_success_evaluated": False,
        "rtc_prefix_scope": "counterfactual projected plan; these actions were never executed",
        "common_delay_steps": 19,
        "num_steps": 5,
        "methods": {},
    }
    start = time.monotonic()
    try:
        torch.set_num_threads(4)
        predictor = SmolVLAChunkPredictor.from_checkpoint(
            args.checkpoint, task=TASK_A, vlm_assets=args.vlm_assets
        )
        versions = [p._version for p in predictor.model.parameters()]
        first = np.load(files[0])
        compare_pair(predictor, first, seed=1000000)  # warmup; not metrics
        safety = None
        if args.safety is not None:
            from lerobot_robot_outcome_piper.safety import load_motion_safety
            from lerobot_robot_outcome_piper.execution_constraints import check_execution_target

            safety = load_motion_safety(args.safety)
        report["safety_source"] = str(args.safety) if args.safety is not None else None
        stored = []
        bounds = {method: [] for method in METHODS}
        for i, file in enumerate(files):
            result = compare_pair(predictor, np.load(file), seed=42 + i)
            np.savez_compressed(args.output / f"pair-{i:03d}.npz", **result)
            stored.append(result)
            if safety is not None:
                for method in METHODS:
                    try:
                        check_execution_target(
                            result["second_state"], result[method + "_execution"][19], safety
                        )
                        bounds[method].append({"pair": i, "status": "within_recorded_state_checks"})
                    except ValueError as exc:
                        bounds[method].append({"pair": i, "status": "rejected", "reason": str(exc)})
        for method in METHODS:
            raw = np.stack([x[method + "_raw"] for x in stored])
            targets = np.stack([x[method + "_execution"] for x in stored])
            seconds = np.asarray([x[method + "_seconds"] for x in stored])
            report["methods"][method] = {
                "pairs": len(stored),
                "prospective_recorded_state_checks": bounds[method],
                "checks_scope": "index19 against captured state; not actual future state or motion approval",
                "inference_ms_p50_p95_max": (np.percentile(seconds, [50, 95, 100]) * 1000).tolist(),
                "calls_above_500ms": int((seconds > 0.5).sum()),
                "prospective_projection": projection_stats(raw, targets),
                "prospective_above_120mm": int((targets[:, :, 6] > 0.12).sum()),
                "deadline_note": "inference only; not real-time50Hz capture/command validation",
            }
        if versions != [p._version for p in predictor.model.parameters()] or any(
            p.grad is not None for p in predictor.model.parameters()
        ):
            raise ValueError("shadow inference changed parameters or acquired parameter gradients")
        report.update(
            status="complete", gpu=torch.cuda.get_device_name(), parameter_versions_unchanged=True
        )
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - start
        (args.output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    print(args.output / "summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "captured"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--safety",
        type=Path,
        help="existing controller safety file; offline candidate diagnostics only",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
