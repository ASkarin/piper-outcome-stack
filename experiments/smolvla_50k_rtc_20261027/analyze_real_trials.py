"""Read recorded RTC trials only; emits descriptive metrics without device I/O."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np


root = Path(sys.argv[1])
keys = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
trials = []
traces = {}


def percentiles(values):
    values = np.asarray(values, dtype=float)
    return np.percentile(values, [50, 95, 100]).tolist() if values.size else None


for folder in sorted((root / "trials").iterdir()):
    if not folder.is_dir():
        continue
    path = folder / "trial.json"
    raw = json.loads(path.read_text())
    label_path = folder / "trial.outcome.json"
    outcome = json.loads(label_path.read_text()) if label_path.is_file() else None
    exit_path = folder / "exit-code.json"
    exit_code = json.loads(exit_path.read_text())["exit_code"] if exit_path.is_file() else None
    rows = [row for row in raw["rows"] if row["phase"] == "policy_sent"]
    if len(rows) != raw["policy_actions_sent"]:
        raise ValueError(f"{folder.name}: action count does not match recorded rows")
    target = np.array([[row["target"][key] for key in keys] for row in rows])
    observed = np.array([[row["observation"][key] for key in keys] for row in rows])
    times = np.array([row["input_telemetry"]["observed_monotonic_s"] for row in rows])
    execution = [row["prediction"]["asynchronous_execution"] for row in rows]
    ticks = np.array([item["control_tick"] for item in execution])
    origins = np.array([item["observation_tick"] for item in execution])
    pair = np.diff(ticks) == 1
    triple = pair[:-1] & pair[1:]
    seams = np.diff(origins) != 0
    if not np.isfinite(target).all() or not np.isfinite(observed).all():
        raise ValueError(f"{folder.name}: nonfinite recorded actions/state")
    goal_deg = np.rad2deg(target[:, :6])
    measured_deg = np.rad2deg(observed[:, :6])
    goal_d2 = np.diff(goal_deg, n=2, axis=0)
    measured_d2 = np.diff(measured_deg, n=2, axis=0)
    steps = np.max(np.abs(np.diff(goal_deg, axis=0)), axis=1)
    publications = [row for row in rows if row["prediction"]["inference_performed"]]
    rtc_publications = [
        row
        for row in publications
        if row["prediction"]["asynchronous_execution"]
        .get("rtc_request", {})
        .get("prefix_model_actions")
        is not None
    ]
    latency = [
        row["prediction"]["asynchronous_execution"]["inference_seconds"] * 1000
        for row in publications
    ]
    sdk_results = dict(Counter(row["sdk"]["result"] for row in rows))
    rejected = [row for row in raw["rows"] if row.get("sdk", {}).get("rejection_reason")]
    frames_path = folder / "trial-rgb" / "frames.jsonl"
    frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
    config = raw["execution"]
    entry = {
        "trial": folder.name,
        "source": str(path),
        "ended_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "first_observation_unix_s": rows[0]["input_telemetry"]["feedback"]["timestamp_s"],
        "model": Path(raw["checkpoint"]).name,
        "checkpoint": raw["checkpoint"],
        "task": config["task"],
        "position": raw["position"],
        "task_outcome": None if outcome is None else outcome["task_outcome"],
        "outcome": outcome,
        "program_status": raw["status"],
        "exit_code": exit_code,
        "hold_confirmed": raw["hold_confirmed"],
        "stop_outcome": raw["stop_outcome"],
        "error": raw.get("error"),
        "end_reason": raw["end_reason"],
        "execution": {
            key: config[key]
            for key in (
                "rtc",
                "rtc_guidance",
                "rtc_num_steps",
                "lowpass",
                "n_action_steps",
                "seed",
                "action_padding",
                "target_transform",
            )
        },
        "policy_actions_sent": len(rows),
        "policy_segment_seconds": float(times[-1] - times[0]),
        "consecutive_pairs": int(pair.sum()),
        "consecutive_triples": int(triple.sum()),
        "logical_tick_gaps": int((~pair).sum()),
        "target_d2_rms_deg_per_tick2": float(np.sqrt(np.mean(goal_d2[triple] ** 2))),
        "observed_d2_rms_deg_per_tick2": float(np.sqrt(np.mean(measured_d2[triple] ** 2))),
        "seam_max_axis_step_deg_p50_p95_max": percentiles(steps[pair & seams]),
        "within_chunk_max_axis_step_deg_p50_p95_max": percentiles(steps[pair & ~seams]),
        "inferences_published": len(publications),
        "rtc_prefix_publications": len(rtc_publications),
        "inference_ms_p50_p95_max": percentiles(latency),
        "cycle_work_ms_p50_p95_max": percentiles([row["cycle_work_s"] * 1000 for row in rows]),
        "cycle_work_over_20ms": sum(row["cycle_work_s"] > 0.02 for row in rows),
        "observed_interval_ms_p50_p95_max": percentiles(np.diff(times) * 1000),
        "target_feedback_joint_abs_deg_p50_p95_max": percentiles(np.abs(goal_deg - measured_deg)),
        "gripper_target_feedback_abs_mm_p50_p95_max": percentiles(
            np.abs(target[:, 6] - observed[:, 6]) * 1000
        ),
        "sdk_results": sdk_results,
        "rejection_rows": len(rejected),
        "rgb": raw["rgb_recording"],
        "rgb_index_entries": len(frames),
        "first_rgb": frames[0],
        "last_rgb": frames[-1],
    }
    trials.append(entry)
    traces[folder.name] = {
        "time_since_first_action_s": (times - times[0]).tolist(),
        "control_tick": ticks.tolist(),
        "chunk_observation_tick": origins.tolist(),
        "target_rad_m": target.tolist(),
        "observed_rad_m": observed.tolist(),
    }

trials.sort(key=lambda trial: trial["first_observation_unix_s"])
groups = {}
for model in ("original", "flat"):
    selected = [trial for trial in trials if trial["model"] == model]
    groups[model] = {
        "outcomes": dict(Counter(trial["task_outcome"] for trial in selected)),
        "trials": len(selected),
        "actions": sum(trial["policy_actions_sent"] for trial in selected),
        "cells": dict(Counter((trial["task"] + " | " + trial["position"]) for trial in selected)),
    }

common_success = {}
for model in ("original", "flat"):
    selected = [
        trial
        for trial in trials
        if trial["model"] == model
        and trial["task"] == "把绿色方块放到左区。"
        and trial["position"] == "B-layout1"
        and trial["task_outcome"] == "success"
    ]
    common_success[model] = {
        "n": len(selected),
        "trials": [trial["trial"] for trial in selected],
        "means": {
            key: float(np.mean([trial[key] for trial in selected]))
            for key in ("target_d2_rms_deg_per_tick2", "observed_d2_rms_deg_per_tick2")
        },
        "seam_step_p95_deg_mean": float(
            np.mean([trial["seam_max_axis_step_deg_p50_p95_max"][1] for trial in selected])
        ),
        "inference_ms_p50_mean": float(
            np.mean([trial["inference_ms_p50_p95_max"][0] for trial in selected])
        ),
        "segment_seconds_mean": float(
            np.mean([trial["policy_segment_seconds"] for trial in selected])
        ),
    }

result = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source_root": str(root),
    "scope": "Recorded real-robot development trials, descriptive analysis only; no training, inference or hardware execution",
    "metric_definition": "Joint second differences in degrees over consecutive logical 20ms control ticks, calculated separately for dispatched targets and raw recorded feedback. Includes task motion and feedback sampling effects; not calibrated physical acceleration/jerk or pure jitter.",
    "common_success_caution": "This subset describes completed successful green-left layout1 rollouts only; it is not a full success-rate or causal model comparison. The operator reported a background change after the first failure.",
    "trials": trials,
    "groups": groups,
    "common_successful_green_left_layout1": common_success,
    "flat_vs_original_change_percent_in_common_success": {
        key: (common_success["flat"]["means"][key] / common_success["original"]["means"][key] - 1)
        * 100
        for key in common_success["flat"]["means"]
    },
    "traces": traces,
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
