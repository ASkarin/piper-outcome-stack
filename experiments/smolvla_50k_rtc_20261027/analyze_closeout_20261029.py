"""Read existing trial files; summarize the operator-designated October 29 batch."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import sys

import numpy as np

root = Path(sys.argv[1])
day = "2026-10-29"
local_tz = timezone(timedelta(hours=8))
keys = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
spec = json.loads((root / "spec.json").read_text())
expected_tasks = {value: key for key, value in spec["tasks"].items()}


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return np.percentile(values, [50, 95, 99, 100]).tolist() if values.size else None


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
    return [max(0, center - half), min(1, center + half)]


trials = []
commands = {}
settings = []
for folder in sorted((root / "trials").iterdir()):
    if not folder.is_dir():
        continue
    path = folder / "trial.json"
    metadata = path if path.exists() else folder / "command.json"
    ended = datetime.fromtimestamp(metadata.stat().st_mtime, local_tz)
    if ended.date().isoformat() != day:
        continue
    raw = json.loads(path.read_text())
    outcome = json.loads((folder / "trial.outcome.json").read_text())
    exit_code = json.loads((folder / "exit-code.json").read_text())["exit_code"]
    command = json.loads((folder / "command.json").read_text())
    config = raw["execution"]
    model = Path(raw["checkpoint"]).name
    task = expected_tasks[config["task"]]
    assert model in spec["models"]
    assert command[command.index("--task") + 1] == config["task"]
    assert command[command.index("--checkpoint") + 1] == raw["checkpoint"]
    assert command[command.index("--position") + 1] == raw["position"]
    assert str(root / "piper_policy_trial.py") in command
    assert raw["mode"] == "run"
    assert outcome["program_status"] == raw["status"]
    assert outcome["stop_outcome"] == raw["stop_outcome"]
    assert outcome["policy_actions_sent"] == raw["policy_actions_sent"]
    assert outcome["task_outcome"] in ("success", "failure", "invalid")
    if outcome["task_outcome"] == "invalid":
        assert outcome["note"], folder.name
    rows = [r for r in raw["rows"] if r["phase"] == "policy_sent"]
    assert len(rows) == raw["policy_actions_sent"]
    times = np.array([r["input_telemetry"]["observed_monotonic_s"] for r in rows])
    first_unix = rows[0]["input_telemetry"]["feedback"]["timestamp_s"]
    assert datetime.fromtimestamp(first_unix, local_tz).date().isoformat() == day
    target = np.array([[r["target"][key] for key in keys] for r in rows])
    observed = np.array([[r["observation"][key] for key in keys] for r in rows])
    assert np.isfinite(target).all() and np.isfinite(observed).all()
    execution = [r["prediction"]["asynchronous_execution"] for r in rows]
    ticks = np.array([e["control_tick"] for e in execution])
    origins = np.array([e["observation_tick"] for e in execution])
    consecutive = np.diff(ticks) == 1
    triples = consecutive[:-1] & consecutive[1:]
    seams = np.diff(origins) != 0
    target_deg = np.rad2deg(target[:, :6])
    observed_deg = np.rad2deg(observed[:, :6])
    target_d2 = np.diff(target_deg, n=2, axis=0)
    observed_d2 = np.diff(observed_deg, n=2, axis=0)
    steps = np.max(np.abs(np.diff(target_deg, axis=0)), axis=1)
    published = [r for r in rows if r["prediction"]["inference_performed"]]
    latency = [
        r["prediction"]["asynchronous_execution"]["inference_seconds"] * 1000 for r in published
    ]
    cycles = [r["cycle_work_s"] * 1000 for r in rows]
    rtc_prefix = sum(
        r["prediction"]["asynchronous_execution"].get("rtc_request", {}).get("prefix_model_actions")
        is not None
        for r in published
    )
    frames = [
        json.loads(line) for line in (folder / "trial-rgb/frames.jsonl").read_text().splitlines()
    ]
    assert len(frames) == raw["rgb_recording"]["written"]
    missing_frames = [f["file"] for f in frames if not (folder / "trial-rgb" / f["file"]).is_file()]
    assert not missing_frames, (folder.name, missing_frames)
    recorded_execution = {
        k: config[k]
        for k in (
            "rtc",
            "rtc_guidance",
            "rtc_num_steps",
            "n_action_steps",
            "lowpass",
            "seed",
            "action_padding",
            "target_transform",
        )
    }
    settings.append(
        {
            "execution": recorded_execution,
            "robot": raw["config"]["robot"],
            "teleop": raw["config"]["teleop"],
            "safety": raw["safety"],
        }
    )
    trials.append(
        {
            "trial": folder.name,
            "source": str(path),
            "ended_local_from_mtime": ended.isoformat(),
            "first_policy_local": datetime.fromtimestamp(first_unix, local_tz).isoformat(),
            "model": model,
            "run_id": spec["models"][model]["run_id"],
            "checkpoint": raw["checkpoint"],
            "task": task,
            "task_text": config["task"],
            "layout": raw["position"],
            "outcome": outcome,
            "task_outcome": outcome["task_outcome"],
            "exit_code": exit_code,
            "program_status": raw["status"],
            "hold_confirmed": raw.get("hold_confirmed"),
            "stop_outcome": raw["stop_outcome"],
            "error": raw.get("error"),
            "disconnect_error": raw.get("disconnect_error"),
            "execution": recorded_execution,
            "policy_actions_sent": len(rows),
            "policy_segment_seconds": float(times[-1] - times[0]),
            "target_d2_rms_deg_per_tick2": float(np.sqrt(np.mean(target_d2[triples] ** 2)))
            if triples.any()
            else None,
            "observed_d2_rms_deg_per_tick2": float(np.sqrt(np.mean(observed_d2[triples] ** 2)))
            if triples.any()
            else None,
            "seam_max_axis_step_deg_p50_p95_p99_max": quantiles(steps[consecutive & seams]),
            "inference_ms_p50_p95_p99_max": quantiles(latency),
            "cycle_work_ms_p50_p95_p99_max": quantiles(cycles),
            "observation_interval_ms_p50_p95_p99_max": quantiles(np.diff(times) * 1000),
            "logical_tick_gaps": int((~consecutive).sum()),
            "valid_triples": int(triples.sum()),
            "cycle_work_over_20ms": sum(x > 20 for x in cycles),
            "inferences_published": len(published),
            "rtc_prefix_publications": rtc_prefix,
            "sdk_results": dict(Counter(r["sdk"]["result"] for r in rows)),
            "rejection_rows": sum(
                bool(r.get("sdk", {}).get("rejection_reason")) for r in raw["rows"]
            ),
            "rgb": raw["rgb_recording"],
            "rgb_index_entries": len(frames),
            "missing_rgb_files": missing_frames,
            "first_rgb": frames[0],
            "last_rgb": frames[-1],
        }
    )
    commands[folder.name] = command

trials.sort(key=lambda t: t["first_policy_local"])
groups = {}
for model in ("original", "flat"):
    selected = [t for t in trials if t["model"] == model]
    counts = Counter(t["task_outcome"] for t in selected)
    scored = counts["success"] + counts["failure"]
    cells = []
    for layout in ("B-layout1", "B-layout2"):
        for task in spec["tasks"]:
            cell = [t for t in selected if t["layout"] == layout and t["task"] == task]
            c = Counter(t["task_outcome"] for t in cell)
            n = c["success"] + c["failure"]
            cells.append(
                {
                    "task": task,
                    "layout": layout,
                    "attempts": len(cell),
                    "success": c["success"],
                    "failure": c["failure"],
                    "invalid": c["invalid"],
                    "scored": n,
                    "rate": c["success"] / n if n else None,
                }
            )
    groups[model] = {
        "attempts": len(selected),
        "success": counts["success"],
        "failure": counts["failure"],
        "invalid": counts["invalid"],
        "scored": scored,
        "observed_rate": counts["success"] / scored,
        "descriptive_wilson95": wilson(counts["success"], scored),
        "cell_macro_rate": float(np.mean([c["rate"] for c in cells]))
        if all(c["rate"] is not None for c in cells)
        else None,
        "cells": cells,
    }

motion_cells = []
for layout in ("B-layout1", "B-layout2"):
    for task in spec["tasks"]:
        item = {"task": task, "layout": layout}
        for model in ("original", "flat"):
            selected = [
                t
                for t in trials
                if t["model"] == model
                and t["layout"] == layout
                and t["task"] == task
                and t["task_outcome"] == "success"
            ]
            item[model] = {"n_success": len(selected)}
            if selected:
                for key in ("target_d2_rms_deg_per_tick2", "observed_d2_rms_deg_per_tick2"):
                    item[model][key] = float(np.mean([t[key] for t in selected]))
        motion_cells.append(item)

motion_macro = {}
for model in ("original", "flat"):
    if all(c[model]["n_success"] for c in motion_cells):
        motion_macro[model] = {
            key: float(np.mean([c[model][key] for c in motion_cells]))
            for key in ("target_d2_rms_deg_per_tick2", "observed_d2_rms_deg_per_tick2")
        }

result = {
    "analysis_utc": datetime.now(timezone.utc).isoformat(),
    "source_root": str(root),
    "batch_day_asia_shanghai": day,
    "selection": "All trial directories whose report mtime is October 29 Asia/Shanghai; each first policy feedback timestamp independently matches that day. No outcome-based selection; October 27 pilot remains separate.",
    "designation": "Operator explicitly designated this actual-command batch as development closeout, superseding the suggested unexecuted 16-trial order; not a preregistered final test.",
    "trial_count": len(trials),
    "groups": groups,
    "trials": trials,
    "commands": commands,
    "common_execution_robot_teleop_safety": settings[0],
    "all_recorded_settings_equal": all(s == settings[0] for s in settings),
    "descriptive_statistics_limit": "Unequal repetitions, sequential nonrandomized model order, small sample and retrospective batch designation. Wilson intervals describe counts only and do not establish independent randomized trials or model superiority.",
    "motion_metric": "Unweighted per-trial RMS joint second differences at consecutive logical 20ms ticks; successful trials only, cell means then eight-cell macro. Feedback includes actual movement and sampling/quantization, not calibrated jerk or pure jitter.",
    "successful_motion_cells": motion_cells,
    "successful_motion_cell_macro": motion_macro,
    "successful_motion_flat_relative_change_percent": {
        key: (motion_macro["flat"][key] / motion_macro["original"][key] - 1) * 100
        for key in motion_macro["flat"]
    }
    if len(motion_macro) == 2
    else None,
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
