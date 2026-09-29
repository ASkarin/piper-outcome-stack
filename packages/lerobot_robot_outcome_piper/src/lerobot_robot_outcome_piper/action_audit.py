"""Offline, source-specific action and Dataset evidence checks; no control loop."""

import math
import json
from pathlib import Path
import numpy as np
from .raw_io import read_jsonl
from .safety import ACTION_KEYS


def holding_reference(action):
    """Validate a retained command, not a fictitious dispatch for this frame."""
    command = action.get("hold_command")
    gripper = action.get("retained_gripper_command")
    values = action.get("values")
    if type(action.get("hold_id")) is not int or action["hold_id"] <= 0:
        raise RuntimeError("holding frame has no valid hold reference")
    if not isinstance(values, dict) or set(values) != set(ACTION_KEYS):
        raise RuntimeError("holding frame lacks seven effective action values")
    for item, name in ((command, "hold_move_j"), (gripper, "move_gripper_m")):
        if (
            not isinstance(item, dict)
            or item.get("name") != name
            or item.get("result") != "sdk_returned"
        ):
            raise RuntimeError("holding frame lacks a successful command reference")
        start, end = item.get("started_monotonic_s"), item.get("ended_monotonic_s")
        if (
            start is None
            or end is None
            or not all(math.isfinite(t) for t in (start, end))
            or end < start
        ):
            raise RuntimeError("holding command timestamps are invalid")
    expected = [values[k] for k in ACTION_KEYS]
    if command.get("target") != expected[:6] or gripper.get("target") != expected[6]:
        raise RuntimeError("holding values differ from retained command targets")
    if not all(math.isfinite(float(v)) for v in expected):
        raise RuntimeError("holding values are not finite")
    return {"hold_command": command, "retained_gripper_command": gripper, "values": values}


def validate_action(action):
    """Prove the seven effective targets using actual or retained SDK calls."""
    if action.get("result") == "holding":
        return holding_reference(action)
    if action.get("result") != "sdk_returned":
        raise RuntimeError("non-dispatched waiting/fault action entered the Dataset")
    values = action.get("values")
    if not isinstance(values, dict) or set(values) != set(ACTION_KEYS):
        raise RuntimeError("action lacks seven effective target values")
    expected = np.asarray([values[k] for k in ACTION_KEYS], dtype=np.float32)
    if not np.isfinite(expected).all():
        raise RuntimeError("non-finite action values")
    commands = action.get("commands", [])
    joints = [c for c in commands if c.get("name") == "move_j"]
    grips = [c for c in commands if c.get("name") == "move_gripper_m"]
    if len(joints) != 1 or len(grips) > 1:
        raise RuntimeError("action lacks one successful joint command")
    grip = grips[0] if grips else action.get("retained_gripper_command")
    for item, name, target in (
        (joints[0], "move_j", expected[:6]),
        (grip, "move_gripper_m", expected[6]),
    ):
        if (
            not isinstance(item, dict)
            or item.get("name") != name
            or item.get("result") != "sdk_returned"
        ):
            raise RuntimeError("action lacks successful SDK command evidence")
        start, end = item.get("started_monotonic_s"), item.get("ended_monotonic_s")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or end < start
        ):
            raise RuntimeError("SDK command timestamps are invalid")
        if not np.array_equal(np.asarray(item.get("target"), dtype=np.float32), target):
            raise RuntimeError("action values differ from SDK command targets")
    if any(c.get("result") != "sdk_returned" for c in commands):
        raise RuntimeError("failed SDK command in recorded action")
    return action


def verify_telemetry(root, dataset):
    if (Path(root) / "teach-conversion.json").exists():
        from .teach_dataset import audit_teach

        return audit_teach(root, dataset)
    rows = {}
    for directory in sorted((Path(root) / "telemetry").iterdir()):
        if not (directory / "complete.json").exists():
            raise RuntimeError(f"incomplete telemetry session: {directory.name}")
        completion = json.loads((directory / "complete.json").read_text())
        if completion.get("status") not in ("complete", "empty"):
            raise RuntimeError(f"incomplete telemetry session: {directory.name}")
        events = read_jsonl(directory / "events.jsonl")
        if any(e["event"] == "failed" for e in events):
            raise RuntimeError(f"failed telemetry session: {directory.name}")
        session = next((e for e in events if e["event"] == "session"), None)
        scene = None if session is None else session.get("scene")
        if scene is not None:
            from .timing_report import summarize_events

            timing = summarize_events(events)
            if timing["missing_timing_samples"]:
                raise RuntimeError("scene recording has missing timing evidence")
            anomalies = (
                "duplicate_frames",
                "backwards_frames",
                "device_time_anomalies",
                "host_time_anomalies",
            )
            if any(
                value
                for key, value in timing["all_samples"]["counters"].items()
                if key.endswith(anomalies)
            ):
                raise RuntimeError("recorded frame timing contains anomalies")
        references = {}
        for event in events:
            if event["event"] == "hold_reference":
                key = event["hold_id"]
                if key in references and references[key] != event["reference"]:
                    raise RuntimeError("inconsistent hold reference")
                references[key] = event["reference"]
        if completion.get("status") == "empty" and (
            completion.get("frames") != 0 or any(e["event"] == "episode_saved" for e in events)
        ):
            raise RuntimeError("empty session contains saved training frames")
        accepted = {
            (e["episode_index"], e["attempt"]) for e in events if e["event"] == "episode_saved"
        }
        for e in events:
            if e["event"] != "frame_pending" or (e["episode_index"], e["attempt"]) not in accepted:
                continue
            if scene is not None and e.get("scene_id") != scene["scene_id"]:
                raise RuntimeError("telemetry frame belongs to a different scene")
            action = e["action"]
            if action.get("result") == "holding":
                reference = holding_reference(action)
                if references.get(action["hold_id"]) != reference:
                    raise RuntimeError("holding frame has no matching recorded reference")
            elif action.get("result") != "sdk_returned":
                raise RuntimeError("non-dispatched waiting/fault action entered the Dataset")
            validate_action(action)
            key = (e["episode_index"], e["frame_index"])
            if key in rows:
                raise RuntimeError("duplicate committed telemetry row")
            rows[key] = e
    if len(rows) != dataset.num_frames:
        raise RuntimeError("telemetry and Dataset frame counts differ")
    depth_keys = dataset.meta.depth_keys
    actions = dataset.select_columns(["episode_index", "frame_index", "action", *depth_keys])
    for index in range(dataset.num_frames):
        frame = actions[index]
        key = (int(frame["episode_index"]), int(frame["frame_index"]))
        row = rows.get(key)
        if row is None:
            raise RuntimeError(f"missing telemetry row {key}")
        action = np.asarray([row["action"]["values"][k] for k in ACTION_KEYS], dtype=np.float32)
        if not np.array_equal(np.asarray(frame["action"]), action):
            raise RuntimeError(f"telemetry action mismatch at {key}")
        raw_depth = row.get("raw_depth", {})
        captured_depth = {
            name
            for name, meta in row["observation"].get("cameras", {}).items()
            if meta.get("depth_scale_m") is not None
        }
        if set(raw_depth) != captured_depth or not set(depth_keys).issubset(
            {f"observation.images.{name}" for name in raw_depth}
        ):
            raise RuntimeError(f"depth evidence does not match captured streams at {key}")
        for stream_key, entry in raw_depth.items():
            with np.load(Path(root) / entry["path"], allow_pickle=False) as stored:
                raw = stored["depth"]
            scale = entry["depth_scale_m"]
            if (
                raw.dtype != np.uint16
                or list(raw.shape) != entry["shape"]
                or not math.isfinite(scale)
                or scale <= 0
            ):
                raise RuntimeError(f"invalid raw depth evidence at {key}")
            if f"observation.images.{stream_key}" not in depth_keys:
                continue  # Z16-only archive: no duplicate metric column required.
            metric = raw.astype(np.float32) * scale
            # select_columns reads the stored HF image values, before the
            # LeRobot reader's requested m/mm conversion.
            info = dataset.features[f"observation.images.{stream_key}"].get("info", {})
            if info.get("depth_unit") != "m":
                raise RuntimeError("stored depth must declare metre units")
            actual = np.asarray(frame[f"observation.images.{stream_key}"]).squeeze(0)
            if not np.array_equal(actual, metric):
                raise RuntimeError(f"depth pixels differ from raw evidence at {key}")
    return {"episodes": dataset.num_episodes, "frames": len(rows)}
