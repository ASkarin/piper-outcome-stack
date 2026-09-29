"""Raw teach records and validation shared by acquisition and offline conversion."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from .camera import CameraTelemetry, validate_frame
from .teach_source import check_feedback
from .timing import CaptureTiming

SCHEMA = "piper-teach-raw-v1"
RULE = "next_observed_state_v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def load_config(path):
    import draccus
    from .config import OutcomePiperConfig

    values = read_json(path)
    if "teleop" in values:
        raise ValueError("teach config must not contain Xbox teleop configuration")
    cfg = draccus.decode(OutcomePiperConfig, values["robot"])
    if cfg.execution_mode != "read_only" or cfg.capture_timing is None or cfg.scene is None:
        raise ValueError("teach requires explicit read_only, scene and capture_timing")
    if not cfg.cameras or not any(c.use_rgb for c in cfg.cameras.values()):
        raise ValueError("teach recording requires an RGB stream; depth is optional")
    if type(values["fps"]) is not int:
        raise ValueError("fps must be an integer Dataset rate")
    for key in ("fps", "episode_time_s"):
        if not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if type(values["writer_queue_size"]) is not int or values["writer_queue_size"] <= 0:
        raise ValueError("writer_queue_size must be a positive integer")
    if not isinstance(values["task"], str) or not values["task"].strip():
        raise ValueError("task must be explicit")
    # Store resolved config values, not a mutable path to another run config.
    values["robot"] = json.loads(json.dumps(asdict(cfg), default=str))
    return cfg, values


def validate_sample(row, previous, config, *, recording=True):
    robot = config["robot"]
    check_feedback(row, robot["feedback_timeout_s"])
    if recording and row["teach_status"] != 1:
        raise RuntimeError("sample is outside teach recording")
    timing = CaptureTiming(**robot["capture_timing"])
    now = row["sampled_monotonic_s"]
    received = row["received_monotonic_s"]
    if max(received[:3]) - min(received[:3]) > timing.joint_max_skew_s:
        raise RuntimeError("joint feedback group skew exceeded")
    if previous is not None:
        if now <= previous["sampled_monotonic_s"]:
            raise RuntimeError("sample monotonic time did not advance")
        if any(
            t <= p for t, p in zip(received[:3], previous["received_monotonic_s"][:3], strict=True)
        ):
            raise RuntimeError("joint feedback group did not advance")
        if (
            row["frame_index"] != previous["frame_index"] + 1
            or row["tick_index"] != previous["tick_index"] + 1
        ):
            raise RuntimeError("missing sampling tick; cannot pair across gap")
    expected = {}
    for name, cfg in robot["cameras"].items():
        if cfg["use_rgb"]:
            expected[name] = "color"
        if cfg["use_depth"]:
            expected[f"{name}.depth"] = "depth"
    if set(row["camera"]) != set(expected):
        raise RuntimeError("camera streams differ from configuration")
    for key, meta in row["camera"].items():
        old = None if previous is None else CameraTelemetry(**previous["camera"][key])
        current = CameraTelemetry(**meta)
        if current.stream != expected[key]:
            raise RuntimeError("camera stream kind differs from configuration")
        validate_frame(old, current)
        if not 0 <= now - current.received_monotonic_s <= timing.camera_max_age_s:
            raise RuntimeError(
                f"camera frame stale: stream={key}, frame={current.frame_number}, "
                f"age_ms={(now - current.received_monotonic_s) * 1000:.3f}, "
                f"limit_ms={timing.camera_max_age_s * 1000:.3f}, "
                f"capture_to_publish_ms={(current.published_monotonic_s - current.received_monotonic_s) * 1000:.3f}, "
                f"read_timing={row.get('sampling_read_timing')}"
            )
        if (
            max(abs(t - current.received_monotonic_s) for t in received)
            > timing.image_state_max_skew_s
        ):
            raise RuntimeError("image/state receive-time skew exceeded")
        if current.stream == "depth" and (
            current.depth_scale_m is None
            or not math.isfinite(current.depth_scale_m)
            or current.depth_scale_m <= 0
        ):
            raise RuntimeError("missing metric depth scale")
    oldest = min(*received, *(m["received_monotonic_s"] for m in row["camera"].values()))
    if now - oldest > timing.observation_max_age_s:
        raise RuntimeError("assembled observation expired")


def validate_pixels(frames, config):
    layout = {}
    for name, cfg in config["robot"]["cameras"].items():
        if cfg["use_rgb"]:
            layout[name] = (cfg, False)
        if cfg["use_depth"]:
            layout[f"{name}.depth"] = (cfg, True)
    for key, pixels in frames.items():
        cfg, depth = layout[key]
        shape = (cfg["height"], cfg["width"]) if depth else (cfg["height"], cfg["width"], 3)
        # Official postprocessing may rotate the configured image.
        if cfg.get("rotation") in (90, 270, "90", "270"):
            shape = (shape[1], shape[0], *shape[2:])
        if pixels.shape != shape or pixels.dtype != (np.uint16 if depth else np.uint8):
            raise RuntimeError(f"wrong raw pixel shape/dtype: {key} {pixels.shape} {pixels.dtype}")


def save_pixels(directory, row, frames, config):
    directory = Path(directory)
    validate_pixels(frames, config)
    for key, entry in row["files"].items():
        pixels = frames[key]
        depth = row["camera"][key]["stream"] == "depth"
        p = directory / entry
        if depth:
            with p.open("xb") as out:
                np.savez(out, depth=pixels)
        else:
            Image.fromarray(pixels).save(p, compress_level=0)


def read_pixels(directory, row, streams=None):
    frames = {}
    for key, entry in row["files"].items():
        if streams is not None and key not in streams:
            continue
        p = Path(directory) / entry
        if row["camera"][key]["stream"] == "depth":
            with np.load(p, allow_pickle=False) as data:
                frames[key] = data["depth"].copy()
        else:
            with Image.open(p) as im:
                frames[key] = np.asarray(im).copy()
    return frames


def load_attempt(directory, config, *, pixel_streams=None):
    directory = Path(directory)
    result = read_json(directory / "result.json")
    if result.get("status") != "saved" or result.get("data_valid") is not True:
        raise ValueError("attempt is not saved and valid")
    if result.get("task_outcome") not in ("success", "failure", "cancelled"):
        raise ValueError("saved attempt lacks task outcome")
    rows = [json.loads(s) for s in (directory / "samples.jsonl").read_text().splitlines()]
    if len(rows) != result["frames"] or len(rows) < 2:
        raise RuntimeError("raw frame count mismatch or fewer than two samples")
    previous = None
    for index, row in enumerate(rows):
        if row["frame_index"] != index:
            raise RuntimeError("raw frame index mismatch")
        validate_sample(row, previous, config)
        if set(row["files"]) != set(row["camera"]):
            raise RuntimeError("raw image files missing")
        validate_pixels(read_pixels(directory, row, pixel_streams), config)
        previous = row
    return result, rows


def timing_summary(rows):
    if not rows:
        return {"frames": 0}
    metrics = {
        "feedback_age_ms": [
            1000 * (r["sampled_monotonic_s"] - min(r["received_monotonic_s"])) for r in rows
        ],
        "joint_skew_ms": [
            1000 * (max(r["received_monotonic_s"][:3]) - min(r["received_monotonic_s"][:3]))
            for r in rows
        ],
        "camera_age_ms": [
            1000
            * (
                r["sampled_monotonic_s"]
                - min(m["received_monotonic_s"] for m in r["camera"].values())
            )
            for r in rows
        ],
        "image_state_skew_ms": [
            1000
            * max(
                abs(t - m["received_monotonic_s"])
                for t in r["received_monotonic_s"]
                for m in r["camera"].values()
            )
            for r in rows
        ],
        "sample_interval_ms": [
            1000 * (b["sampled_monotonic_s"] - a["sampled_monotonic_s"])
            for a, b in zip(rows, rows[1:])
        ],
    }
    return {
        "frames": len(rows),
        "actual_hz": None
        if len(rows) < 2
        else (len(rows) - 1) / (rows[-1]["sampled_monotonic_s"] - rows[0]["sampled_monotonic_s"]),
        "host_receive_alignment_only": True,
        **{
            k: dict(zip(("p50", "p95", "p99", "max"), np.percentile(v, [50, 95, 99, 100]).tolist()))
            for k, v in metrics.items()
            if v
        },
    }
