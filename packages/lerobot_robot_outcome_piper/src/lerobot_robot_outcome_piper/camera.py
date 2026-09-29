"""Configured camera streams and metadata, without importing a hardware driver."""

from dataclasses import dataclass
import math
import time
from .errors import OutcomePiperCameraError


def observation_camera_features(configs):
    features = {}
    reserved = {*(f"joint_{i}.pos" for i in range(1, 7)), "gripper.pos"}
    for name, cfg in configs.items():
        for key, channels, enabled in ((name, 3, cfg.use_rgb), (f"{name}.depth", 1, cfg.use_depth)):
            if not enabled:
                continue
            if key in features or key in reserved:
                raise ValueError(f"camera stream key collision: {key}")
            features[key] = (cfg.height, cfg.width, channels)
    return features


def make_timed_cameras(configs):
    if not configs:
        return {}
    from .realsense import TimedRealSenseCamera

    return {name: TimedRealSenseCamera(config) for name, config in configs.items()}


@dataclass(frozen=True)
class CameraTelemetry:
    frame_number: int
    device_timestamp_ms: float
    timestamp_domain: str
    received_monotonic_s: float
    published_monotonic_s: float
    stream: str = "color"
    pixel_format: str = "rgb8"
    serial_number: str | None = None
    intrinsics: dict | None = None
    depth_scale_m: float | None = None
    depth_to_color: dict | None = None
    alignment: str = "native"
    rotation: int | None = None


def validate_frame(previous, current):
    if (
        current.frame_number < 0
        or not math.isfinite(current.device_timestamp_ms)
        or current.device_timestamp_ms < 0
    ):
        raise RuntimeError("invalid camera device frame number or timestamp")
    if (
        not all(
            math.isfinite(t) and t >= 0
            for t in (current.received_monotonic_s, current.published_monotonic_s)
        )
        or current.published_monotonic_s < current.received_monotonic_s
    ):
        raise RuntimeError("invalid camera host monotonic timestamps")
    if previous is not None:
        if current.timestamp_domain != previous.timestamp_domain:
            raise RuntimeError("camera timestamp domain changed")
        if current.frame_number <= previous.frame_number:
            raise RuntimeError("duplicate or backwards camera frame number")
        if current.device_timestamp_ms <= previous.device_timestamp_ms:
            raise RuntimeError("camera device timestamp did not advance")
        if current.received_monotonic_s < previous.received_monotonic_s:
            raise RuntimeError("camera receive monotonic clock moved backwards")


def read_new_frame(camera, timeout_s):
    started = time.monotonic()
    deadline = started + timeout_s
    rejected = None
    skipped = 0
    detail = "no new frame"
    while True:
        # No camera lock is held while servicing input/robot feedback.
        service = getattr(camera, "wait_service", None)
        if service is not None:
            service()
        with camera.metadata_condition:
            now = time.monotonic()
            if camera.capture_error is not None:
                raise OutcomePiperCameraError(
                    f"camera capture failed: {camera.capture_error}"
                ) from camera.capture_error
            if now >= deadline and timeout_s != 0:
                raise OutcomePiperCameraError(
                    f"no fresh camera frame within {timeout_s:.6f}s: {detail}"
                )
            current = camera.latest_metadata
            sequence = tuple((key, value.frame_number) for key, value in current.items())
            max_age = getattr(camera, "max_frame_age_s", None)
            ages = [now - value.received_monotonic_s for value in current.values()]
            if any(not math.isfinite(age) or age < 0 for age in ages):
                raise OutcomePiperCameraError("camera monotonic timestamp is invalid")
            fresh = max_age is None or all(age <= max_age for age in ages)
            if sequence and sequence != camera.consumed_frame_number:
                if fresh:
                    camera.consumed_frame_number = sequence
                    camera.last_read_diagnostics = {
                        "wait_s": now - started,
                        "stale_frames_skipped": skipped,
                    }
                    return {key: value.copy() for key, value in camera.latest_frames.items()}, dict(
                        current
                    )
                detail = f"stale frame sequence={sequence}, ages_s={ages}, limit_s={max_age}"
                if sequence != rejected:
                    skipped += 1
                    rejected = sequence
            if camera.thread is None or not camera.thread.is_alive():
                raise OutcomePiperCameraError("camera capture thread is not running")
            if now >= deadline:
                raise OutcomePiperCameraError(
                    f"no fresh camera frame within {timeout_s:.6f}s: {detail}"
                )
            camera.metadata_condition.wait(min(deadline - now, timeout_s / 4, 0.01))
