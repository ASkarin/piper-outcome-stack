"""Metadata-preserving capture, with lazy loading of the official RealSense driver."""

from dataclasses import dataclass
import math
import time


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
    deadline = time.monotonic() + timeout_s
    with camera.metadata_condition:
        while True:
            if camera.capture_error is not None:
                raise RuntimeError(
                    f"D435 capture failed: {camera.capture_error}"
                ) from camera.capture_error
            current = camera.latest_metadata
            if current is not None and current.frame_number != camera.consumed_frame_number:
                camera.consumed_frame_number = current.frame_number
                return camera.latest_color_frame.copy(), current
            if camera.thread is None or not camera.thread.is_alive():
                raise RuntimeError("D435 capture thread is not running")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no new D435 frame within feedback timeout")
            camera.metadata_condition.wait(remaining)
