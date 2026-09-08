"""Read archived commissioning reports, never query a device."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from piper_outcome_stack.sim.model import validate_action, validate_values


@dataclass(frozen=True)
class MotionRecord:
    sample_times: np.ndarray
    samples: np.ndarray
    commands: tuple
    source: str
    source_status: str
    time_origin: float

    @property
    def duration(self):
        return float(self.sample_times[-1])

    def measured(self, t):
        if not 0 <= t <= self.duration:
            raise ValueError("time outside recorded interval")
        return np.array([np.interp(t, self.sample_times, self.samples[:, i]) for i in range(7)])

    def initial_target(self):
        return self.samples[0].copy()


def read_record(path):
    path = Path(path)
    doc = json.loads(path.read_text())
    raw = doc["samples"]
    commands = doc["commands"]
    if len(raw) < 2 or not commands:
        raise ValueError("report needs at least two feedback samples and explicit commands")
    times = np.array([s["monotonic_s"] for s in raw], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("feedback timestamps must be finite and strictly increasing")
    samples = np.array([validate_values([*s["joint_rad"], s["gripper_width_m"]]) for s in raw])
    out = []
    last = -np.inf
    for c in commands:
        t = float(c["started_monotonic_s"])
        if not np.isfinite(t) or t < last or not times[0] <= t <= times[-1]:
            raise ValueError("invalid or unordered command timestamp")
        returned = float(c["returned_monotonic_s"])
        if not np.isfinite(returned) or returned < t:
            raise ValueError("command lacks a valid SDK return timestamp")
        last = t
        name = c["name"]
        if name in ("joint_waypoint", "hold_current_position"):
            q = np.asarray(c["args"][0], dtype=float)
            validate_action([*q, 0.0])
            out.append((t - times[0], "joints", q))
        elif name == "set_gripper_width":
            width = float(c["args"][0])
            validate_action([0, 0, 0, 0, 0, 0, width])
            out.append((t - times[0], "gripper", width))
        elif name not in (
            "disable_auto_mode",
            "disable_sdk_clipping",
            "speed_percent",
            "CAN_J_mode",
            "enable",
            "enable_gripper",
        ):
            raise ValueError(f"unsupported commissioning command: {name}")
    if not any(c[1] == "joints" for c in out) or not any(c[1] == "gripper" for c in out):
        raise ValueError("report must include joint and gripper targets; no commands are invented")
    return MotionRecord(
        times - times[0],
        samples,
        tuple(out),
        str(path.resolve()),
        str(doc["status"]),
        float(times[0]),
    )
