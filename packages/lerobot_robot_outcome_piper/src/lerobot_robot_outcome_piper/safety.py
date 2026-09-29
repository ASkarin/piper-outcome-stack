"""Explicit runtime motion limits and live firmware compatibility."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .documents import load_object
from .workspace import WorkspaceGeometry
from .errors import OutcomePiperValidationError

JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7))
ACTION_KEYS = (*JOINT_KEYS, "gripper.pos")


@dataclass(frozen=True)
class MotionSafety:
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    max_joint_step: tuple[float, ...]
    gripper_lower: float
    gripper_upper: float
    max_gripper_step: float
    workspace_lower: tuple[float, float, float]
    workspace_upper: tuple[float, float, float]
    feedback_timeout_s: float
    watchdog_timeout_s: float
    motion_speed_percent: int
    gripper_force_n: float
    stop_strategy: str
    workspace_geometry: WorkspaceGeometry | None = None

    def __post_init__(self):
        if isinstance(self.workspace_geometry, dict):
            object.__setattr__(
                self, "workspace_geometry", WorkspaceGeometry(**self.workspace_geometry)
            )
        if self.workspace_geometry is not None and not isinstance(
            self.workspace_geometry, WorkspaceGeometry
        ):
            raise ValueError("workspace_geometry must be a WorkspaceGeometry object")


def check_joint_feedback(values, safety, *, target=None, tolerance=0.0):
    """Keep measured angles intact; distinguish command bounds from arrival error.

    A small excursion beyond a command boundary is measured from that boundary,
    not from a moving target. A legal target sent by this session is still required;
    target-to-feedback step checks and hold-arrival checks remain separate.
    """
    events = []
    for i, (q, lo, hi) in enumerate(
        zip(values, safety.joint_lower, safety.joint_upper, strict=True)
    ):
        if lo <= q <= hi:
            continue
        anchor = None if target is None else target[i]
        boundary = lo if q < lo else hi
        event = dict(
            joint=i + 1,
            measured_rad=q,
            lower_rad=lo,
            upper_rad=hi,
            excess_rad=max(lo - q, q - hi),
            last_target_rad=anchor,
            boundary_rad=boundary,
            tracking_error_rad=None if anchor is None else q - anchor,
            arrival_tolerance_rad=tolerance,
        )
        if (
            not math.isfinite(q)
            or anchor is None
            or not lo <= anchor <= hi
            or not math.isfinite(tolerance)
            or tolerance <= 0
            or abs(q - boundary) > tolerance
        ):
            raise OutcomePiperValidationError(
                f"joint feedback outside joint limits and arrival tolerance: {event}; "
                f"measured_joint_rad={list(values)}"
            )
        events.append(event)
    return events


_FIRMWARE_IDENTITY_KEYS = (
    "hardware_version",
    "motor_ratio_and_batch",
    "node_type",
    "software_version",
    "production_date",
    "node_number",
)
_SOFTWARE_VERSION = re.compile(r"^S-V(?P<major>\d+)\.(?P<minor>\d+)-(?P<patch>\d+)$")


def _finite_vector(value: object, label: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise OutcomePiperValidationError(f"{label} must contain exactly {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise OutcomePiperValidationError(f"{label} values must be finite")
    return result


def _validate_firmware_driver(software_version: str, firmware: str) -> tuple[int, int, int]:
    match = _SOFTWARE_VERSION.fullmatch(software_version)
    if match is None:
        raise OutcomePiperValidationError(
            f"unsupported PiPER software_version format: {software_version!r}"
        )
    version = tuple(int(match.group(name)) for name in ("major", "minor", "patch"))
    compatible = {
        "default": version <= (1, 8, 2),
        "v183": (1, 8, 3) <= version <= (1, 8, 7),
        "v188": version == (1, 8, 8),
        "v189": version >= (1, 8, 9),
    }
    if not compatible[firmware]:
        raise OutcomePiperValidationError(
            f"firmware driver {firmware!r} does not match live software_version "
            f"{software_version!r}"
        )
    return version


def validate_live_firmware_driver(
    live_firmware: Mapping[str, Any], *, firmware: str
) -> dict[str, str]:
    """Validate received identity fields against the explicitly selected SDK driver."""

    observed: dict[str, str] = {}
    for key in _FIRMWARE_IDENTITY_KEYS:
        value = live_firmware.get(key)
        if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value).strip():
            raise OutcomePiperValidationError(f"live firmware identity is missing {key}")
        observed[key] = str(value)
    if observed["node_type"] != "ARM_MC":
        raise OutcomePiperValidationError("live firmware is not a PiPER arm controller")
    _validate_firmware_driver(observed["software_version"], firmware)
    return observed


def validate_motion_firmware(live_firmware: Mapping[str, Any], *, firmware: str) -> dict[str, str]:
    observed = validate_live_firmware_driver(live_firmware, firmware=firmware)
    version = _validate_firmware_driver(observed["software_version"], firmware)
    if version < (1, 6, 3):
        raise OutcomePiperValidationError(
            "motion requires software_version >= S-V1.6-3 for the pinned PiPER MDH model"
        )
    return observed


def load_motion_safety(safety_path: Path) -> MotionSafety:
    safety = load_object(safety_path)
    if safety.get("schema_version") != "outcome-piper-safety-v1":
        raise OutcomePiperValidationError("unsupported safety schema")
    lower = _finite_vector(safety.get("joint_lower_rad"), "joint_lower_rad", 6)
    upper = _finite_vector(safety.get("joint_upper_rad"), "joint_upper_rad", 6)
    steps = _finite_vector(safety.get("max_joint_step_rad"), "max_joint_step_rad", 6)
    if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
        raise OutcomePiperValidationError("each joint lower limit must be below its upper limit")
    if any(step <= 0 for step in steps):
        raise OutcomePiperValidationError("joint step limits must be positive")

    gripper_lower = float(safety.get("gripper_lower_m"))
    gripper_upper = float(safety.get("gripper_upper_m"))
    max_gripper_step = float(safety.get("max_gripper_step_m"))
    workspace_lower = _finite_vector(safety.get("workspace_lower_m"), "workspace_lower_m", 3)
    workspace_upper = _finite_vector(safety.get("workspace_upper_m"), "workspace_upper_m", 3)
    feedback_timeout_s = float(safety.get("feedback_timeout_s"))
    watchdog_timeout_s = float(safety.get("watchdog_timeout_s"))
    motion_speed_percent = safety.get("motion_speed_percent")
    gripper_force_n = float(safety.get("gripper_force_n"))
    scalars = (
        gripper_lower,
        gripper_upper,
        max_gripper_step,
        feedback_timeout_s,
        watchdog_timeout_s,
        gripper_force_n,
    )
    if not all(math.isfinite(item) for item in scalars):
        raise OutcomePiperValidationError("safety scalar values must be finite")
    if gripper_lower < 0 or gripper_lower >= gripper_upper or max_gripper_step <= 0:
        raise OutcomePiperValidationError("invalid gripper bounds or step")
    if any(lo >= hi for lo, hi in zip(workspace_lower, workspace_upper, strict=True)):
        raise OutcomePiperValidationError(
            "each workspace lower limit must be below its upper limit"
        )
    if feedback_timeout_s <= 0 or watchdog_timeout_s <= 0:
        raise OutcomePiperValidationError("feedback and watchdog timeouts must be positive")
    if type(motion_speed_percent) is not int or not 1 <= motion_speed_percent <= 100:
        raise OutcomePiperValidationError("motion_speed_percent must be an integer from 1 to 100")
    if not 0 < gripper_force_n <= 3.0:
        raise OutcomePiperValidationError("gripper_force_n must be within the official 0-3 N range")
    stop_strategy = safety.get("stop_strategy")
    if stop_strategy != "electronic_emergency_stop":
        raise OutcomePiperValidationError("stop_strategy must be electronic_emergency_stop")
    return MotionSafety(
        lower,
        upper,
        steps,
        gripper_lower,
        gripper_upper,
        max_gripper_step,
        workspace_lower,
        workspace_upper,
        feedback_timeout_s,
        watchdog_timeout_s,
        motion_speed_percent,
        gripper_force_n,
        stop_strategy,
        None
        if safety.get("workspace_geometry") is None
        else WorkspaceGeometry(**safety["workspace_geometry"]),
    )


def step_within_limit(target: float, current: float, limit: float) -> bool:
    """Allow arithmetic roundoff only, measured in float ULPs, not physical tolerance."""
    roundoff = 4 * max(math.ulp(target), math.ulp(current), math.ulp(limit))
    return abs(target - current) <= limit + roundoff
