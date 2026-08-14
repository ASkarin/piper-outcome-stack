"""Frozen motion limits and the hardware acceptance binding."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .documents import load_object, sha256_file
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


_FIRMWARE_IDENTITY_KEYS = (
    "hardware_version",
    "motor_ratio_and_batch",
    "node_type",
    "software_version",
    "production_date",
    "node_number",
)
_HARDWARE_IDENTITY_FIELDS = (
    "acceptance_id",
    "validated_at_utc",
    "validated_by",
    "nameplate_model",
    "robot_serial_number",
    "gripper_identifier",
    "usb_can_identifier",
    "physical_emergency_stop_identifier",
)
_SOFTWARE_VERSION = re.compile(r"^S-V(?P<major>\d+)\.(?P<minor>\d+)-(?P<patch>\d+)$")


def _finite_vector(value: object, label: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise OutcomePiperValidationError(f"{label} must contain exactly {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise OutcomePiperValidationError(f"{label} values must be finite")
    return result


def _validate_firmware_driver(software_version: str, firmware: str) -> None:
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


def _validated_acceptance(
    acceptance_path: Path,
    *,
    can_interface: str,
    firmware: str,
    safety_path: Path | None = None,
) -> dict[str, Any]:
    acceptance = load_object(acceptance_path)
    if acceptance.get("schema_version") != "outcome-piper-hardware-acceptance-v1":
        raise OutcomePiperValidationError("unsupported hardware acceptance schema")
    required_true = (
        "standard_piper_verified",
        "official_gripper_verified",
        "official_power_and_harness_verified",
        "official_usb_can_verified",
        "physical_emergency_stop_verified",
        "five_read_only_cycles_verified",
        "communication_loss_stop_verified",
        "watchdog_stop_verified",
        "hold_to_run_stop_verified",
        "electronic_emergency_stop_verified",
        "no_drop_stop_verified",
        "stop_strategy_verified",
    )
    if any(acceptance.get(key) is not True for key in required_true):
        raise OutcomePiperValidationError("hardware acceptance gate is incomplete")
    for field in _HARDWARE_IDENTITY_FIELDS:
        value = acceptance.get(field)
        if not isinstance(value, str) or not value.strip():
            raise OutcomePiperValidationError(
                f"hardware acceptance {field} must be explicit and non-empty"
            )
    if acceptance["nameplate_model"] != "PiPER":
        raise OutcomePiperValidationError(
            "hardware acceptance nameplate_model must be exactly 'PiPER'"
        )
    if acceptance.get("can_interface") != can_interface:
        raise OutcomePiperValidationError("hardware acceptance CAN interface does not match")
    if acceptance.get("firmware") != firmware:
        raise OutcomePiperValidationError("hardware acceptance firmware does not match")
    if acceptance.get("stop_strategy") != "electronic_emergency_stop":
        raise OutcomePiperValidationError(
            "motion requires the hardware-verified electronic emergency-stop strategy"
        )
    if safety_path is not None and acceptance.get("safety_sha256") != sha256_file(safety_path):
        raise OutcomePiperValidationError("hardware acceptance safety digest does not match")
    expected_firmware = acceptance.get("firmware_identity")
    if not isinstance(expected_firmware, dict):
        raise OutcomePiperValidationError("hardware acceptance firmware_identity must be an object")
    for key in _FIRMWARE_IDENTITY_KEYS:
        value = expected_firmware.get(key)
        if not isinstance(value, str) or not value.strip():
            raise OutcomePiperValidationError(
                f"hardware acceptance firmware_identity.{key} must be explicit"
            )
    if expected_firmware["node_type"] != "ARM_MC":
        raise OutcomePiperValidationError("firmware identity is not a PiPER arm controller")
    _validate_firmware_driver(expected_firmware["software_version"], firmware)
    return acceptance


def validate_live_hardware_acceptance(
    acceptance_path: Path,
    *,
    can_interface: str,
    firmware: str,
    live_firmware: Mapping[str, Any],
) -> dict[str, str]:
    """Bind a motion session to the exact firmware identity observed over CAN."""

    acceptance = _validated_acceptance(
        acceptance_path,
        can_interface=can_interface,
        firmware=firmware,
    )
    expected = acceptance["firmware_identity"]
    observed: dict[str, str] = {}
    for key in _FIRMWARE_IDENTITY_KEYS:
        value = live_firmware.get(key)
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise OutcomePiperValidationError(f"live firmware identity is missing {key}")
        observed[key] = str(value)
        if observed[key] != expected[key]:
            raise OutcomePiperValidationError(
                f"live firmware identity mismatch for {key}: "
                f"expected {expected[key]!r}, observed {observed[key]!r}"
            )
    if observed["node_type"] != "ARM_MC":
        raise OutcomePiperValidationError("live firmware is not a PiPER arm controller")
    _validate_firmware_driver(observed["software_version"], firmware)
    return observed


def validate_live_firmware_driver(
    live_firmware: Mapping[str, Any], *, firmware: str
) -> dict[str, str]:
    """Validate read-only sessions against the explicitly selected SDK driver."""

    observed: dict[str, str] = {}
    for key in _FIRMWARE_IDENTITY_KEYS:
        value = live_firmware.get(key)
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise OutcomePiperValidationError(f"live firmware identity is missing {key}")
        observed[key] = str(value)
    if observed["node_type"] != "ARM_MC":
        raise OutcomePiperValidationError("live firmware is not a PiPER arm controller")
    _validate_firmware_driver(observed["software_version"], firmware)
    return observed


def load_motion_safety(
    safety_path: Path,
    acceptance_path: Path,
    *,
    can_interface: str,
    firmware: str,
) -> MotionSafety:
    safety = load_object(safety_path)
    if safety.get("schema_version") != "outcome-piper-safety-v1":
        raise OutcomePiperValidationError("unsupported safety schema")
    acceptance = _validated_acceptance(
        acceptance_path,
        can_interface=can_interface,
        firmware=firmware,
        safety_path=safety_path,
    )

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
        raise OutcomePiperValidationError(
            "stop_strategy must be the hardware-verified electronic_emergency_stop"
        )
    if acceptance["stop_strategy"] != stop_strategy:
        raise OutcomePiperValidationError("hardware acceptance stop strategy does not match safety")
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
    )
