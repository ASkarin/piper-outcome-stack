"""LeRobot configurations for the standard PiPER and the measured Xbox mapping."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig


Firmware = Literal["default", "v183", "v188", "v189"]
ExecutionMode = Literal["read_only", "motion"]


@RobotConfig.register_subclass("outcome_piper")
@dataclass(kw_only=True)
class OutcomePiperConfig(RobotConfig):
    can_interface: str
    firmware: Firmware
    feedback_timeout_s: float
    execution_mode: ExecutionMode = "read_only"
    safety_path: Path | None = None
    hardware_acceptance_path: Path | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    id: str = "outcome_piper"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.can_interface.strip():
            raise ValueError("can_interface must be explicit and non-empty")
        if self.firmware not in {"default", "v183", "v188", "v189"}:
            raise ValueError("firmware must be one of default, v183, v188, v189")
        if self.execution_mode not in {"read_only", "motion"}:
            raise ValueError("execution_mode must be read_only or motion")
        if not math.isfinite(self.feedback_timeout_s) or self.feedback_timeout_s <= 0:
            raise ValueError("feedback_timeout_s must be a measured positive value")
        if self.execution_mode == "motion" and (
            self.safety_path is None or self.hardware_acceptance_path is None
        ):
            raise ValueError("motion mode requires safety_path and hardware_acceptance_path")
        if self.safety_path is not None:
            self.safety_path = Path(self.safety_path)
        if self.hardware_acceptance_path is not None:
            self.hardware_acceptance_path = Path(self.hardware_acceptance_path)


@TeleoperatorConfig.register_subclass("outcome_piper_xbox")
@dataclass(kw_only=True)
class OutcomePiperXboxConfig(TeleoperatorConfig):
    device_guid: str
    axis_x: int
    axis_y: int
    axis_z: int
    axis_yaw: int
    axis_left_trigger: int
    axis_right_trigger: int
    hold_button: int
    deadzone: float
    control_hz: int
    xyz_step_m: float
    yaw_step_rad: float
    gripper_step_m: float
    axis_signs: tuple[int, int, int, int]
    trigger_rest_values: tuple[float, float]
    trigger_pressed_values: tuple[float, float]
    ik_max_nfev: int
    ik_timeout_s: float
    ik_residual_tolerance: float
    ik_min_singular_value: float
    id: str = "outcome_piper_xbox"

    def __post_init__(self) -> None:
        if not self.device_guid.strip():
            raise ValueError("device_guid must be measured and explicit")
        axes = (
            self.axis_x,
            self.axis_y,
            self.axis_z,
            self.axis_yaw,
            self.axis_left_trigger,
            self.axis_right_trigger,
        )
        if any(index < 0 for index in (*axes, self.hold_button)):
            raise ValueError("axis and button indices must be non-negative")
        if len(set(axes)) != len(axes):
            raise ValueError("Xbox axis indices must be distinct")
        measured_floats = (
            self.deadzone,
            self.xyz_step_m,
            self.yaw_step_rad,
            self.gripper_step_m,
            self.ik_timeout_s,
            self.ik_residual_tolerance,
            self.ik_min_singular_value,
        )
        if not all(math.isfinite(value) for value in measured_floats):
            raise ValueError("Xbox and IK numeric configuration must be finite")
        if not 0 < self.deadzone < 1:
            raise ValueError("deadzone must be between zero and one")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if min(self.xyz_step_m, self.yaw_step_rad, self.gripper_step_m) <= 0:
            raise ValueError("Xbox step limits must be positive")
        if (
            self.ik_max_nfev <= 0
            or self.ik_timeout_s <= 0
            or self.ik_residual_tolerance <= 0
            or self.ik_min_singular_value <= 0
        ):
            raise ValueError("IK budget and tolerance must be positive")
        if len(self.axis_signs) != 4 or any(sign not in {-1, 1} for sign in self.axis_signs):
            raise ValueError("axis_signs must contain four values chosen from -1 and 1")
        trigger_values = (*self.trigger_rest_values, *self.trigger_pressed_values)
        if len(self.trigger_rest_values) != 2 or len(self.trigger_pressed_values) != 2:
            raise ValueError("trigger calibration must contain left and right values")
        if not all(math.isfinite(value) for value in trigger_values):
            raise ValueError("trigger calibration values must be finite")
        if any(
            math.isclose(rest, pressed)
            for rest, pressed in zip(
                self.trigger_rest_values, self.trigger_pressed_values, strict=True
            )
        ):
            raise ValueError("each trigger rest and pressed value must differ")
