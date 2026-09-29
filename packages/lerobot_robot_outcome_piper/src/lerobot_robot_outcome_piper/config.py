"""LeRobot configurations for the standard PiPER and the measured Xbox mapping."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

from .timing import CaptureTiming
from .scene import SceneContext
from .camera import observation_camera_features


@RobotConfig.register_subclass("outcome_piper")
@dataclass(kw_only=True)
class OutcomePiperConfig(RobotConfig):
    can_interface: str
    firmware: str
    feedback_timeout_s: float
    execution_mode: str = "read_only"
    safety_path: Path | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    capture_timing: CaptureTiming | None = None
    scene: SceneContext | None = None
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
        if self.execution_mode == "motion" and self.safety_path is None:
            raise ValueError("motion mode requires explicit safety_path")
        if self.safety_path is not None:
            self.safety_path = Path(self.safety_path)
        if isinstance(self.scene, dict):
            self.scene = SceneContext(**self.scene)
        if isinstance(self.capture_timing, dict):
            self.capture_timing = CaptureTiming(**self.capture_timing)
        if self.execution_mode == "motion" and self.cameras and self.capture_timing is None:
            raise ValueError("motion with cameras requires capture_timing")
        # Camera count, logical names and device selection belong to configuration.
        for name, camera in self.cameras.items():
            if not name or "/" in name:
                raise ValueError("camera names must be non-empty Dataset feature names without '/'")
            if not isinstance(camera, RealSenseCameraConfig):
                raise ValueError("timed capture currently uses the official RealSense backend")
            if not camera.serial_number_or_name.strip():
                raise ValueError("select a RealSense device by serial number or unique name")
            if any(
                value is None or value <= 0 for value in (camera.width, camera.height, camera.fps)
            ):
                raise ValueError("camera width, height and fps must define the observation shape")
        observation_camera_features(self.cameras)


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
    emergency_stop_button: int
    mode_switch_button: int
    home_button: int
    translation_switch_button: int
    hold_joint_tolerance_rad: float
    hold_stable_time_s: float
    hold_timeout_s: float
    deadzone: float
    control_hz: int
    xyz_step_m: float
    rotation_step_rad: float
    gripper_step_m: float
    axis_signs: tuple[int, int, int, int]
    trigger_rest_values: tuple[float, float]
    trigger_pressed_values: tuple[float, float]
    ik_max_nfev: int
    ik_timeout_s: float
    ik_residual_tolerance: float
    ik_min_singular_value: float
    work_pose_button: int | None = None
    work_joint_rad: tuple[float, ...] | None = None
    work_gripper_m: float | None = None
    pose_timing: dict | None = None
    streaming_reference: dict | None = None
    gripper_reference: dict | None = None
    id: str = "outcome_piper_xbox"

    def __post_init__(self) -> None:
        if self.gripper_reference is not None:
            from .gripper_reference import GripperReferenceSettings

            settings = GripperReferenceSettings(**self.gripper_reference)
            if self.control_hz <= 0 or not math.isclose(settings.period_s, 1 / self.control_hz):
                raise ValueError("gripper reference period must match control_hz")
        if self.streaming_reference is not None:
            from .continuous_reference import ReferenceSettings

            settings = ReferenceSettings(**self.streaming_reference)
            if self.control_hz <= 0 or not math.isclose(settings.period_s, 1 / self.control_hz):
                raise ValueError("streaming reference period must match control_hz")
        if self.pose_timing is not None:
            from .joint_pose import PoseTiming

            timing = PoseTiming(**self.pose_timing)
            if self.control_hz <= 0 or not math.isclose(timing.period_s, 1 / self.control_hz):
                raise ValueError("pose timing period must match control_hz")
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
        if any(
            type(index) is not int or index < 0
            for index in (
                *axes,
                self.hold_button,
                self.emergency_stop_button,
                self.mode_switch_button,
                self.home_button,
                self.translation_switch_button,
            )
        ):
            raise ValueError("axis and button indices must be non-negative")
        if (
            len(
                {
                    self.hold_button,
                    self.emergency_stop_button,
                    self.mode_switch_button,
                    self.home_button,
                    self.translation_switch_button,
                }
            )
            != 5
        ):
            raise ValueError("hold, home, RB, X and emergency-stop buttons must be distinct")
        work = (self.work_pose_button, self.work_joint_rad, self.work_gripper_m)
        if any(v is not None for v in work):
            if any(v is None for v in work):
                raise ValueError(
                    "work pose requires a measured button, six joints and gripper target"
                )
            if type(self.work_pose_button) is not int or self.work_pose_button < 0:
                raise ValueError("work pose button must be a measured non-negative index")
            if self.work_pose_button in {
                self.hold_button,
                self.emergency_stop_button,
                self.mode_switch_button,
                self.home_button,
                self.translation_switch_button,
            }:
                raise ValueError("work pose button must be distinct")
            if len(self.work_joint_rad) != 6 or not all(
                math.isfinite(v) for v in self.work_joint_rad
            ):
                raise ValueError("work pose requires six finite joint radians")
            if not math.isfinite(self.work_gripper_m) or self.work_gripper_m < 0:
                raise ValueError("work gripper target must be finite and non-negative")
        self.hold_settings()
        if len(set(axes)) != len(axes):
            raise ValueError("Xbox axis indices must be distinct")
        measured_floats = (
            self.deadzone,
            self.xyz_step_m,
            self.rotation_step_rad,
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
        if min(self.xyz_step_m, self.rotation_step_rad, self.gripper_step_m) <= 0:
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

    def hold_settings(self):
        from .teleop_control import HoldSettings

        return HoldSettings(
            self.hold_joint_tolerance_rad, self.hold_stable_time_s, self.hold_timeout_s
        )
