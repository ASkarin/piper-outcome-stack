"""Fail-fast standard PiPER LeRobot implementation."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Callable, Mapping

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot

from .config import OutcomePiperConfig
from .errors import OutcomePiperStateError, OutcomePiperValidationError
from .input_safety import register_active_motion_session
from .processor import OutcomePiperAction
from .safety import (
    ACTION_KEYS,
    JOINT_KEYS,
    MotionSafety,
    load_motion_safety,
    validate_live_firmware_driver,
    validate_live_hardware_acceptance,
)
from .sdk import PiperFactory, create_piper


class PiperState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTED_DISABLED = "CONNECTED_DISABLED"
    ACTIVE = "ACTIVE"
    FAULT = "FAULT"
    E_STOP = "E_STOP"


_TERMINAL_STATES = frozenset({PiperState.FAULT, PiperState.E_STOP})


@dataclass(frozen=True)
class FeedbackTelemetry:
    timestamp_s: float
    joint_group_timestamps_s: tuple[float, float, float]
    joint_group_hz: tuple[float, float, float]
    arm_status_timestamp_s: float
    arm_status_hz: float
    gripper_timestamp_s: float
    gripper_hz: float
    ctrl_mode: int
    mode_feedback: int
    arm_status: int
    arm_error_code: int
    gripper_status_code: int


class OutcomePiper(Robot):
    config_class = OutcomePiperConfig
    name = "outcome_piper"

    def __init__(
        self,
        config: OutcomePiperConfig,
        *,
        piper_factory: PiperFactory = create_piper,
        camera_factory: Callable[[dict[str, Any]], dict[str, Any]] = make_cameras_from_configs,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(config)
        self.config = config
        self._piper_factory = piper_factory
        self._camera_factory = camera_factory
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._arm: Any | None = None
        self._gripper: Any | None = None
        self.cameras: dict[str, Any] = {}
        self._state = PiperState.DISCONNECTED
        self._safety: MotionSafety | None = None
        self._last_action_at: float | None = None
        self._last_feedback: FeedbackTelemetry | None = None
        self._latched_cause: str | None = None
        self._stop_error: str | None = None
        self._firmware_identity: dict[str, str] | None = None
        self._hardware_identity_verified = False
        self._command_lock = threading.RLock()
        self._emergency_stop_requested = threading.Event()
        self._emergency_stop_cause: BaseException | str | None = None
        self._emergency_stop_request_lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, ...]]:
        features: dict[str, type | tuple[int, ...]] = dict.fromkeys(ACTION_KEYS, float)
        for name, camera in self.config.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[name] = (camera.height, camera.width, 3)
            if getattr(camera, "use_depth", False):
                features[f"{name}_depth"] = (camera.height, camera.width, 1)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def state(self) -> PiperState:
        with self._command_lock:
            return self._state

    @property
    def last_feedback_telemetry(self) -> FeedbackTelemetry | None:
        with self._command_lock:
            return self._last_feedback

    @property
    def firmware_identity(self) -> dict[str, str] | None:
        with self._command_lock:
            return None if self._firmware_identity is None else dict(self._firmware_identity)

    @property
    def latched_cause(self) -> str | None:
        with self._command_lock:
            return self._latched_cause

    @property
    def stop_error(self) -> str | None:
        with self._command_lock:
            return self._stop_error

    @property
    def is_connected(self) -> bool:
        with self._command_lock:
            if self._state in _TERMINAL_STATES:
                # Official LeRobot cleanup is gated by this property. A terminal
                # session reports owned resources until disconnect releases them.
                return self._arm is not None or self._gripper is not None or bool(self.cameras)
            if self._arm is None:
                return False
            try:
                arm_connected = bool(self._arm.is_connected())
                disconnected_cameras = tuple(
                    name for name, camera in self.cameras.items() if not camera.is_connected
                )
            except Exception as exc:
                if self._state in {PiperState.CONNECTED_DISABLED, PiperState.ACTIVE}:
                    self._latch(PiperState.FAULT, exc)
                raise
            if self._state in {PiperState.CONNECTED_DISABLED, PiperState.ACTIVE} and (
                not arm_connected or disconnected_cameras
            ):
                unavailable = []
                if not arm_connected:
                    unavailable.append("arm")
                unavailable.extend(f"camera {name!r}" for name in disconnected_cameras)
                self._latch(
                    PiperState.FAULT,
                    f"connection lost: {', '.join(unavailable)}",
                )
            return arm_connected and not disconnected_cameras

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        raise OutcomePiperStateError("interactive calibration is not supported")

    def configure(self) -> None:
        with self._command_lock:
            if self._arm is None:
                raise OutcomePiperStateError("PiPER is not connected")
            if self.config.execution_mode != "motion":
                return
            assert self._safety is not None
            self._arm.set_auto_set_motion_mode_enabled(False)
            self._raise_if_comm_error("disable automatic motion-mode switching")
            self._arm.set_joint_limits_enabled(False)
            self._raise_if_comm_error("disable SDK joint limits")
            self._arm.set_motion_mode(self._arm.OPTIONS.MOTION_MODE.J)
            self._raise_if_comm_error("set joint position-velocity mode")
            self._confirm_motion_mode_locked()
            self._arm.set_speed_percent(self._safety.motion_speed_percent)
            self._raise_if_comm_error("set frozen motion speed")

    def _confirm_motion_mode_locked(self) -> None:
        assert self._arm is not None
        status = self._arm.get_arm_status()
        self._raise_if_comm_error("confirm joint position-velocity mode")
        if status is None:
            raise OutcomePiperStateError("motion-mode feedback is missing")
        ctrl_mode = int(status.msg.ctrl_mode)
        mode_feedback = int(status.msg.mode_feedback)
        if ctrl_mode != 0x01 or mode_feedback != 0x01:
            raise OutcomePiperStateError(
                "motion-mode feedback mismatch: "
                f"ctrl_mode=0x{ctrl_mode:02x}, mode_feedback=0x{mode_feedback:02x}"
            )

    @staticmethod
    def _cause_text(cause: BaseException | str) -> str:
        if isinstance(cause, BaseException):
            return f"{type(cause).__name__}: {cause}"
        return str(cause)

    def _set_latch(self, state: PiperState, cause: BaseException | str) -> None:
        with self._command_lock:
            if state not in _TERMINAL_STATES:
                raise ValueError("only terminal fault states may be latched")
            if self._emergency_stop_requested.is_set():
                state = PiperState.E_STOP
                cause = self._emergency_stop_cause or cause
            first_latch = self._state not in _TERMINAL_STATES
            if first_latch:
                self._state = state
                self._latched_cause = self._cause_text(cause)
                if self._safety is not None and self._hardware_identity_verified:
                    try:
                        if self._arm is not None:
                            self._arm.electronic_emergency_stop()
                    except Exception as exc:
                        self._stop_error = self._cause_text(exc)
                    else:
                        try:
                            self._raise_if_comm_error("electronic emergency stop")
                        except Exception as exc:
                            self._stop_error = self._cause_text(exc)

    def request_emergency_stop(self, cause: BaseException | str) -> None:
        """Synchronously request and latch the verified electronic stop action.

        The request event is set before waiting for an in-flight SDK command, so
        ``send_action`` cannot continue to a second command after an Xbox input
        failure. The SDK stop itself is serialized with all other SDK access.
        """

        with self._emergency_stop_request_lock:
            if self._emergency_stop_cause is None:
                self._emergency_stop_cause = cause
            self._emergency_stop_requested.set()
            self._set_latch(PiperState.E_STOP, cause)
            if self._stop_error is not None:
                raise OutcomePiperStateError(self._latched_message())

    def _raise_if_emergency_stop_requested_locked(self) -> None:
        if not self._emergency_stop_requested.is_set():
            return
        cause = self._emergency_stop_cause or "emergency stop requested"
        self._latch(PiperState.E_STOP, cause)

    def _latched_message(self) -> str:
        message = f"PiPER session latched {self._state.value}: {self._latched_cause}"
        if self._stop_error is not None:
            message += f"; stop action failed: {self._stop_error}"
        return message

    def _latch(self, state: PiperState, cause: BaseException | str) -> None:
        self._set_latch(state, cause)
        message = self._latched_message()
        if isinstance(cause, BaseException):
            raise OutcomePiperStateError(message) from cause
        raise OutcomePiperStateError(message)

    def _raise_if_comm_error(self, operation: str) -> None:
        assert self._arm is not None
        if self._arm.has_comm_error():
            detail = self._arm.get_comm_error()
            raise OutcomePiperStateError(f"{operation} left CAN communication in error: {detail}")

    def _start_watchdog(self) -> None:
        assert self._safety is not None
        self._watchdog_stop.clear()

        def monitor() -> None:
            interval = min(self._safety.watchdog_timeout_s / 4, 0.05)
            while not self._watchdog_stop.wait(interval):
                with self._command_lock:
                    if self._watchdog_stop.is_set():
                        return
                    last = self._last_action_at
                    if (
                        self._state == PiperState.ACTIVE
                        and last is not None
                        and self._monotonic() - last > self._safety.watchdog_timeout_s
                    ):
                        try:
                            self._latch(PiperState.FAULT, "action watchdog expired")
                        except OutcomePiperStateError:
                            return

        self._watchdog_thread = threading.Thread(
            target=monitor, name="outcome-piper-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join()
            self._watchdog_thread = None

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        with self._command_lock:
            if self._latched_cause is not None:
                raise OutcomePiperStateError(
                    f"PiPER session is terminally latched {self._state.value}; create a new session"
                )
            if self._emergency_stop_requested.is_set():
                raise OutcomePiperStateError("PiPER session has an emergency-stop request")
            if self._state != PiperState.DISCONNECTED:
                raise OutcomePiperStateError("PiPER session is already connected or faulted")
            if self.config.execution_mode == "motion":
                assert self.config.safety_path is not None
                assert self.config.hardware_acceptance_path is not None
                self._safety = load_motion_safety(
                    self.config.safety_path,
                    self.config.hardware_acceptance_path,
                    can_interface=self.config.can_interface,
                    firmware=self.config.firmware,
                )
                if not math.isclose(
                    self.config.feedback_timeout_s,
                    self._safety.feedback_timeout_s,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise OutcomePiperValidationError(
                        "feedback_timeout_s must match the frozen motion safety value"
                    )
            arm = self._piper_factory(self.config.can_interface, self.config.firmware)
            cameras = self._camera_factory(self.config.cameras)
            try:
                arm.connect()
                self._arm = arm
                self._raise_if_comm_error("connect")
                live_firmware = arm.get_firmware(
                    timeout=self.config.feedback_timeout_s,
                    min_interval=0.0,
                )
                self._raise_if_comm_error("read firmware identity")
                if not isinstance(live_firmware, Mapping):
                    raise OutcomePiperValidationError("live firmware identity is unavailable")
                if self.config.execution_mode == "motion":
                    assert self.config.hardware_acceptance_path is not None
                    self._firmware_identity = validate_live_hardware_acceptance(
                        self.config.hardware_acceptance_path,
                        can_interface=self.config.can_interface,
                        firmware=self.config.firmware,
                        live_firmware=live_firmware,
                    )
                    self._hardware_identity_verified = True
                else:
                    self._firmware_identity = validate_live_firmware_driver(
                        live_firmware,
                        firmware=self.config.firmware,
                    )
                self._gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
                self.configure()
                for camera in cameras.values():
                    camera.connect()
                self.cameras = cameras
                self._state = PiperState.CONNECTED_DISABLED
                self.get_observation()
                if self.config.execution_mode == "motion":
                    if not arm.enable():
                        raise OutcomePiperStateError("PiPER enable did not confirm all joints")
                    self._raise_if_comm_error("enable")
                    self._state = PiperState.ACTIVE
                    self._last_action_at = self._monotonic()
                    self._start_watchdog()
                    register_active_motion_session(self)
            except Exception as exc:
                self._set_latch(PiperState.FAULT, exc)
                for camera in cameras.values():
                    if getattr(camera, "is_connected", False):
                        camera.disconnect()
                if arm.is_connected():
                    arm.disconnect()
                self._arm = None
                self._gripper = None
                self.cameras = {}
                message = self._latched_message()
                if isinstance(exc, OutcomePiperStateError) and str(exc) == message:
                    raise
                raise OutcomePiperStateError(message) from exc

    def _joint_frames(self) -> tuple[Any, Any, Any]:
        assert self._arm is not None
        parser = self._arm._parser
        frames = tuple(getattr(parser, name, None) for name in ("joint_12", "joint_34", "joint_56"))
        if any(frame is None for frame in frames):
            self._latch(PiperState.FAULT, "incomplete joint feedback groups")
        return frames

    def _frame_frequency(self, frame: Any) -> float:
        assert self._arm is not None
        value = float(self._arm._ctx.fps.get_fps(frame.msg_type))
        if not math.isfinite(value) or value < 0:
            self._latch(PiperState.FAULT, "invalid joint feedback frequency")
        return value

    def _feedback(self) -> tuple[list[float], float]:
        if self._state in _TERMINAL_STATES:
            raise OutcomePiperStateError(self._latched_message())
        if self._arm is None or self._gripper is None:
            if self._state in {PiperState.CONNECTED_DISABLED, PiperState.ACTIVE}:
                self._latch(PiperState.FAULT, "PiPER feedback resources are unavailable")
            raise OutcomePiperStateError("PiPER is not connected")
        try:
            return self._read_feedback()
        except OutcomePiperStateError as exc:
            if self._state in _TERMINAL_STATES:
                raise
            self._latch(PiperState.FAULT, exc)
        except Exception as exc:
            self._latch(PiperState.FAULT, exc)

    def _read_feedback(self) -> tuple[list[float], float]:
        assert self._arm is not None
        assert self._gripper is not None
        if not self.is_connected:
            raise OutcomePiperStateError("PiPER is not connected")
        self._raise_if_comm_error("feedback read")
        joints = self._arm.get_joint_angles()
        gripper = self._gripper.get_gripper_status()
        status = self._arm.get_arm_status()
        self._raise_if_comm_error("feedback read")
        if joints is None or gripper is None or status is None:
            self._latch(PiperState.FAULT, "missing arm, status, or gripper feedback")
        frames = self._joint_frames()
        values = [float(item) for item in joints.msg]
        width = float(gripper.msg.value)
        if len(values) != 6 or not all(math.isfinite(item) for item in (*values, width)):
            self._latch(PiperState.FAULT, "malformed or non-finite feedback")
        if gripper.msg.mode != "width":
            self._latch(PiperState.FAULT, f"gripper mode is {gripper.msg.mode!r}, expected 'width'")
        gripper_status_code = int(gripper.msg.status_code)
        if gripper_status_code & 0x3F:
            self._latch(
                PiperState.FAULT,
                f"gripper fault status_code=0x{gripper_status_code:02x}",
            )
        arm_status = int(status.msg.arm_status)
        arm_error_code = int(status.msg.err_code)
        ctrl_mode = int(status.msg.ctrl_mode)
        mode_feedback = int(status.msg.mode_feedback)
        if arm_status == 1:
            self._latch(PiperState.E_STOP, "controller reports emergency stop")
        if arm_status != 0 or arm_error_code != 0:
            self._latch(
                PiperState.FAULT,
                f"controller arm_status={arm_status}, err_code=0x{arm_error_code:04x}",
            )
        joint_timestamps = tuple(float(frame.timestamp) for frame in frames)
        gripper_timestamp = float(gripper.timestamp)
        timestamps = (*joint_timestamps, float(status.timestamp), gripper_timestamp)
        if not all(math.isfinite(value) and value > 0 for value in timestamps):
            self._latch(PiperState.FAULT, "feedback timestamps are missing or invalid")
        now = self._wall_time()
        if any(timestamp > now for timestamp in timestamps):
            self._latch(PiperState.FAULT, "feedback timestamp is in the future")
        age = max(now - timestamp for timestamp in timestamps)
        if age > self.config.feedback_timeout_s:
            self._latch(PiperState.FAULT, f"feedback is stale by {age:.6f}s")
        joint_hz = tuple(self._frame_frequency(frame) for frame in frames)
        arm_status_hz = float(status.hz)
        gripper_hz = float(gripper.hz)
        if not all(
            math.isfinite(value) and value >= 0 for value in (*joint_hz, arm_status_hz, gripper_hz)
        ):
            self._latch(PiperState.FAULT, "feedback frequency is missing or invalid")
        self._last_feedback = FeedbackTelemetry(
            timestamp_s=min(timestamps),
            joint_group_timestamps_s=joint_timestamps,
            joint_group_hz=joint_hz,
            arm_status_timestamp_s=float(status.timestamp),
            arm_status_hz=arm_status_hz,
            gripper_timestamp_s=gripper_timestamp,
            gripper_hz=gripper_hz,
            ctrl_mode=ctrl_mode,
            mode_feedback=mode_feedback,
            arm_status=arm_status,
            arm_error_code=arm_error_code,
            gripper_status_code=gripper_status_code,
        )
        return values, width

    def get_observation(self) -> dict[str, Any]:
        with self._command_lock:
            joints, width = self._feedback()
            observation: dict[str, Any] = {
                **{key: value for key, value in zip(JOINT_KEYS, joints, strict=True)},
                "gripper.pos": width,
            }
            try:
                for name, camera in self.cameras.items():
                    if getattr(self.config.cameras[name], "use_rgb", True):
                        observation[name] = camera.async_read()
                    if getattr(self.config.cameras[name], "use_depth", False):
                        observation[f"{name}_depth"] = camera.async_read_depth()
            except Exception as exc:
                self._latch(PiperState.FAULT, exc)
            return observation

    def _validate_action(self, action: Mapping[str, Any]) -> tuple[list[float], float]:
        if set(action) != set(ACTION_KEYS):
            raise OutcomePiperValidationError(
                f"action keys mismatch: missing={sorted(set(ACTION_KEYS) - set(action))}, "
                f"extra={sorted(set(action) - set(ACTION_KEYS))}"
            )
        try:
            values = [float(action[key]) for key in ACTION_KEYS]
        except (TypeError, ValueError) as exc:
            raise OutcomePiperValidationError("action values must be numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise OutcomePiperValidationError("action values must be finite")
        assert self._safety is not None
        current_joints, current_gripper = self._feedback()
        for index, (target, current, lower, upper, step) in enumerate(
            zip(
                values[:6],
                current_joints,
                self._safety.joint_lower,
                self._safety.joint_upper,
                self._safety.max_joint_step,
                strict=True,
            ),
            start=1,
        ):
            if not lower <= target <= upper:
                raise OutcomePiperValidationError(f"joint_{index} target is outside frozen limits")
            if abs(target - current) > step:
                raise OutcomePiperValidationError(f"joint_{index} target exceeds frozen step limit")
        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

        target_pose = fk_from_mdh(list(get_mdh("piper")), values[:6])
        target_xyz = tuple(float(value) for value in target_pose[:3])
        if len(target_xyz) != 3 or not all(math.isfinite(value) for value in target_xyz):
            raise OutcomePiperValidationError("target FK did not produce a finite XYZ position")
        if any(
            not lower <= value <= upper
            for value, lower, upper in zip(
                target_xyz,
                self._safety.workspace_lower,
                self._safety.workspace_upper,
                strict=True,
            )
        ):
            raise OutcomePiperValidationError("action target is outside the frozen workspace")
        gripper = values[6]
        if not self._safety.gripper_lower <= gripper <= self._safety.gripper_upper:
            raise OutcomePiperValidationError("gripper target is outside frozen limits")
        if abs(gripper - current_gripper) > self._safety.max_gripper_step:
            raise OutcomePiperValidationError("gripper target exceeds frozen step limit")
        return values[:6], gripper

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        with self._command_lock:
            if self.config.execution_mode != "motion":
                raise OutcomePiperStateError("send_action requires execution_mode=motion")
            self._require_active_locked("send_action")
            joints, gripper = self._validate_action(action)
            if type(action) is OutcomePiperAction and action.execute_motion is False:
                self._latch(PiperState.E_STOP, "hold-to-run was released")
            try:
                self._require_active_locked("move_j")
                assert self._arm is not None
                self._arm.move_j(joints)
                self._raise_if_comm_error("move_j")
            except Exception as exc:
                self._latch(PiperState.FAULT, exc)
            self._raise_if_emergency_stop_requested_locked()
            try:
                self._require_active_locked("gripper command")
                assert self._gripper is not None
                assert self._safety is not None
                self._gripper.move_gripper_m(gripper, force=self._safety.gripper_force_n)
                self._raise_if_comm_error("gripper command")
            except Exception as exc:
                self._latch(PiperState.E_STOP, exc)
            self._raise_if_emergency_stop_requested_locked()
            self._last_action_at = self._monotonic()
            return {key: float(action[key]) for key in ACTION_KEYS}

    def _require_active_locked(self, operation: str) -> None:
        if self._state in _TERMINAL_STATES:
            raise OutcomePiperStateError(self._latched_message())
        if self._state != PiperState.ACTIVE or self._arm is None or self._gripper is None:
            raise OutcomePiperStateError(
                f"{operation} requires ACTIVE state, got {self._state.value}"
            )

    def disconnect(self) -> None:
        with self._command_lock:
            if self._state not in _TERMINAL_STATES:
                self._state = PiperState.DISCONNECTED
            self._watchdog_stop.set()
        self._stop_watchdog()
        with self._emergency_stop_request_lock:
            with self._command_lock:
                if self._emergency_stop_requested.is_set() and self._state not in _TERMINAL_STATES:
                    cause = self._emergency_stop_cause or "emergency stop requested"
                    self._set_latch(PiperState.E_STOP, cause)
                first_error: Exception | None = None
                for camera in self.cameras.values():
                    try:
                        if getattr(camera, "is_connected", False):
                            camera.disconnect()
                    except Exception as exc:
                        first_error = first_error or exc
                if self._arm is not None:
                    try:
                        self._arm.disconnect()
                    except Exception as exc:
                        first_error = first_error or exc
                self.cameras = {}
                self._arm = None
                self._gripper = None
                if first_error is not None:
                    raise first_error
