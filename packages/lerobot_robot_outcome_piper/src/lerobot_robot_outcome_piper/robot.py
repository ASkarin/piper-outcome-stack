"""Fail-fast standard PiPER LeRobot implementation."""

from __future__ import annotations

import math
import logging
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Callable, Mapping

from lerobot.robots.robot import Robot

from .stage_timing import measured, reset, span, snapshot
from .camera import make_timed_cameras, observation_camera_features
from .config import OutcomePiperConfig
from .timing import FeedbackReceiver
from .errors import (
    OutcomePiperStateError,
    OutcomePiperValidationError,
    OutcomePiperIntentRejected,
    OutcomePiperCameraError,
)
from .input_safety import register_active_motion_session
from .processor import OutcomePiperAction
from .teleop_control import HoldSettings, JointHold, TeleopControl, TeleopState
from .safety import (
    ACTION_KEYS,
    JOINT_KEYS,
    MotionSafety,
    load_motion_safety,
    validate_live_firmware_driver,
    validate_motion_firmware,
    step_within_limit,
    check_joint_feedback,
)
from .sdk import PiperFactory, create_piper
from .workspace import workspace_pose_allowed, workspace_coordinates, gripper_table_allowed


class PiperState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    ACTIVE = "ACTIVE"
    FAULT = "FAULT"
    E_STOP = "E_STOP"


class ServoState(str, Enum):
    UNKNOWN = "UNKNOWN"
    DISABLED = "DISABLED"
    ENABLED = "ENABLED"
    PARTIAL = "PARTIAL"


@dataclass(frozen=True)
class ServoFeedback:
    joints: tuple[bool | None, ...] = (None,) * 6
    gripper: bool | None = None
    driver_errors: tuple[bool | None, ...] = (None,) * 6
    received_monotonic_s: tuple[float | None, ...] = (None,) * 7

    @property
    def state(self) -> ServoState:
        if any(v is None for v in self.joints):
            return ServoState.UNKNOWN
        if all(self.joints):
            return ServoState.ENABLED
        if not any(self.joints):
            return ServoState.DISABLED
        return ServoState.PARTIAL


_TERMINAL_STATES = frozenset({PiperState.FAULT, PiperState.E_STOP})


@dataclass(frozen=True)
class FeedbackTelemetry:
    timestamp_s: float
    received_monotonic_s: tuple[float, float, float, float, float]
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
        camera_factory: Callable[[dict[str, Any]], dict[str, Any]] = make_timed_cameras,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        receiver_factory: Callable = FeedbackReceiver,
    ) -> None:
        super().__init__(config)
        self.config = config
        self._piper_factory = piper_factory
        self._camera_factory = camera_factory
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._receiver_factory = receiver_factory
        self._receiver = None
        self.last_depth_frames = {}
        self.last_observation_telemetry = None
        self.control_trace = None
        self.camera_input_poll = None
        self.emergency_stop_poll = None
        self._input_thread_id = threading.get_ident()
        self.last_action_telemetry = None
        self._observation_sequence = 0
        self._arm: Any | None = None
        self._gripper: Any | None = None
        self.cameras: dict[str, Any] = {}
        self._state = PiperState.DISCONNECTED
        self._control_started = False
        self._motion_configured = False
        self.last_servo_feedback = ServoFeedback()
        self.last_servo_command = None
        self._safety: MotionSafety | None = None
        self._last_action_at: float | None = None
        self._last_control_tick_s: float | None = None
        self._teleop: TeleopControl | None = None
        self._hold_settings: HoldSettings | None = None
        self._hold_window: JointHold | None = None
        self._hold_command: dict | None = None
        self._hold_id = 0
        self._hold_epoch = -1
        self._running_epoch = -1
        self._last_gripper_target: float | None = None
        self._last_gripper_command: dict | None = None
        self._input_fault_requested = threading.Event()
        self._electronic_stop_attempted = False
        self._stop_outcome: str | None = None
        self._last_feedback: FeedbackTelemetry | None = None
        self._latched_cause: str | None = None
        self._stop_error: str | None = None
        self._firmware_identity: dict[str, str] | None = None
        self._firmware_verified = False
        self._command_lock = threading.RLock()
        self._emergency_stop_requested = threading.Event()
        self._emergency_stop_cause: BaseException | str | None = None
        self._emergency_stop_request_lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, ...]]:
        features: dict[str, type | tuple[int, ...]] = dict.fromkeys(ACTION_KEYS, float)
        features.update(observation_camera_features(self.config.cameras))
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
            except Exception as exc:
                if self._state in {PiperState.CONNECTED, PiperState.ACTIVE}:
                    self._latch(PiperState.FAULT, exc)
                raise
            if self._state in {PiperState.CONNECTED, PiperState.ACTIVE} and not arm_connected:
                self._latch(PiperState.FAULT, "connection lost: arm")
            # Camera availability is checked when images are required, independently
            # of ownership of the robot control connection.
            return arm_connected

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
            self._raise_if_emergency_stop_requested_locked()
            initial_status, received_s = self._receiver.status()
            now = self._monotonic()
            if (
                initial_status is None
                or received_s is None
                or not math.isfinite(received_s)
                or not 0 <= now - received_s <= self.config.feedback_timeout_s
            ):
                raise OutcomePiperStateError("initial controller status unavailable or stale")
            timestamp = float(initial_status.timestamp)
            if not math.isfinite(timestamp) or timestamp <= 0:
                raise OutcomePiperStateError("initial controller SDK timestamp is invalid")
            self._raise_if_comm_error("motion preflight")
            self._controller_status(initial_status)
            teach_status = int(initial_status.msg.teach_status)
            if teach_status not in (0, 2):
                raise OutcomePiperStateError(
                    f"cannot take control while teach_status={teach_status}; stop teaching first"
                )
            # Ownership starts at the first control write, never at a rejected preflight.
            self._control_started = True
            self._arm.set_auto_set_motion_mode_enabled(False)
            self._raise_if_comm_error("disable automatic motion-mode switching")
            self._arm.set_joint_limits_enabled(False)
            self._raise_if_comm_error("disable SDK joint limits")
            self._arm.set_speed_percent(self._safety.motion_speed_percent)
            self._raise_if_comm_error("set frozen motion speed")
            mode_requested_at_s = self._monotonic()
            self._arm.set_motion_mode(self._arm.OPTIONS.MOTION_MODE.J)
            self._raise_if_comm_error("set joint position-velocity mode")
            self._confirm_motion_mode_locked(mode_requested_at_s)

    def _confirm_motion_mode_locked(self, requested_at_s: float) -> None:
        """Wait for receive-time CAN/J confirmation after one mode request."""
        assert self._arm is not None
        deadline = self._monotonic() + self.config.feedback_timeout_s
        last_feedback = "no status received"
        while self._monotonic() < deadline:
            self._raise_if_emergency_stop_requested_locked()
            status, received_s = self._receiver.status()
            self._raise_if_comm_error("confirm joint position-velocity mode")
            if status is not None:
                ctrl_mode, mode_feedback, _, _ = self._controller_status(status)
                timestamp_s = float(status.timestamp)
                if not math.isfinite(timestamp_s) or timestamp_s <= 0:
                    raise OutcomePiperStateError("motion-mode SDK timestamp is invalid")
                now_s = self._monotonic()
                if now_s < requested_at_s or (received_s is not None and received_s > now_s):
                    raise OutcomePiperStateError("motion-mode monotonic timestamp is invalid")
                last_feedback = (
                    f"ctrl_mode=0x{ctrl_mode:02x}, mode_feedback=0x{mode_feedback:02x}, "
                    f"timestamp_s={timestamp_s}"
                )
                if (
                    received_s is not None
                    and received_s >= requested_at_s
                    and ctrl_mode == 0x01
                    and mode_feedback == 0x01
                    and self._monotonic() < deadline
                ):
                    return
            remaining = deadline - self._monotonic()
            if remaining > 0:
                self._emergency_stop_requested.wait(min(0.005, remaining))
        raise OutcomePiperStateError(
            f"motion-mode feedback confirmation timed out: {last_feedback}"
        )

    def _controller_status(self, status: Any) -> tuple[int, int, int, int]:
        ctrl_mode = int(status.msg.ctrl_mode)
        mode_feedback = int(status.msg.mode_feedback)
        arm_status = int(status.msg.arm_status)
        arm_error_code = int(status.msg.err_code)
        if arm_status == 1:
            self._latch(PiperState.E_STOP, "controller reports emergency stop")
        if arm_status != 0 or arm_error_code != 0:
            self._latch(
                PiperState.FAULT,
                f"controller arm_status={arm_status}, err_code=0x{arm_error_code:04x}",
            )
        return ctrl_mode, mode_feedback, arm_status, arm_error_code

    @staticmethod
    def _cause_text(cause: BaseException | str) -> str:
        if isinstance(cause, BaseException):
            return f"{type(cause).__name__}: {cause}"
        return str(cause)

    def _latch_state_only(self, state: PiperState, cause: BaseException | str) -> None:
        if state not in _TERMINAL_STATES:
            raise ValueError("only terminal fault states may be latched")
        if self._emergency_stop_requested.is_set():
            state, cause = PiperState.E_STOP, self._emergency_stop_cause or cause
        if self._state not in _TERMINAL_STATES or (
            self._emergency_stop_requested.is_set() and self._state is not PiperState.E_STOP
        ):
            self._state = state
            self._latched_cause = self._cause_text(cause)
            if self._teleop is not None:
                self._teleop.stop(state is PiperState.E_STOP)

    @property
    def stop_outcome(self) -> str | None:
        with self._command_lock:
            return self._stop_outcome

    def _send_electronic_stop_locked(self) -> None:
        if self._electronic_stop_attempted or self._safety is None or not self._firmware_verified:
            return
        self._electronic_stop_attempted = True
        self._stop_outcome = "electronic_stop_requested"
        try:
            if self._arm is not None:
                self._arm.electronic_emergency_stop()
                self._raise_if_comm_error("electronic emergency stop")
                self._stop_outcome = "electronic_stop_sent_unverified"
        except Exception as exc:
            self._stop_error = self._cause_text(exc)
            self._stop_outcome = "stop_unknown"

    def _set_latch(self, state: PiperState, cause: BaseException | str) -> None:
        with self._command_lock:
            first = self._state not in _TERMINAL_STATES
            self._latch_state_only(state, cause)
            if (first and self._control_started) or self._emergency_stop_requested.is_set():
                self._send_electronic_stop_locked()
            if self.last_action_telemetry is not None:
                self.last_action_telemetry.update(
                    stop_outcome=self._stop_outcome,
                    stop_error=self._stop_error,
                    hold_command=self._hold_command,
                    fault=self._latched_cause,
                )

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

    def _watchdog_check_locked(self) -> bool:
        if self._state != PiperState.ACTIVE:
            return False
        last = self._last_control_tick_s
        if last is not None and self._monotonic() - last > self._safety.watchdog_timeout_s:
            if self._teleop is None:
                self._set_latch(PiperState.FAULT, "action watchdog expired")
            else:
                self.request_input_fault("control-loop watchdog expired")
            return False
        if self._teleop is not None and self._teleop.state not in (
            TeleopState.RUNNING,
            TeleopState.POSE_MOVING,
        ):
            try:
                if (
                    self._teleop.state in (TeleopState.CENTERING, TeleopState.HOLD_REQUESTED)
                    and self._hold_epoch != self._teleop.epoch
                ):
                    self._feedback()
                    return True
                self._update_hold_locked()
            except Exception as exc:
                self._set_latch(PiperState.FAULT, exc)
                return False
        return True

    def _start_watchdog(self) -> None:
        assert self._safety is not None
        self._watchdog_stop.clear()

        def monitor() -> None:
            interval = min(self._safety.watchdog_timeout_s / 4, 0.05)
            while not self._watchdog_stop.wait(interval):
                with self._command_lock:
                    if self._watchdog_stop.is_set():
                        return
                    if not self._watchdog_check_locked():
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
                self._safety = load_motion_safety(self.config.safety_path)
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
                self._gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
                self._receiver = self._receiver_factory(
                    arm, self._gripper, lambda: self._monotonic()
                )
                if self.config.capture_timing is not None:
                    self._receiver.joint_max_skew_s = self.config.capture_timing.joint_max_skew_s
                    self._receiver.snapshot_wait_s = min(
                        self.config.feedback_timeout_s, self.config.capture_timing.joint_max_skew_s
                    )
                    self._receiver.snapshot_wait_service = (
                        self._raise_if_emergency_stop_requested_locked
                    )
                # Cold-start readiness is separate from runtime staleness. No requests
                # are resent; a quiet bus may still answer the single firmware query.
                self._receiver.wait_ready(self.config.feedback_timeout_s)
                live_firmware = arm.get_firmware(
                    timeout=self.config.feedback_timeout_s,
                    min_interval=0.0,
                )
                self._raise_if_comm_error("read firmware identity")
                if not isinstance(live_firmware, Mapping):
                    raise OutcomePiperValidationError("live firmware identity is unavailable")
                if self.config.execution_mode == "motion":
                    self._firmware_identity = validate_motion_firmware(
                        live_firmware, firmware=self.config.firmware
                    )
                    self._firmware_verified = True
                else:
                    self._firmware_identity = validate_live_firmware_driver(
                        live_firmware,
                        firmware=self.config.firmware,
                    )
                if not self._receiver.wait_ready(self.config.feedback_timeout_s):
                    raise OutcomePiperStateError("initial complete feedback timed out")
                for camera in cameras.values():
                    camera.connect()
                self.cameras = cameras
                self._state = PiperState.CONNECTED
                self.get_observation()
                self.get_servo_status()
            except Exception as exc:
                self._set_latch(PiperState.FAULT, exc)
                for camera in cameras.values():
                    if getattr(camera, "is_connected", False):
                        camera.disconnect()
                if arm.is_connected():
                    arm.disconnect()
                self._arm = None
                self._receiver = None
                self._gripper = None
                self.cameras = {}
                self.camera_input_poll = None
                self.emergency_stop_poll = None
                message = self._latched_message()
                if isinstance(exc, OutcomePiperStateError) and str(exc) == message:
                    raise
                raise OutcomePiperStateError(message) from exc

    def get_servo_status(self) -> ServoFeedback:
        """Read received driver bits; disconnected/stale feedback is UNKNOWN."""
        with self._command_lock:
            if (
                self._arm is None
                or self._receiver is None
                or not self._arm.is_connected()
                or self._arm.has_comm_error()
            ):
                return ServoFeedback()
            states = self._receiver.driver_states()
            snapshot = self._receiver.snapshot()
            now = self._monotonic()
            flags, stamps, errors = [], [], []
            for state, stamp in states:
                fresh = (
                    state is not None
                    and stamp is not None
                    and math.isfinite(stamp)
                    and 0 <= now - stamp <= self.config.feedback_timeout_s
                )
                flags.append(bool(state.msg.foc_status.driver_enable_status) if fresh else None)
                stamps.append(stamp if fresh else None)
                errors.append(bool(state.msg.foc_status.driver_error_status) if fresh else None)
            if len(flags) != 6:
                flags, stamps, errors = [None] * 6, [None] * 6, [None] * 6
            stamp = snapshot.received_s[4]
            fresh = (
                snapshot.gripper is not None
                and math.isfinite(stamp)
                and 0 <= now - stamp <= self.config.feedback_timeout_s
            )
            grip = bool(snapshot.gripper.msg.foc_status.driver_enable_status) if fresh else None
            result = ServoFeedback(
                joints=tuple(flags),
                gripper=grip,
                driver_errors=tuple(errors),
                received_monotonic_s=tuple([*stamps, stamp if fresh else None]),
            )
            if result.state is not ServoState.UNKNOWN and grip is not None:
                self.last_servo_feedback = result
            return result

    def enable(self) -> None:
        """Explicitly prepare motion and confirm joint power; never enable gripper implicitly."""
        with self._command_lock:
            if self._state is PiperState.ACTIVE:
                self._feedback()
                return
            if self.config.execution_mode != "motion" or self._state != PiperState.CONNECTED:
                raise OutcomePiperStateError("enable requires a connected motion session")
            try:
                self.configure()
                self._motion_configured = True
                self._feedback()
                status = self.get_servo_status()
                if status.state is ServoState.UNKNOWN:
                    raise OutcomePiperStateError("joint enable feedback unavailable")
                requested = self._monotonic()
                self.last_servo_command = {
                    "operation": "enable",
                    "requested_s": requested,
                    "sent": False,
                    "result": "pending",
                }
                if status.state is not ServoState.ENABLED:
                    self.last_servo_command["sent"] = True
                    self._arm.enable()
                    self._raise_if_comm_error("enable")
                    self._confirm_enabled_locked(requested)
                self.last_servo_command.update(result="confirmed", ended_s=self._monotonic())
                self._state = PiperState.ACTIVE
                self._last_action_at = self._monotonic()
                self._last_control_tick_s = self._last_action_at
                if self._teleop is not None:
                    self._teleop.prepare_enable()
                    self._start_hold_locked()
                self._start_watchdog()
                register_active_motion_session(self)
            except Exception as exc:
                if self.last_servo_command is not None:
                    self.last_servo_command.update(result="failed", error=str(exc))
                self._latch(PiperState.FAULT, exc)

    def disable(self, *, include_gripper: bool) -> None:
        """Explicit loss of joint torque; caller must choose whether to release gripper torque."""
        if type(include_gripper) is not bool:
            raise ValueError("include_gripper must be explicitly true or false")
        with self._command_lock:
            if self.config.execution_mode != "motion" or self._state not in {
                PiperState.CONNECTED,
                PiperState.ACTIVE,
            }:
                raise OutcomePiperStateError(
                    "disable requires a connected, non-faulted motion session"
                )
            self._state = PiperState.CONNECTED
            self._motion_configured = False
            self._control_started = False
            self._watchdog_stop.set()
            if self._teleop is not None:
                self._teleop.prepare_enable()
        self._stop_watchdog()
        with self._command_lock:
            requested = self._monotonic()
            self.last_servo_command = {
                "operation": "disable",
                "include_gripper": include_gripper,
                "requested_s": requested,
                "result": "pending",
            }
            try:
                self._feedback()
                initial = self.get_servo_status()
                if initial.state is ServoState.UNKNOWN or (
                    include_gripper and initial.gripper is None
                ):
                    raise OutcomePiperStateError("disable feedback unavailable")
                joint_sent = initial.state is not ServoState.DISABLED
                grip_sent = include_gripper and initial.gripper
                self.last_servo_command.update(joint_sent=joint_sent, gripper_sent=grip_sent)
                self._raise_if_emergency_stop_requested_locked()
                if joint_sent:
                    self._arm.disable()
                    self._raise_if_comm_error("disable joints")
                self._raise_if_emergency_stop_requested_locked()
                if grip_sent:
                    self._gripper.disable_gripper()
                    self._raise_if_comm_error("disable gripper")
                deadline = requested + self.config.feedback_timeout_s
                while True:
                    self._raise_if_emergency_stop_requested_locked()
                    self._feedback()
                    status = self.get_servo_status()
                    joints_done = status.state is ServoState.DISABLED and (
                        not joint_sent
                        or all(
                            t is not None and t >= requested
                            for t in status.received_monotonic_s[:6]
                        )
                    )
                    grip_done = not include_gripper or (
                        status.gripper is False
                        and (not grip_sent or status.received_monotonic_s[6] >= requested)
                    )
                    if joints_done and grip_done and self._monotonic() < deadline:
                        break
                    if self._monotonic() >= deadline:
                        raise OutcomePiperStateError(
                            "disable confirmation timed out; no command was resent"
                        )
                    self._emergency_stop_requested.wait(0.005)
                if include_gripper:
                    self._last_gripper_target = None
                    self._last_gripper_command = None
                    if self._teleop is not None:
                        self._teleop.gripper_target = None
                if self._teleop is not None:
                    self._teleop.joint_target = None
                self.last_servo_command.update(result="confirmed", ended_s=self._monotonic())
            except Exception as exc:
                self.last_servo_command.update(result="failed", error=str(exc))
                self._latch_state_only(PiperState.FAULT, exc)
                raise OutcomePiperStateError(self._latched_message()) from exc

    def _confirm_enabled_locked(self, requested_s: float) -> None:
        deadline = requested_s + self.config.feedback_timeout_s
        while self._monotonic() < deadline:
            self._raise_if_emergency_stop_requested_locked()
            self._feedback()
            states = self._receiver.driver_states()
            now = self._monotonic()
            if now < requested_s:
                raise OutcomePiperStateError("enable confirmation monotonic clock moved backwards")
            confirmed = len(states) == 6
            for state, received_s in states:
                if state is None or received_s is None:
                    confirmed = False
                    continue
                if not math.isfinite(received_s) or received_s < 0 or received_s > now:
                    raise OutcomePiperStateError("invalid enable feedback receive timestamp")
                if state.msg.foc_status.driver_error_status:
                    raise OutcomePiperStateError("driver fault during enable confirmation")
                if received_s < requested_s or not state.msg.foc_status.driver_enable_status:
                    confirmed = False
            if confirmed and self._monotonic() < deadline:
                return
            remaining = deadline - self._monotonic()
            if remaining > 0:
                self._emergency_stop_requested.wait(min(0.005, remaining))
        raise OutcomePiperStateError("enable confirmation timed out; no command was resent")

    @measured("feedback")
    def _feedback(self) -> tuple[list[float], float]:
        if self._state in _TERMINAL_STATES:
            raise OutcomePiperStateError(self._latched_message())
        if self._arm is None or self._gripper is None:
            if self._state in {PiperState.CONNECTED, PiperState.ACTIVE}:
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
        snapshot = self._receiver.snapshot()
        joints, gripper, status = snapshot.joints, snapshot.gripper, snapshot.status
        self._raise_if_comm_error("feedback read")
        if joints is None or gripper is None or status is None:
            self._latch(PiperState.FAULT, "missing arm, status, or gripper feedback")
        frames = snapshot.frames
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
        ctrl_mode, mode_feedback, arm_status, arm_error_code = self._controller_status(status)
        if self._motion_configured and (ctrl_mode != 0x01 or mode_feedback != 0x01):
            self._latch(
                PiperState.FAULT,
                "controller left CAN joint position-velocity mode: "
                f"ctrl_mode=0x{ctrl_mode:02x}, mode_feedback=0x{mode_feedback:02x}",
            )
        joint_timestamps = tuple(float(frame.timestamp) for frame in frames)
        gripper_timestamp = float(gripper.timestamp)
        timestamps = (*joint_timestamps, float(status.timestamp), gripper_timestamp)
        if not all(math.isfinite(value) and value > 0 for value in timestamps):
            self._latch(PiperState.FAULT, "feedback timestamps are missing or invalid")
        now = self._monotonic()
        received = snapshot.received_s
        if not math.isfinite(now) or any(
            not math.isfinite(t) or t < 0 or t > now for t in received
        ):
            self._latch(
                PiperState.FAULT, "feedback monotonic timestamp is invalid or in the future"
            )
        age = max(now - t for t in received)
        if age > self.config.feedback_timeout_s:
            self._latch(PiperState.FAULT, f"feedback is stale by {age:.6f}s")
        if self.config.capture_timing is not None:
            skew = max(received[:3]) - min(received[:3])
            if skew > self.config.capture_timing.joint_max_skew_s:
                self._latch(
                    PiperState.FAULT,
                    f"joint feedback group skew exceeds capture_timing: skew_s={skew:.6f}, "
                    f"limit_s={self.config.capture_timing.joint_max_skew_s:.6f}, "
                    f"received_s={received[:3]}, observed_s={now:.6f}",
                )
        joint_hz = snapshot.joint_hz
        arm_status_hz = float(status.hz)
        gripper_hz = float(gripper.hz)
        if not all(
            math.isfinite(value) and value >= 0 for value in (*joint_hz, arm_status_hz, gripper_hz)
        ):
            self._latch(PiperState.FAULT, "feedback frequency is missing or invalid")
        self._last_feedback = FeedbackTelemetry(
            timestamp_s=min(timestamps),
            received_monotonic_s=received,
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
        servo = self.get_servo_status()
        if self._control_started and any(error is True for error in servo.driver_errors):
            self._latch(PiperState.FAULT, "motor driver fault in servo feedback")
        if self._state is PiperState.ACTIVE and servo.state is not ServoState.ENABLED:
            self._latch(PiperState.FAULT, "joint enable feedback lost or not all enabled")
        return values, width

    def _service_camera_wait(self):
        raw = self.camera_input_poll() if self.camera_input_poll is not None else None
        if raw is not None and raw.get("emergency_stop"):
            self.request_emergency_stop("Xbox B pressed while waiting for camera")
        with self._command_lock:
            self._raise_if_emergency_stop_requested_locked()
            if self._state in _TERMINAL_STATES:
                raise OutcomePiperStateError(self._latched_message())
            self._feedback()
            if raw is not None and not raw.get("hold", False) and self._teleop is not None:
                if self._teleop.state in (TeleopState.RUNNING, TeleopState.POSE_MOVING):
                    self._teleop.request_hold()
                    self._start_hold_locked()
            if self._state is PiperState.ACTIVE:
                if not self._watchdog_check_locked():
                    raise OutcomePiperStateError(self._latched_message())
                # A bounded camera wait services input and live feedback, not only time.
                self._last_control_tick_s = self._monotonic()

    def get_observation(self) -> dict[str, Any]:
        reset()
        with self._command_lock:
            self.last_action_telemetry = None
            if not self.is_connected:
                raise OutcomePiperStateError("PiPER is not connected")
        try:
            with self._command_lock:
                control_only = self._teleop is not None and self._teleop.recording_phase in (
                    "review",
                    "saving",
                    "finalizing",
                )
                self.last_depth_frames = {}
                cameras = tuple(self.cameras.items())
            images, camera_metadata = {}, {}
            if not control_only:
                for name, camera in cameras:
                    try:
                        if not camera.is_connected:
                            raise OutcomePiperCameraError(f"camera {name!r} is disconnected")
                        timing = self.config.capture_timing
                        camera.max_frame_age_s = None if timing is None else timing.camera_max_age_s
                        camera.wait_service = self._service_camera_wait
                        with span("camera_read"):
                            frames, metadata = camera.read_with_metadata(
                                self.config.feedback_timeout_s
                            )
                    except (OutcomePiperStateError,):
                        raise
                    except Exception as exc:
                        raise OutcomePiperCameraError(f"camera {name} read failed: {exc}") from exc
                    try:
                        for stream, frame in frames.items():
                            key = name if stream == "color" else f"{name}.depth"
                            meta = metadata[stream]
                            if stream == "depth":
                                import numpy as np

                                raw = frame.copy()
                                self.last_depth_frames[key] = raw
                                images[key] = (raw.astype(np.float32) * meta.depth_scale_m)[
                                    ..., None
                                ]
                            else:
                                images[key] = (
                                    frame[..., ::-1].copy()
                                    if meta.pixel_format == "bgr8"
                                    else frame
                                )
                            camera_metadata[key] = {
                                **asdict(meta),
                                "observation_format": "depth_m_float32"
                                if stream == "depth"
                                else "rgb8",
                                "read_wait": getattr(camera, "last_read_diagnostics", {}),
                            }
                    except Exception as exc:
                        raise OutcomePiperCameraError(
                            f"camera {name} image processing failed: {exc}"
                        ) from exc
            with self._command_lock:
                self._raise_if_emergency_stop_requested_locked()
                joints, width = self._feedback()
                now = self._monotonic()
                received = self._last_feedback.received_monotonic_s
                ages = [now - t for t in received]
                timing = self.config.capture_timing
                for camera_key, metadata in camera_metadata.items():
                    camera_t = metadata["received_monotonic_s"]
                    age = now - camera_t
                    if not math.isfinite(age) or age < 0:
                        raise OutcomePiperCameraError("camera monotonic timestamp is invalid")
                    ages.append(age)
                    if timing is not None:
                        if age > timing.camera_max_age_s:
                            raise OutcomePiperCameraError(
                                f"camera frame is stale: stream={camera_key}, "
                                f"frame={metadata['frame_number']}, age_s={age:.6f}, "
                                f"limit_s={timing.camera_max_age_s:.6f}, "
                                f"received_s={camera_t:.6f}, "
                                f"published_s={metadata['published_monotonic_s']:.6f}, "
                                f"observed_s={now:.6f}"
                            )
                        if max(abs(camera_t - t) for t in received) > timing.image_state_max_skew_s:
                            raise OutcomePiperCameraError("image-state skew exceeds capture_timing")
                if (
                    self._teleop is not None
                    and self._teleop.state is TeleopState.POSE_MOVING
                    and self._teleop.pose_sequence is not None
                ):
                    self._teleop.pose_sequence.observe(joints, received[:3], now, width)
                self._observation_sequence += 1
                self.last_observation_telemetry = {
                    "sequence": self._observation_sequence,
                    "observed_monotonic_s": now,
                    "oldest_received_monotonic_s": now - max(ages),
                    "feedback": asdict(self._last_feedback),
                    "cameras": camera_metadata,
                    "quality": "control_only"
                    if control_only
                    else ("checked" if timing is not None else "measurement_only"),
                }
                if self.control_trace is not None:
                    self.control_trace.append(
                        {
                            "event": "observation",
                            "monotonic_s": now,
                            "sequence": self._observation_sequence,
                            "values": {**dict(zip(JOINT_KEYS, joints)), "gripper.pos": width},
                            "feedback": asdict(self._last_feedback),
                            "teleop_state": None
                            if self._teleop is None
                            else self._teleop.state.value,
                        }
                    )
                return {
                    **dict(zip(JOINT_KEYS, joints, strict=True)),
                    "gripper.pos": width,
                    **images,
                }
        except OutcomePiperCameraError as exc:
            if self._state is PiperState.ACTIVE:
                if self._teleop is not None:
                    self.request_input_fault(exc)
                else:
                    self._set_latch(PiperState.FAULT, exc)
            else:
                self._latch_state_only(PiperState.FAULT, exc)
            raise OutcomePiperCameraError(self._latched_message()) from exc
        except Exception as exc:
            self._latch(PiperState.FAULT, exc)

    def _check_observation_age(self):
        if self.last_observation_telemetry is None:
            self._latch(PiperState.FAULT, "action requires an observation")
        age = self._monotonic() - self.last_observation_telemetry["oldest_received_monotonic_s"]
        limit = (
            self.config.capture_timing.observation_max_age_s
            if self.config.capture_timing is not None
            else self.config.feedback_timeout_s
        )
        if not math.isfinite(age) or age < 0 or age > limit:
            cause = "observation expired before SDK command"
            if self._teleop is not None:
                self.request_input_fault(cause)
                raise OutcomePiperStateError(self._latched_message())
            self._latch(PiperState.FAULT, cause)

    @property
    def teleop_state(self):
        return None if self._teleop is None else self._teleop.state

    def configure_teleoperation(self, control: TeleopControl, settings: HoldSettings) -> None:
        with self._command_lock:
            if self._state is not PiperState.DISCONNECTED or self.config.execution_mode != "motion":
                raise OutcomePiperStateError(
                    "configure teleoperation before connecting a motion session"
                )
            self._teleop, self._hold_settings = control, settings
            control.hold_settings = settings

    @measured("hold_capture")
    def _start_hold_locked(self) -> None:
        self._require_active_locked("capture hold")
        self._raise_if_emergency_stop_requested_locked()
        joints, width = self._feedback()
        # Capture measured joints normally. For a tolerated boundary excursion,
        # explicitly hold the legal boundary, not an inward target that may have
        # advanced since the last measurement. Raw feedback remains in telemetry.
        measured_joints = list(joints)
        boundary_events = check_joint_feedback(
            joints,
            self._safety,
            target=self._teleop.joint_target,
            tolerance=self._hold_settings.joint_tolerance_rad,
        )
        if boundary_events:
            for event in boundary_events:
                joints[event["joint"] - 1] = event["boundary_rad"]
            logging.info(
                "Xbox 边界保持：反馈=%s，合法保持目标=%s，偏差=%s",
                measured_joints,
                joints,
                boundary_events,
            )
        if self._teleop.orientation_target is None:
            from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
            from scipy.spatial.transform import Rotation

            pose = fk_from_mdh(list(get_mdh("piper")), joints)
            self._teleop.initialize_orientation(Rotation.from_euler("xyz", pose[3:]).as_matrix())
        requested = self._monotonic()
        command = {
            "name": "hold_move_j",
            "measured_joint_rad": measured_joints,
            "feedback_limit_events": boundary_events,
            "target": list(joints),
            "started_monotonic_s": requested,
            "result": "failed",
        }
        self._hold_window = JointHold(joints, self._hold_settings, requested)
        self._hold_command = command
        self._hold_id += 1
        self._hold_epoch = self._teleop.epoch
        try:
            with span("sdk_move_j"):
                self._arm.move_j(joints)
            self._raise_if_comm_error("hold_move_j")
            self._raise_if_emergency_stop_requested_locked()
            command["result"] = "sdk_returned"
            self._teleop.joint_target = tuple(joints)
            logging.info("Xbox 保持目标：%s", [round(math.degrees(q), 3) for q in joints])
            logging.info("[保持] 正在确认位置稳定…")
        finally:
            command["ended_monotonic_s"] = self._monotonic()

    @measured("hold_confirmation")
    def _update_hold_locked(self) -> None:
        self._require_active_locked("confirm hold")
        self._raise_if_emergency_stop_requested_locked()
        joints, _ = self._feedback()
        check_joint_feedback(
            joints,
            self._safety,
            target=self._hold_window.target,
            tolerance=self._hold_settings.joint_tolerance_rad,
        )
        confirmed = self._hold_window.observe(
            joints, self._last_feedback.received_monotonic_s[:3], self._monotonic()
        )
        if self._teleop.hold_confirmed and not confirmed:
            self._teleop.reconfirm_hold()
            self._hold_epoch = self._teleop.epoch
        elif confirmed and not self._teleop.hold_confirmed:
            from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
            from scipy.spatial.transform import Rotation

            pose = fk_from_mdh(list(get_mdh("piper")), self._hold_window.target)
            self._teleop.confirm_hold(Rotation.from_euler("xyz", pose[3:]).as_matrix())
            self._stop_outcome = "hold_confirmed"
            phase = self._teleop.recording_phase
            hint = (
                "本回合已结束，请选择保存或重录；当前不接受遥操作。"
                if phase == "review"
                else "正在保存，请等待；当前不接受遥操作，B急停仍有效。"
                if phase == "saving"
                else "正在完成退出，请等待；B急停仍有效。"
                if phase == "finalizing"
                else "请继续松开LB、输入回中，等待模式就绪提示。"
                if self._teleop.pending_mode is not None
                else "LB可保持按住，直接推杆继续；扳机独立控制夹爪"
                if self._teleop.state is TeleopState.CENTERED
                else f"松开{'A' if self._teleop.pose_kind == 'work' else 'Y'}，保持摇杆/扳机回中，按住LB执行所选姿态"
                if self._teleop.state is TeleopState.POSE_READY
                else "全部回中后重新按LB，再推杆启动"
            )
            logging.info("Xbox 保持状态：%s", self._teleop.state.value)
            logging.info("[保持已确认] %s", hint)

    def _teleop_run_allowed_locked(self, action: OutcomePiperAction) -> bool:
        self._raise_if_emergency_stop_requested_locked()
        if (
            action.intent not in {"run", "pose", "hold", "wait", "center"}
            or type(action.epoch) is not int
        ):
            self._latch(PiperState.FAULT, "invalid teleoperation intent")
        if action.epoch != self._teleop.epoch:
            return False
        age = self._monotonic() - action.generated_monotonic_s
        limit = (
            self.config.capture_timing.observation_max_age_s
            if self.config.capture_timing
            else self.config.feedback_timeout_s
        )
        if not math.isfinite(age) or age < 0:
            self._latch(PiperState.FAULT, "teleoperation action generation time is invalid")
        if age > limit:
            self.request_input_fault("teleoperation control tick expired")
            raise OutcomePiperStateError(self._latched_message())
        self._last_control_tick_s = self._monotonic()
        if (
            self._input_fault_requested.is_set()
            or action.intent not in ("run", "pose")
            or (action.intent == "pose") != (self._teleop.state is TeleopState.POSE_MOVING)
            or not self._teleop.permits(action.epoch)
        ):
            return False
        if self._running_epoch != action.epoch:
            joints, _ = self._feedback()
            if not self._teleop.hold_confirmed or any(
                abs(a - b) > self._hold_settings.joint_tolerance_rad
                for a, b in zip(joints, self._hold_window.target, strict=True)
            ):
                self._teleop.request_hold()
                self._hold_epoch = self._teleop.epoch
                self._hold_window.restart(self._monotonic())
                return False
            self._running_epoch = action.epoch
            if action.intent == "pose":
                logging.info("[预设姿态执行中] 持续按住LB；松LB或偏转摇杆/扳机取消。")
            else:
                logging.info("[运行中] 松开LB或让摇杆回中可保持。")
        return True

    def _teleop_idle_locked(self, action):
        try:
            center_tick = (
                action.intent == "center"
                and action.epoch == self._teleop.epoch
                and self._teleop.state in (TeleopState.CENTERING, TeleopState.CENTERED)
            )
            if (action.intent == "center" and not center_tick) or (
                self._teleop.state in (TeleopState.RUNNING, TeleopState.POSE_MOVING)
            ):
                self.last_action_telemetry.update(
                    result="discarded", reason=action.rejection_reason or "stale control epoch"
                )
                return dict(action)
            if center_tick:
                self._action_values(action)
            if (
                self._teleop.hold_confirmed
                and self._hold_window is not None
                and action.epoch == self._teleop.epoch
                and (self._teleop.mode_event or {}).get("accepted")
            ):
                # Mode selection invalidates motion, not the already confirmed hold command.
                self._hold_epoch = self._teleop.epoch
            new_command = (
                (
                    self._teleop.state is TeleopState.CENTERING
                    and self._hold_epoch != self._teleop.epoch
                )
                if center_tick
                else (self._hold_epoch != self._teleop.epoch)
            )
            if new_command:
                self._start_hold_locked()
                self.last_action_telemetry["commands"].append(dict(self._hold_command))
            self._update_hold_locked()
            if (
                center_tick
                and action.epoch == self._teleop.epoch
                and action.gripper_input
                and not self._input_fault_requested.is_set()
            ):
                self._raise_if_emergency_stop_requested_locked()
                held_joints, width = self._feedback()
                try:
                    if (
                        action.gripper_plan is not None
                        and "reference" in action.gripper_plan
                        and not self._teleop.gripper_reference_valid(
                            action.epoch, action.gripper_plan
                        )
                    ):
                        raise OutcomePiperIntentRejected("stale gripper reference")
                    target = self._validate_gripper_target(
                        float(action["gripper.pos"]), width, current_joints=held_joints
                    )
                except OutcomePiperIntentRejected as exc:
                    self.last_action_telemetry["rejection_reason"] = str(exc)
                    self._teleop.request_hold()
                    self._start_hold_locked()
                    self.last_action_telemetry["commands"].append(dict(self._hold_command))
                    self._update_hold_locked()
                else:
                    if target != self._last_gripper_target:
                        self._dispatch_gripper_locked(target)
                        self._hold_id += 1  # New joint/gripper reference pair, same joint command.
                        self._raise_if_emergency_stop_requested_locked()
            if (
                center_tick
                and not self._input_fault_requested.is_set()
                and self._last_gripper_target == float(action["gripper.pos"])
            ):
                self.last_action_telemetry["gripper_reference_committed"] = (
                    self._teleop.commit_gripper_reference(action.epoch, action.gripper_plan)
                )
            values = (
                None
                if self._last_gripper_target is None
                else {
                    **dict(zip(JOINT_KEYS, self._hold_window.target)),
                    "gripper.pos": self._last_gripper_target,
                }
            )
            self.last_action_telemetry.update(
                result="waiting" if values is None else "holding",
                values=values,
                control_state=self._teleop.state.value,
                waiting_for_new_input=center_tick and self._teleop.state is TeleopState.CENTERED,
                arm_centered=center_tick,
                gripper_input=action.gripper_input if center_tick else False,
                reason=action.rejection_reason,
                orientation_target=self._teleop.orientation_target,
                control_epoch=self._teleop.epoch,
                hold_id=self._hold_id,
                hold_command=dict(self._hold_command),
                retained_gripper_command=None
                if self._last_gripper_command is None
                else dict(self._last_gripper_command),
                hold_confirmed=self._teleop.hold_confirmed,
                commands=list(self.last_action_telemetry["commands"]),
            )
            return dict(action) if values is None else values
        except Exception as exc:
            self._latch(PiperState.FAULT, exc)

    def request_input_fault(self, cause: BaseException | str) -> None:
        """Hold on confirmed input loss, then lock the session; no automatic reconnect."""
        if self._teleop is None:
            self.request_emergency_stop(cause)
            return
        self._input_fault_requested.set()
        if self._teleop.state in (
            TeleopState.RUNNING,
            TeleopState.POSE_MOVING,
            TeleopState.POSE_READY,
            TeleopState.CENTERING,
            TeleopState.CENTERED,
        ):
            self._teleop.request_hold()
        with self._command_lock:
            if self._state in _TERMINAL_STATES:
                return
            try:
                self._raise_if_emergency_stop_requested_locked()
                if self._teleop.state in (
                    TeleopState.RUNNING,
                    TeleopState.POSE_MOVING,
                    TeleopState.POSE_READY,
                    TeleopState.CENTERING,
                    TeleopState.CENTERED,
                ):
                    self._teleop.request_hold()
                if self._hold_epoch != self._teleop.epoch:
                    self._start_hold_locked()
                while True:
                    if (
                        threading.get_ident() == self._input_thread_id
                        and self.emergency_stop_poll is not None
                        and self.emergency_stop_poll()
                    ):
                        self.request_emergency_stop("Xbox B pressed during fault hold confirmation")
                    self._update_hold_locked()
                    if self._teleop.hold_confirmed:
                        break
                    self._emergency_stop_requested.wait(
                        min(0.005, max(0.0, self._hold_window.deadline - self._monotonic()))
                    )
                self._stop_outcome = "hold_confirmed"
                self._latch_state_only(PiperState.FAULT, cause)
            except Exception as exc:
                self._set_latch(PiperState.FAULT, exc)
            if self.last_action_telemetry is not None:
                self.last_action_telemetry.update(
                    input_fault=self._cause_text(cause), stop_outcome=self._stop_outcome
                )

    @staticmethod
    def _action_values(action: Mapping[str, Any]) -> list[float]:
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
        return values

    def _validate_action(
        self, action: Mapping[str, Any], *, joints_only=False
    ) -> tuple[list[float], float]:
        values = self._action_values(action)
        assert self._safety is not None
        current_joints, current_gripper = self._feedback()
        if self._teleop is not None and getattr(action, "intent", None) == "pose":
            sequence = self._teleop.pose_sequence
            if sequence is None or sequence.control_epoch != action.epoch:
                raise OutcomePiperIntentRejected("pose plan is no longer valid")
            if sequence.window is None and sequence.index == 0:
                sequence.validate_start(current_joints, current_gripper)
        check_joint_feedback(
            current_joints,
            self._safety,
            target=None if self._teleop is None else self._teleop.joint_target,
            tolerance=0.0
            if self._hold_settings is None
            else self._hold_settings.joint_tolerance_rad,
        )
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
                raise OutcomePiperIntentRejected(f"joint_{index} target is outside frozen limits")
            if not step_within_limit(target, current, step):
                raise OutcomePiperIntentRejected(f"joint_{index} target exceeds frozen step limit")
        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

        target_pose = fk_from_mdh(list(get_mdh("piper")), values[:6])
        target_xyz = tuple(float(value) for value in target_pose[:3])
        if len(target_xyz) != 3 or not all(math.isfinite(value) for value in target_xyz):
            raise OutcomePiperValidationError("target FK did not produce a finite XYZ position")
        current_pose = fk_from_mdh(list(get_mdh("piper")), current_joints)
        checked_gripper = (
            current_gripper
            if joints_only
            else self._validate_gripper_target(
                values[6], current_gripper, current_joints=current_joints
            )
        )
        if self.last_action_telemetry is not None:
            self.last_action_telemetry["workspace"] = {
                "coordinates": workspace_coordinates(
                    target_pose, checked_gripper, self._safety.workspace_geometry
                ),
                "reference": "flange_xyz"
                if self._safety.workspace_geometry is None
                else "tip_xy_table_height",
            }
        if not workspace_pose_allowed(
            current_pose,
            target_pose,
            current_gripper,
            checked_gripper,
            self._safety,
            allow_reentry=self._teleop is not None,
        ):
            raise OutcomePiperIntentRejected(
                "action target is outside the frozen workspace without inward progress"
            )
        return values[:6], checked_gripper

    def _validate_gripper_target(self, gripper, current_gripper, *, current_joints=None):
        if not math.isfinite(gripper):
            raise OutcomePiperValidationError("gripper target must be finite")
        if not self._safety.gripper_lower <= gripper <= self._safety.gripper_upper:
            raise OutcomePiperIntentRejected("gripper target is outside frozen limits")
        retained_gripper = self._teleop is not None and gripper == self._last_gripper_target
        if not retained_gripper and not step_within_limit(
            gripper, current_gripper, self._safety.max_gripper_step
        ):
            raise OutcomePiperIntentRejected("gripper target exceeds frozen step limit")
        if (
            self._safety.workspace_geometry is not None
            and gripper != current_gripper
            and not retained_gripper
        ):
            from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

            if current_joints is None:
                current_joints, _ = self._feedback()
            pose = fk_from_mdh(list(get_mdh("piper")), current_joints)
            if not gripper_table_allowed(pose, current_gripper, gripper, self._safety):
                raise OutcomePiperIntentRejected(
                    "gripper change violates tool workspace/table clearance"
                )
        return gripper

    def observe_control(self) -> dict[str, Any]:
        """Service an active control tick without resending a position command.

        Used when observing one already-dispatched target. Feedback/mode checks
        still run; failed/late observations cannot revive a faulted session.
        """
        observation = self.get_observation()
        with self._command_lock:
            self._require_active_locked("observe control")
            try:
                if not self._watchdog_check_locked():
                    raise OutcomePiperStateError(
                        "control-loop watchdog expired before observation tick"
                    )
                self._raise_if_emergency_stop_requested_locked()
                self._check_observation_age()
                check_joint_feedback(
                    [observation[k] for k in JOINT_KEYS],
                    self._safety,
                    target=None if self._teleop is None else self._teleop.joint_target,
                    tolerance=0.0
                    if self._hold_settings is None
                    else self._hold_settings.joint_tolerance_rad,
                )
                self._last_control_tick_s = self._monotonic()
            except Exception as exc:
                self._latch(PiperState.FAULT, exc)
        return observation

    def send_joint_target(self, joints) -> dict[str, float]:
        """Move joints in a dedicated preparation session, without any gripper write."""
        with self._command_lock:
            self._require_active_locked("joint-only preparation")
            if self._teleop is not None:
                raise OutcomePiperStateError(
                    "joint-only preparation cannot bypass Xbox intent handling"
                )
            if len(joints) != 6:
                raise OutcomePiperValidationError("joint-only target requires six angles")
            _, width = self._feedback()
            action = {**dict(zip(JOINT_KEYS, joints)), "gripper.pos": width}
            try:
                return self._send_action(action, joints_only=True)
            finally:
                if self.control_trace is not None:
                    self.control_trace.append(
                        {
                            "event": "joint_preparation_action",
                            "joint_target": list(joints),
                            "telemetry": self.last_action_telemetry,
                        }
                    )

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        if self.control_trace is None:
            return self._send_action(action)
        started = self._monotonic()
        result = "raised"
        try:
            returned = self._send_action(action)
            result = "returned"
            return returned
        finally:
            self.control_trace.append(
                {
                    "event": "action",
                    "started_s": started,
                    "ended_s": self._monotonic(),
                    "call_result": result,
                    "stage_timing": snapshot(),
                    "recording_phase": None
                    if self._teleop is None
                    else self._teleop.recording_phase,
                    "requested": dict(action),
                    "intent": getattr(action, "intent", None),
                    "epoch": getattr(action, "epoch", None),
                    "teleop_state": None if self._teleop is None else self._teleop.state.value,
                    "telemetry": self.last_action_telemetry,
                }
            )

    def _send_action(self, action: dict[str, Any], *, joints_only=False) -> dict[str, float]:
        with self._command_lock:
            if self.config.execution_mode != "motion":
                raise OutcomePiperStateError("send_action requires execution_mode=motion")
            self._require_active_locked("send_action")
            self.last_action_telemetry = {
                "observation_sequence": self.last_observation_telemetry["sequence"]
                if self.last_observation_telemetry
                else None,
                "generated_monotonic_s": getattr(action, "generated_monotonic_s", None),
                "dispatch_monotonic_s": self._monotonic(),
                "commands": [],
                "result": "rejected",
                "rejection_reason": getattr(action, "rejection_reason", None),
                "joint_plan": getattr(action, "joint_plan", None),
                "gripper_plan": getattr(action, "gripper_plan", None),
                "pose_plan": getattr(action, "pose_plan", None),
                "reference_plan": getattr(action, "reference_plan", None),
                "feedback_limit_events": getattr(action, "feedback_limit_events", []),
                "pose_event": None if self._teleop is None else self._teleop.pose_event,
                "pose_kind": None if self._teleop is None else self._teleop.pose_kind,
                "pose_cancel_reason": None
                if self._teleop is None
                else self._teleop.pose_cancel_reason,
                "teleop_mode": None if self._teleop is None else self._teleop.mode.value,
                "pending_mode": None
                if self._teleop is None or self._teleop.pending_mode is None
                else self._teleop.pending_mode.value,
                "translation_strategy": None
                if self._teleop is None
                else self._teleop.translation_strategy.value,
                "control_point": "grasp_center",
                "mode_event": None if self._teleop is None else self._teleop.mode_event,
                "orientation_candidate": getattr(action, "orientation_target", None),
                "orientation_target": None
                if self._teleop is None
                else self._teleop.orientation_target,
            }
            if self._teleop is None and type(action) is OutcomePiperAction:
                raise OutcomePiperStateError("Xbox control must be configured before dispatch")
            if self._teleop is not None:
                if type(action) is not OutcomePiperAction:
                    self._latch(PiperState.FAULT, "Xbox session requires control intent metadata")
                if not self._teleop_run_allowed_locked(action):
                    return self._teleop_idle_locked(action)
            if self._teleop is not None and action.intent == "pose":
                sequence = self._teleop.pose_sequence
                if sequence is None or dict(action) != sequence.values:
                    self._latch(
                        PiperState.FAULT, "home waypoint does not match the current sequence"
                    )
            try:
                if (
                    self._teleop is not None
                    and action.gripper_plan is not None
                    and "reference" in action.gripper_plan
                    and not self._teleop.gripper_reference_valid(action.epoch, action.gripper_plan)
                ):
                    raise OutcomePiperIntentRejected("stale gripper reference")
                if (
                    self._teleop is not None
                    and action.reference_plan is not None
                    and not self._teleop.reference_valid(action.epoch, action.reference_plan)
                ):
                    raise OutcomePiperIntentRejected("stale continuous reference")
                joints, gripper = (
                    self._validate_action(action, joints_only=True)
                    if joints_only
                    else self._validate_action(action)
                )
            except OutcomePiperIntentRejected as exc:
                if self._teleop is None:
                    raise
                self._teleop.request_hold()
                self.last_action_telemetry["rejection_reason"] = str(exc)
                return self._teleop_idle_locked(action)
            except Exception as exc:
                if self._teleop is not None:
                    self._latch(PiperState.FAULT, exc)
                raise
            command = None
            try:
                self._require_active_locked("move_j")
                assert self._arm is not None
                self._check_observation_age()
                command = {
                    "name": "move_j",
                    "target": list(joints),
                    "started_monotonic_s": self._monotonic(),
                    "result": "failed",
                }
                self.last_action_telemetry["commands"].append(command)
                with span("sdk_move_j"):
                    self._arm.move_j(joints)
                self._raise_if_comm_error("move_j")
                command.update(ended_monotonic_s=self._monotonic(), result="sdk_returned")
                if self._teleop is not None:
                    self._teleop.joint_target = tuple(joints)
            except Exception as exc:
                if command is not None:
                    command["ended_monotonic_s"] = self._monotonic()
                self._latch(PiperState.FAULT, exc)
            self._raise_if_emergency_stop_requested_locked()
            if self._teleop is not None and (
                self._input_fault_requested.is_set() or not self._teleop.permits(action.epoch)
            ):
                return self._teleop_idle_locked(action)
            if not joints_only and (self._teleop is None or gripper != self._last_gripper_target):
                self._dispatch_gripper_locked(gripper)
            self._raise_if_emergency_stop_requested_locked()
            if self._teleop is not None and (
                self._input_fault_requested.is_set() or not self._teleop.permits(action.epoch)
            ):
                return self._teleop_idle_locked(action)
            if self._teleop is not None:
                if action.intent == "pose":
                    sequence = self._teleop.pose_sequence
                    if sequence is None or dict(action) != sequence.values:
                        self._latch(
                            PiperState.FAULT, "home waypoint does not match the current sequence"
                        )
                    sequence.sent(self._monotonic())
                self.last_action_telemetry["gripper_reference_committed"] = (
                    self._teleop.commit_gripper_reference(action.epoch, action.gripper_plan)
                )
                reference_committed = self._teleop.commit_reference(
                    action.epoch, action.reference_plan
                )
                self.last_action_telemetry["reference_committed"] = reference_committed
                committed = self._teleop.commit_orientation(action.epoch, action.orientation_target)
                self.last_action_telemetry.update(
                    orientation_committed=committed,
                    orientation_target=self._teleop.orientation_target,
                )
            self._last_action_at = self._monotonic()
            self._last_control_tick_s = self._last_action_at
            if not joints_only:
                self._last_gripper_target = gripper
            self.last_action_telemetry.update(
                result="sdk_returned",
                values={
                    key: float(action[key]) for key in (JOINT_KEYS if joints_only else ACTION_KEYS)
                },
                gripper_commanded=not joints_only,
                retained_gripper_command=self._last_gripper_command,
            )
            return {key: float(action[key]) for key in (JOINT_KEYS if joints_only else ACTION_KEYS)}

    def prepare_recording_gripper(self, epoch):
        """Explicit start request: retain an existing command or acquire current width."""
        with self._command_lock:
            self._require_active_locked("start recording")
            self._raise_if_emergency_stop_requested_locked()
            if (
                self._teleop is None
                or self._teleop.recording_phase != "preparing"
                or self._teleop.epoch != epoch
                or self._input_fault_requested.is_set()
                or self._teleop.state not in (TeleopState.WAITING, TeleopState.PAUSED)
                or not self._teleop.hold_confirmed
            ):
                raise OutcomePiperIntentRejected("start requires the current confirmed pause")
            self._update_hold_locked()
            if not self._teleop.hold_confirmed or self._teleop.epoch != epoch:
                raise OutcomePiperIntentRejected("joint hold must be confirmed again")
            if self._last_gripper_command is not None:
                return {"initialized": False, "command": dict(self._last_gripper_command)}
            joints, width = self._feedback()
            target = self._validate_gripper_target(width, width, current_joints=joints)
            # Same ordinary dispatcher and observation deadline; no synthesized SDK proof.
            self._dispatch_gripper_locked(target)
            self._raise_if_emergency_stop_requested_locked()
            self._hold_id += 1
            logging.info(
                "[夹爪已接管] 保持目标 %.1f mm，配置保持力 %.1f N。",
                target * 1000,
                self._safety.gripper_force_n,
            )
            return {"initialized": True, "command": dict(self._last_gripper_command)}

    def _dispatch_gripper_locked(self, gripper):
        command = None
        try:
            self._require_active_locked("gripper command")
            assert self._gripper is not None
            assert self._safety is not None
            self._check_observation_age()
            self._raise_if_emergency_stop_requested_locked()
            command = {
                "name": "move_gripper_m",
                "target": gripper,
                "force": self._safety.gripper_force_n,
                "started_monotonic_s": self._monotonic(),
                "result": "failed",
            }
            self.last_action_telemetry["commands"].append(command)
            with span("sdk_gripper"):
                self._gripper.move_gripper_m(gripper, force=self._safety.gripper_force_n)
            self._raise_if_comm_error("gripper command")
            self._last_gripper_target = gripper
            command.update(ended_monotonic_s=self._monotonic(), result="sdk_returned")
            self._last_gripper_command = dict(command)
            if self._teleop is not None:
                self._teleop.gripper_target = gripper
        except Exception as exc:
            if command is not None:
                command["ended_monotonic_s"] = self._monotonic()
            self._latch(PiperState.E_STOP, exc)

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
            self._motion_configured = False
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
                self.camera_input_poll = None
                self.emergency_stop_poll = None
                self._arm = None
                self._receiver = None
                self._gripper = None
                self._control_started = False
                self._last_gripper_target = None
                self._last_gripper_command = None
                if self._teleop is not None:
                    self._teleop.gripper_target = None
                    self._teleop.joint_target = None
                if first_error is not None:
                    raise first_error
