"""Fail-fast standard PiPER LeRobot implementation."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Callable, Mapping

from lerobot.robots.robot import Robot

from .camera import make_timed_cameras
from .config import OutcomePiperConfig
from .timing import FeedbackReceiver
from .errors import OutcomePiperStateError, OutcomePiperValidationError
from .input_safety import register_active_motion_session
from .processor import OutcomePiperAction
from .teleop_control import HoldSettings, JointHold, TeleopControl, TeleopState
from .safety import (
    ACTION_KEYS,
    JOINT_KEYS,
    MotionSafety,
    load_motion_safety,
    validate_live_firmware_driver,
    validate_live_hardware_acceptance,
    validate_teleoperation_hold,
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
        self.last_observation_telemetry = None
        self.last_action_telemetry = None
        self._observation_sequence = 0
        self._arm: Any | None = None
        self._gripper: Any | None = None
        self.cameras: dict[str, Any] = {}
        self._state = PiperState.DISCONNECTED
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
            features[name] = (camera.height, camera.width, 3)
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
            mode_requested_at_s = self._monotonic()
            self._arm.set_motion_mode(self._arm.OPTIONS.MOTION_MODE.J)
            self._raise_if_comm_error("set joint position-velocity mode")
            self._confirm_motion_mode_locked(mode_requested_at_s)
            self._arm.set_speed_percent(self._safety.motion_speed_percent)
            self._raise_if_comm_error("set frozen motion speed")

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
        if (
            self._electronic_stop_attempted
            or self._safety is None
            or not self._hardware_identity_verified
        ):
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
            if first or self._emergency_stop_requested.is_set():
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
        if self._teleop is not None and self._teleop.state is not TeleopState.RUNNING:
            try:
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
                self._gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
                self._receiver = self._receiver_factory(
                    arm, self._gripper, lambda: self._monotonic()
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
                if not self._receiver.wait_ready(self.config.feedback_timeout_s):
                    raise OutcomePiperStateError("initial complete feedback timed out")
                self.configure()
                for camera in cameras.values():
                    camera.connect()
                self.cameras = cameras
                self._state = PiperState.CONNECTED_DISABLED
                self.get_observation()
                if self.config.execution_mode == "motion":
                    enable_requested_s = self._monotonic()
                    # The SDK returns cached flags immediately after sending once.
                    arm.enable()
                    self._raise_if_comm_error("enable")
                    self._confirm_enabled_locked(enable_requested_s)
                    self._state = PiperState.ACTIVE
                    self._last_action_at = self._monotonic()
                    self._last_control_tick_s = self._last_action_at
                    if self._teleop is not None:
                        self._start_hold_locked()
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
                self._receiver = None
                self._gripper = None
                self.cameras = {}
                message = self._latched_message()
                if isinstance(exc, OutcomePiperStateError) and str(exc) == message:
                    raise
                raise OutcomePiperStateError(message) from exc

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
        if self.config.execution_mode == "motion" and (ctrl_mode != 0x01 or mode_feedback != 0x01):
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
                self._latch(PiperState.FAULT, "joint feedback group skew exceeds capture_timing")
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
        return values, width

    def get_observation(self) -> dict[str, Any]:
        with self._command_lock:
            self.last_action_telemetry = None
            if not self.is_connected:
                raise OutcomePiperStateError("PiPER is not connected")
            try:
                images = {}
                camera_metadata = {}
                for name, camera in self.cameras.items():
                    frame, metadata = camera.read_with_metadata(self.config.feedback_timeout_s)
                    images[name] = frame
                    camera_metadata[name] = asdict(metadata)
                joints, width = self._feedback()
                now = self._monotonic()
                received = self._last_feedback.received_monotonic_s
                ages = [now - t for t in received]
                timing = self.config.capture_timing
                for metadata in camera_metadata.values():
                    camera_t = metadata["received_monotonic_s"]
                    age = now - camera_t
                    if not math.isfinite(age) or age < 0:
                        raise OutcomePiperStateError("camera monotonic timestamp is invalid")
                    ages.append(age)
                    if timing is not None:
                        if age > timing.camera_max_age_s:
                            raise OutcomePiperStateError("camera frame is stale")
                        if max(abs(camera_t - t) for t in received) > timing.image_state_max_skew_s:
                            raise OutcomePiperStateError("image-state skew exceeds capture_timing")
                self._observation_sequence += 1
                self.last_observation_telemetry = {
                    "sequence": self._observation_sequence,
                    "observed_monotonic_s": now,
                    "oldest_received_monotonic_s": now - max(ages),
                    "feedback": asdict(self._last_feedback),
                    "cameras": camera_metadata,
                    "quality": "checked" if timing is not None else "measurement_only",
                }
                return {
                    **dict(zip(JOINT_KEYS, joints, strict=True)),
                    "gripper.pos": width,
                    **images,
                }
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
            self._latch(PiperState.FAULT, "observation expired before SDK command")

    @property
    def teleop_state(self):
        return None if self._teleop is None else self._teleop.state

    def configure_teleoperation(self, control: TeleopControl, settings: HoldSettings) -> None:
        with self._command_lock:
            if self._state is not PiperState.DISCONNECTED or self.config.execution_mode != "motion":
                raise OutcomePiperStateError(
                    "configure teleoperation before connecting a motion session"
                )
            validate_teleoperation_hold(self.config.hardware_acceptance_path, settings)
            self._teleop, self._hold_settings = control, settings

    def _start_hold_locked(self) -> None:
        self._require_active_locked("capture hold")
        self._raise_if_emergency_stop_requested_locked()
        joints, width = self._feedback()
        # Validate the captured arm target through the existing limits/FK path.
        # The observed gripper value is validation input only; no gripper command follows.
        self._validate_action({**dict(zip(JOINT_KEYS, joints)), "gripper.pos": width})
        requested = self._monotonic()
        command = {
            "name": "hold_move_j",
            "target": list(joints),
            "started_monotonic_s": requested,
            "result": "failed",
        }
        self._hold_window = JointHold(joints, self._hold_settings, requested)
        self._hold_command = command
        self._hold_id += 1
        self._hold_epoch = self._teleop.epoch
        try:
            self._arm.move_j(joints)
            self._raise_if_comm_error("hold_move_j")
            self._raise_if_emergency_stop_requested_locked()
            command["result"] = "sdk_returned"
        finally:
            command["ended_monotonic_s"] = self._monotonic()

    def _update_hold_locked(self) -> None:
        self._require_active_locked("confirm hold")
        self._raise_if_emergency_stop_requested_locked()
        joints, _ = self._feedback()
        confirmed = self._hold_window.observe(
            joints, self._last_feedback.received_monotonic_s[:3], self._monotonic()
        )
        if self._teleop.hold_confirmed and not confirmed:
            self._teleop.request_hold()
            self._hold_epoch = self._teleop.epoch
        elif confirmed and not self._teleop.hold_confirmed:
            self._teleop.confirm_hold()
            self._stop_outcome = "hold_confirmed"

    def _teleop_run_allowed_locked(self, action: OutcomePiperAction) -> bool:
        self._raise_if_emergency_stop_requested_locked()
        if action.intent not in {"run", "hold", "wait"} or type(action.epoch) is not int:
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
            or action.intent != "run"
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
        return True

    def _teleop_idle_locked(self, action):
        try:
            if self._teleop.state is TeleopState.RUNNING:
                self.last_action_telemetry.update(result="discarded", reason="stale control epoch")
                return dict(action)
            new_command = self._hold_epoch != self._teleop.epoch
            if new_command:
                self._start_hold_locked()
            self._update_hold_locked()
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
                control_epoch=self._teleop.epoch,
                hold_id=self._hold_id,
                hold_command=dict(self._hold_command),
                retained_gripper_command=None
                if self._last_gripper_command is None
                else dict(self._last_gripper_command),
                hold_confirmed=self._teleop.hold_confirmed,
                commands=[
                    *self.last_action_telemetry["commands"],
                    *([dict(self._hold_command)] if new_command else []),
                ],
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
        if self._teleop.state is TeleopState.RUNNING:
            self._teleop.request_hold()
        with self._command_lock:
            if self._state in _TERMINAL_STATES:
                return
            try:
                self._raise_if_emergency_stop_requested_locked()
                if self._teleop.state is TeleopState.RUNNING:
                    self._teleop.request_hold()
                if self._hold_epoch != self._teleop.epoch:
                    self._start_hold_locked()
                while True:
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
        retained_gripper = self._teleop is not None and gripper == self._last_gripper_target
        if not retained_gripper and abs(gripper - current_gripper) > self._safety.max_gripper_step:
            raise OutcomePiperValidationError("gripper target exceeds frozen step limit")
        return values[:6], gripper

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
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
            }
            if self._teleop is None and type(action) is OutcomePiperAction:
                raise OutcomePiperStateError("Xbox control must be configured before dispatch")
            if self._teleop is not None:
                if type(action) is not OutcomePiperAction:
                    self._latch(PiperState.FAULT, "Xbox session requires control intent metadata")
                if not self._teleop_run_allowed_locked(action):
                    return self._teleop_idle_locked(action)
            try:
                joints, gripper = self._validate_action(action)
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
                self._arm.move_j(joints)
                self._raise_if_comm_error("move_j")
                command.update(ended_monotonic_s=self._monotonic(), result="sdk_returned")
            except Exception as exc:
                if command is not None:
                    command["ended_monotonic_s"] = self._monotonic()
                self._latch(PiperState.FAULT, exc)
            self._raise_if_emergency_stop_requested_locked()
            if self._teleop is not None and (
                self._input_fault_requested.is_set() or not self._teleop.permits(action.epoch)
            ):
                return self._teleop_idle_locked(action)
            if self._teleop is None or gripper != self._last_gripper_target:
                command = None
                try:
                    self._require_active_locked("gripper command")
                    assert self._gripper is not None
                    assert self._safety is not None
                    self._check_observation_age()
                    command = {
                        "name": "move_gripper_m",
                        "target": gripper,
                        "force": self._safety.gripper_force_n,
                        "started_monotonic_s": self._monotonic(),
                        "result": "failed",
                    }
                    self.last_action_telemetry["commands"].append(command)
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
            self._raise_if_emergency_stop_requested_locked()
            if self._teleop is not None and (
                self._input_fault_requested.is_set() or not self._teleop.permits(action.epoch)
            ):
                return self._teleop_idle_locked(action)
            self._last_action_at = self._monotonic()
            self._last_control_tick_s = self._last_action_at
            self._last_gripper_target = gripper
            self.last_action_telemetry.update(
                result="sdk_returned",
                values={key: float(action[key]) for key in ACTION_KEYS},
                retained_gripper_command=self._last_gripper_command,
            )
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
                self._receiver = None
                self._gripper = None
                if first_error is not None:
                    raise first_error
