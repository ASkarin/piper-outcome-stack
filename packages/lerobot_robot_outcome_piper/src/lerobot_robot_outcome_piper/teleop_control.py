"""Per-session Xbox intent and rearming; no SDK or device I/O."""

from __future__ import annotations

import math
import logging
import threading
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class HoldSettings:
    joint_tolerance_rad: float
    stable_time_s: float
    timeout_s: float

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("hold settings must be explicit positive finite values")
        if self.stable_time_s >= self.timeout_s:
            raise ValueError("hold stable time must be less than its timeout")


class TeleopState(str, Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    POSE_READY = "POSE_READY"
    POSE_MOVING = "POSE_MOVING"
    CENTERING = "CENTERING"
    CENTERED = "CENTERED"
    HOLD_REQUESTED = "HOLD_REQUESTED"
    PAUSED = "PAUSED"
    FAULT = "FAULT"
    E_STOP = "E_STOP"


class TeleopMode(str, Enum):
    TRANSLATION = "TRANSLATION"
    ORIENTATION = "ORIENTATION"


class TranslationStrategy(str, Enum):
    WRIST_PRIORITY = "WRIST_PRIORITY"
    FIXED_ORIENTATION = "FIXED_ORIENTATION"


class TeleopControl:
    def __init__(self):
        self._lock = threading.RLock()
        self.state = TeleopState.WAITING
        self.epoch = 0
        self.hold_confirmed = False
        self._armed = False
        self._previous_hold = False
        self._ever_running = False
        self.gripper_target: float | None = None
        self.gripper_reference_state = None
        self.gripper_reference_revision = 0
        self.joint_target = None
        self.mode = TeleopMode.TRANSLATION
        self.pending_mode = None
        self._pending_mode_phase = None
        self.translation_strategy = TranslationStrategy.WRIST_PRIORITY
        self._previous_translation = None
        self._previous_mode = None
        self.mode_event = None
        self.reference_state = None
        self.reference_revision = 0
        self.orientation_target = None
        self._arm_input_active = False
        self._previous_home = None
        self._previous_work = None
        self.pose_kind = None
        self.recording_phase = None
        self.pose_sequence = None
        self._pose_execution_requested = False
        self.pose_event = None
        self.pose_cancel_reason = None
        self.hold_settings = None

    def prepare_enable(self):
        """Invalidate prior input after an explicit servo lifecycle operation."""
        with self._lock:
            if self.state in (TeleopState.FAULT, TeleopState.E_STOP):
                raise RuntimeError("cannot enable a terminal teleoperation session")
            self.epoch += 1
            self.state = TeleopState.WAITING
            self.hold_confirmed = False
            self._armed = False
            self._previous_hold = False
            self._ever_running = False
            self.joint_target = None
            self._previous_home = None
            self._previous_work = None
            self.pose_kind = None
            self.pose_sequence = None
            self._pose_execution_requested = False
            self.pose_event = None
            self.pose_cancel_reason = None
            self.mode = TeleopMode.TRANSLATION
            self.pending_mode = None
            self._pending_mode_phase = None
            self.translation_strategy = TranslationStrategy.WRIST_PRIORITY
            self._previous_translation = None
            self._previous_mode = None
            self.mode_event = None
            self.orientation_target = None
            self._arm_input_active = False

    def observe(
        self,
        hold: bool,
        neutral: bool,
        mode_switch: bool = False,
        home: bool = False,
        work: bool = False,
        translation_switch: bool = False,
    ) -> tuple[str, int]:
        with self._lock:
            self.mode_event = None
            if self.state in (TeleopState.FAULT, TeleopState.E_STOP):
                return "wait", self.epoch
            if self.pending_mode is not None and (
                hold
                or not neutral
                or home
                or work
                or translation_switch
                or self.recording_phase != self._pending_mode_phase
            ):
                reason = (
                    "LB pressed before mode ready"
                    if hold
                    else "inputs not neutral"
                    if not neutral
                    else "conflicting button"
                    if home or work or translation_switch
                    else "recording phase changed"
                )
                self._cancel_pending_mode(reason)
                self._previous_hold = hold
                self._previous_mode = mode_switch
                self._previous_home, self._previous_work = home, work
                self._previous_translation = translation_switch
                self._armed = False
                return "hold", self.epoch
            translation_edge = self._previous_translation is False and translation_switch
            translation_startup = self._previous_translation is None and translation_switch
            self._previous_translation = translation_switch
            if translation_switch and (mode_switch or home or work):
                self._previous_mode, self._previous_home, self._previous_work = (
                    mode_switch,
                    home,
                    work,
                )
                self._armed = False
                self._previous_hold = hold
                self.mode_event = {"accepted": False, "reason": "conflicting buttons"}
                if self.state in (TeleopState.POSE_READY, TeleopState.POSE_MOVING):
                    self.request_hold()
                logging.info("[未切换] X不能与RB、A或Y同时按下。")
                return "hold", self.epoch
            if translation_edge or translation_startup:
                accepted = (
                    not translation_startup
                    and not hold
                    and neutral
                    and self.hold_confirmed
                    and self.mode is TeleopMode.TRANSLATION
                    and self.recording_phase in (None, "preparing", "recording")
                    and self.state in (TeleopState.WAITING, TeleopState.PAUSED)
                )
                if accepted:
                    self.translation_strategy = (
                        TranslationStrategy.FIXED_ORIENTATION
                        if self.translation_strategy is TranslationStrategy.WRIST_PRIORITY
                        else TranslationStrategy.WRIST_PRIORITY
                    )
                    self.epoch += 1
                    self.reset_reference()
                self._armed = False
                self._previous_hold = hold
                self.mode_event = {
                    "accepted": accepted,
                    "translation_strategy": self.translation_strategy.value,
                    "reason": "switched"
                    if accepted
                    else "requires neutral confirmed hold in translation mode",
                }
                logging.info(
                    "[平移策略] %s",
                    (
                        "腕部优先保持"
                        if self.translation_strategy is TranslationStrategy.WRIST_PRIORITY
                        else "保持朝向"
                    )
                    if accepted
                    else "未切换：请在平移模式松开LB、回中并等待保持确认，再按X。",
                )
                return "hold", self.epoch
            home_edge = self._previous_home is False and home
            work_edge = self._previous_work is False and work
            startup = (self._previous_home is None and home) or (
                self._previous_work is None and work
            )
            self._previous_home, self._previous_work = home, work
            if home_edge or work_edge or startup:
                accepted = (
                    not startup
                    and home != work
                    and not hold
                    and neutral
                    and not mode_switch
                    and not translation_switch
                    and self.recording_phase in (None, "preparing")
                    and self.hold_confirmed
                    and self.state in (TeleopState.WAITING, TeleopState.PAUSED)
                )
                if not accepted and self.state is TeleopState.POSE_READY:
                    self.request_hold()  # A rejected replacement must not leave an old target armed.
                self.pose_event = "ready" if accepted else "request_rejected"
                if accepted:
                    logging.info(
                        "[姿态已选择] %s；等待保持确认后，松开目标键，再按住LB执行。",
                        "回零" if home else "进入工作姿态",
                    )
                elif self.recording_phase not in (None, "preparing"):
                    logging.info("[未执行] 当前阶段不接受姿态操作；请先结束回合并返回准备阶段。")
                elif home and work:
                    logging.info("[未执行] 请松开A和Y，再只按一个目标键。")
                else:
                    logging.info("[未执行] 请松开LB并回中，等待保持确认后，再按目标键。")
                if accepted:
                    self.pose_kind = "home" if home else "work"
                    self.pose_cancel_reason = None
                    self.state = TeleopState.POSE_READY
                    self.hold_confirmed = False
                    self.pose_sequence = None
                    self._pose_execution_requested = False
                    self.epoch += 1
            if self.recording_phase in ("review", "saving", "finalizing"):
                self._previous_hold, self._previous_mode = hold, mode_switch
                self._armed = False
                return "hold", self.epoch
            if self.state is TeleopState.POSE_READY:
                rising = hold and not self._previous_hold
                self._previous_hold = hold
                self._previous_mode = mode_switch
                self._armed = False
                if not neutral or mode_switch or (self._pose_execution_requested and not hold):
                    self.request_hold(
                        "input_not_neutral"
                        if not neutral
                        else "mode_switch"
                        if mode_switch
                        else "lb_released"
                    )
                else:
                    if rising and not home and not work and self.hold_confirmed:
                        self._pose_execution_requested = True
                        if self.pose_sequence is None or not self.pose_sequence.planning_complete:
                            logging.info("[姿态规划中] 持续按住LB；检查完成后开始，松LB取消。")
                    if (
                        self._pose_execution_requested
                        and hold
                        and not home
                        and not work
                        and self.hold_confirmed
                        and self.pose_sequence is not None
                        and self.pose_sequence.planning_complete
                        and self.pose_sequence.control_epoch == self.epoch
                    ):
                        self.state = TeleopState.POSE_MOVING
                        self._ever_running = True
                        self.epoch += 1
                        self.pose_sequence.control_epoch = self.epoch
                        self._pose_execution_requested = False
                        self.pose_event = "started"
                        logging.info(
                            "[姿态移动] %s：按住LB，松开取消；B急停",
                            "回零" if self.pose_kind == "home" else "进入工作姿态",
                        )
                        return "pose", self.epoch
                return "hold", self.epoch
            if self.state is TeleopState.POSE_MOVING:
                self._previous_hold = hold
                self._previous_mode = mode_switch
                if not hold or not neutral:
                    self.request_hold("lb_released" if not hold else "input_not_neutral")
                    return "hold", self.epoch
                return "pose", self.epoch
            # Process shoulder release before RB, so the same neutral input
            # sample can request ordinary hold and select the upcoming mode.
            if (
                self.state in (TeleopState.RUNNING, TeleopState.CENTERING, TeleopState.CENTERED)
                and not hold
            ):
                self.request_hold()
            mode_edge = self._previous_mode is False and mode_switch
            startup_pressed = self._previous_mode is None and mode_switch
            self._previous_mode = mode_switch
            if self.pending_mode is not None:
                if mode_edge:
                    self._cancel_pending_mode("RB pressed again")
                    self._previous_hold = hold
                    self._armed = False
                    return "hold", self.epoch
                if self.hold_confirmed:
                    self._apply_mode(self.pending_mode)
                # No movement this tick; readiness still needs a fresh LB edge.
                self._previous_hold = hold
                self._armed = self.hold_confirmed and neutral and not hold
                return "hold", self.epoch
            if mode_edge or startup_pressed:
                reason = None
                if startup_pressed:
                    reason = "release startup-held RB"
                elif hold:
                    reason = "release LB"
                elif home or work or translation_switch:
                    reason = "release conflicting buttons"
                elif not neutral:
                    reason = "center sticks and release triggers"
                elif self.recording_phase not in (None, "preparing", "recording"):
                    reason = "mode selection unavailable in this phase"
                elif self.state not in (
                    TeleopState.WAITING,
                    TeleopState.PAUSED,
                    TeleopState.HOLD_REQUESTED,
                ):
                    reason = "mode selection unavailable in this state"
                if reason is None:
                    target = (
                        TeleopMode.ORIENTATION
                        if self.mode is TeleopMode.TRANSLATION
                        else TeleopMode.TRANSLATION
                    )
                    if self.hold_confirmed:
                        self._apply_mode(target)
                    else:
                        self.pending_mode = target
                        self._pending_mode_phase = self.recording_phase
                        self.reset_reference()
                        self._armed = False
                        self.mode_event = dict(
                            accepted=True,
                            ready=False,
                            mode=self.mode.value,
                            pending_mode=target.value,
                            reason="awaiting hold",
                        )
                        logging.info(
                            "[模式已选择] %s模式；正在确认保持，请保持LB松开、输入回中。",
                            "姿态" if target is TeleopMode.ORIENTATION else "平移",
                        )
                else:
                    self._armed = False  # Do not arm an old-mode move on a simultaneous RB/LB edge.
                    self.mode_event = dict(
                        accepted=False, ready=False, mode=self.mode.value, reason=reason
                    )
                    messages = {
                        "release startup-held RB": "请先释放启动时按住的RB",
                        "release LB": "请先松开LB",
                        "release conflicting buttons": "请释放冲突按键",
                        "center sticks and release triggers": "请让摇杆回中并释放扳机",
                    }
                    logging.info(
                        "[未切换] %s；需重新单击RB。", messages.get(reason, "当前阶段不能选择模式")
                    )
            rising = hold and not self._previous_hold
            self._previous_hold = hold
            if not hold:
                self._armed = neutral
            elif rising:
                if self._armed and neutral and self.hold_confirmed:
                    self.state = TeleopState.RUNNING
                    self._ever_running = True
                    self._arm_input_active = False
                    self.epoch += 1
                    logging.info("[LB已按住] 推动摇杆操作；摇杆回中保持，扳机控制夹爪。")
                self._armed = False
            intent = (
                "run"
                if self.state is TeleopState.RUNNING
                else (
                    "center"
                    if self.state in (TeleopState.CENTERING, TeleopState.CENTERED)
                    else "hold"
                )
            )
            return intent, self.epoch

    def _apply_mode(self, target):
        self.mode = target
        self.pending_mode = None
        self._pending_mode_phase = None
        self._armed = False
        self.epoch += 1
        self.reset_reference()
        self.mode_event = dict(accepted=True, ready=True, mode=target.value, reason="switched")
        logging.info(
            "[模式已就绪] %s模式；重新按住LB操作。",
            "姿态" if target is TeleopMode.ORIENTATION else "平移",
        )

    def _cancel_pending_mode(self, reason):
        if self.pending_mode is None:
            return
        target = self.pending_mode
        self.pending_mode = None
        self._pending_mode_phase = None
        self._armed = False
        self.mode_event = dict(
            accepted=False,
            ready=False,
            mode=self.mode.value,
            cancelled_mode=target.value,
            reason=reason,
        )
        messages = {
            "LB pressed before mode ready": "模式就绪前按下了LB",
            "inputs not neutral": "摇杆或扳机偏转",
            "conflicting button": "按下了其他模式/姿态键",
            "RB pressed again": "再次按下RB",
            "recording phase changed": "录制阶段已改变",
            "processor_reset": "控制参考已重置",
        }
        logging.info(
            "[模式选择已取消] %s；仍为%s模式，松LB回中后重新单击RB。",
            messages.get(reason, "收到保持或故障请求"),
            "姿态" if self.mode is TeleopMode.ORIENTATION else "平移",
        )

    def reset_reference(self):
        with self._lock:
            self.reference_state = None
            self.reference_revision += 1
            self.gripper_reference_state = None
            self.gripper_reference_revision += 1

    def gripper_reference_snapshot(self, epoch):
        with self._lock:
            state = self.gripper_reference_state
            return (
                dict(state) if state is not None and state["epoch"] == epoch else None,
                self.gripper_reference_revision,
            )

    def gripper_reference_valid(self, epoch, plan):
        with self._lock:
            return (
                epoch == self.epoch
                and self.state in (TeleopState.RUNNING, TeleopState.CENTERING, TeleopState.CENTERED)
                and plan["base_revision"] == self.gripper_reference_revision
            )

    def commit_gripper_reference(self, epoch, plan):
        with self._lock:
            if (
                plan is None
                or "reference" not in plan
                or not self.gripper_reference_valid(epoch, plan)
            ):
                return False
            self.gripper_reference_state = dict(plan["reference"], epoch=epoch)
            self.gripper_reference_revision += 1
            return True

    def reference_snapshot(self, epoch):
        from copy import deepcopy

        with self._lock:
            state = self.reference_state
            return (
                deepcopy(state) if state is not None and state["epoch"] == epoch else None,
                self.reference_revision,
            )

    def reference_valid(self, epoch, plan):
        with self._lock:
            return self.permits(epoch) and plan["base_revision"] == self.reference_revision

    def commit_reference(self, epoch, plan):
        from copy import deepcopy

        with self._lock:
            if plan is None or not self.reference_valid(epoch, plan):
                return False
            self.reference_state = deepcopy(plan["reference"])
            self.reference_state["epoch"] = epoch
            self.reference_revision += 1
            return True

    def initialize_orientation(self, matrix):
        with self._lock:
            if self.orientation_target is None:
                self.orientation_target = tuple(tuple(float(v) for v in row) for row in matrix)

    def commit_orientation(self, epoch, matrix):
        with self._lock:
            if matrix is not None and self.permits(epoch):
                self.orientation_target = tuple(tuple(float(v) for v in row) for row in matrix)
                return True
            return False

    def arm_input_intent(self, epoch, moving, detail=None):
        """Centering cancels arm intent without releasing the operator's LB permission."""
        with self._lock:
            if epoch != self.epoch or self.state not in (
                TeleopState.RUNNING,
                TeleopState.CENTERING,
                TeleopState.CENTERED,
            ):
                return "hold", self.epoch
            if self.state is TeleopState.CENTERING:
                return "center", self.epoch
            if moving:
                if self.state is TeleopState.CENTERED:
                    if not self.hold_confirmed:
                        return "center", self.epoch
                    self.epoch += 1
                    self.state = TeleopState.RUNNING
                if not self._arm_input_active:
                    logging.info("Xbox 机械臂新输入 epoch=%s: %s", self.epoch, detail)
                self._arm_input_active = True
                return "run", self.epoch
            if self._arm_input_active:
                self.epoch += 1
                self.hold_confirmed = False
                self.state = TeleopState.CENTERING
                logging.info("[摇杆回中] 正在保持位置…")
            else:
                self.state = TeleopState.CENTERED
            self._arm_input_active = False
            return "center", self.epoch

    def reconfirm_hold(self):
        with self._lock:
            if self.state is TeleopState.CENTERED:
                self.epoch += 1
                self.hold_confirmed = False
                self.state = TeleopState.CENTERING
            else:
                self.request_hold()

    def request_hold(self, reason=None):
        with self._lock:
            self._cancel_pending_mode(reason or "hold requested")
            if self.state in (TeleopState.FAULT, TeleopState.E_STOP):
                return
            if self.state in (TeleopState.POSE_READY, TeleopState.POSE_MOVING):
                if self.pose_event != "completed":
                    self.pose_event = "cancelled"
                    self.pose_cancel_reason = reason or "hold_requested"
                    label = {
                        "lb_released": "LB已松开",
                        "input_not_neutral": "摇杆或扳机偏转",
                        "mode_switch": "请求切换模式",
                        "hold_requested": "收到保持请求",
                    }.get(self.pose_cancel_reason, self.pose_cancel_reason)
                    logging.info("[姿态移动已取消] %s；正在保持当前位置。", label)
                self.pose_sequence = None
            self.epoch += 1
            self._pose_execution_requested = False
            self.hold_confirmed = False
            self._armed = False
            self.state = TeleopState.HOLD_REQUESTED

    def confirm_hold(self, orientation=None):
        with self._lock:
            if self.state in (TeleopState.FAULT, TeleopState.E_STOP):
                return
            if orientation is not None:
                self.orientation_target = tuple(tuple(float(v) for v in row) for row in orientation)
            self.hold_confirmed = True
            if self.state is TeleopState.POSE_READY:
                return
            self.state = (
                TeleopState.CENTERED
                if self.state is TeleopState.CENTERING
                else (TeleopState.PAUSED if self._ever_running else TeleopState.WAITING)
            )

    def stop(self, emergency: bool):
        with self._lock:
            self.pending_mode = None
            self._pending_mode_phase = None
            if self.state is TeleopState.E_STOP and not emergency:
                return
            self.epoch += 1
            self._armed = False
            self.hold_confirmed = False
            self.pose_sequence = None
            self._pose_execution_requested = False
            self.state = TeleopState.E_STOP if emergency else TeleopState.FAULT

    def permits(self, epoch: int) -> bool:
        with self._lock:
            return (
                self.state in (TeleopState.RUNNING, TeleopState.POSE_MOVING) and epoch == self.epoch
            )


class JointHold:
    """Fixed target and receive-time stability window, shared with commissioning.

    Callers validate feedback and own SDK dispatch. This class never sends commands
    or grants hardware acceptance.
    """

    def __init__(self, target, settings: HoldSettings, requested_s: float):
        self.target = list(target)
        self.settings = settings
        self.restart(requested_s)

    def restart(self, now):
        self.confirmed = False
        self.after_s = now
        self.deadline = now + self.settings.timeout_s
        self.stable_since = None
        self.last_received = None

    def within(self, joints):
        return all(
            abs(a - b) <= self.settings.joint_tolerance_rad
            for a, b in zip(joints, self.target, strict=True)
        )

    def observe(self, joints, received, now):
        from .errors import OutcomePiperStateError

        within = self.within(joints)
        if self.confirmed:
            if within:
                return True
            self.restart(now)
        if now >= self.deadline:
            raise OutcomePiperStateError("hold confirmation timed out")
        if min(received) < self.after_s:
            return False
        if self.last_received is not None and any(
            a <= b for a, b in zip(received, self.last_received, strict=True)
        ):
            return False
        self.last_received = tuple(received)
        if within:
            if self.stable_since is None:
                self.stable_since = min(received)
            self.confirmed = min(received) - self.stable_since >= self.settings.stable_time_s
        else:
            self.stable_since = None
        return self.confirmed
