"""Per-session Xbox intent and rearming; no SDK or device I/O."""

from __future__ import annotations

import math
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
    HOLD_REQUESTED = "HOLD_REQUESTED"
    PAUSED = "PAUSED"
    FAULT = "FAULT"
    E_STOP = "E_STOP"


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

    def observe(self, hold: bool, neutral: bool) -> tuple[str, int]:
        with self._lock:
            if self.state in (TeleopState.FAULT, TeleopState.E_STOP):
                return "wait", self.epoch
            rising = hold and not self._previous_hold
            self._previous_hold = hold
            if self.state is TeleopState.RUNNING and not hold:
                self.request_hold()
            if not hold:
                self._armed = neutral
            elif rising:
                if self._armed and neutral and self.hold_confirmed:
                    self.state = TeleopState.RUNNING
                    self._ever_running = True
                    self.epoch += 1
                self._armed = False
            intent = "run" if self.state is TeleopState.RUNNING else "hold"
            return intent, self.epoch

    def request_hold(self):
        with self._lock:
            self.epoch += 1
            self.hold_confirmed = False
            self._armed = False
            self.state = TeleopState.HOLD_REQUESTED

    def confirm_hold(self):
        with self._lock:
            if self.state in (TeleopState.FAULT, TeleopState.E_STOP):
                return
            self.hold_confirmed = True
            self.state = TeleopState.PAUSED if self._ever_running else TeleopState.WAITING

    def stop(self, emergency: bool):
        with self._lock:
            self.epoch += 1
            self._armed = False
            self.hold_confirmed = False
            self.state = TeleopState.E_STOP if emergency else TeleopState.FAULT

    def permits(self, epoch: int) -> bool:
        with self._lock:
            return self.state is TeleopState.RUNNING and epoch == self.epoch


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
