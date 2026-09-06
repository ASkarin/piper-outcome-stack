"""Host receive timing for the pinned official SDK; no CAN decoding or sending."""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CaptureTiming:
    camera_max_age_s: float
    joint_max_skew_s: float
    image_state_max_skew_s: float
    observation_max_age_s: float

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a measured positive value")


@dataclass(frozen=True)
class ReceivedFeedback:
    joints: object
    gripper: object
    status: object
    frames: tuple
    joint_hz: tuple
    received_s: tuple[float, float, float, float, float]


class FeedbackReceiver:
    # Official PiPER joint groups, status, and official gripper feedback.
    IDS = (0x2A5, 0x2A6, 0x2A7, 0x2A1, 0x2A8)

    def __init__(self, arm, gripper, clock: Callable[[], float]):
        self.arm, self.gripper, self.clock = arm, gripper, clock
        self.condition = threading.Condition(threading.RLock())
        self.received = {}
        self.error = None
        self.last_received = None
        comm = arm.get_context().get_comm()
        self.original = comm.get_callback()
        if self.original is None:
            raise RuntimeError("official SDK parser callback is unavailable")
        # One session-local callback; the original SDK still parses every packet.
        comm.set_callback(self.receive)

    def receive(self, packet):
        with self.condition:
            now = self.clock()
            try:
                if (
                    not math.isfinite(now)
                    or now < 0
                    or (self.last_received is not None and now < self.last_received)
                ):
                    raise RuntimeError("receive monotonic timestamp moved backwards or is invalid")
                self.last_received = now
                self.original(packet)
                if packet.arbitration_id in self.IDS:
                    if packet.is_extended_id or packet.is_error_frame or len(packet.data) != 8:
                        raise RuntimeError("malformed PiPER feedback frame")
                    self.received[packet.arbitration_id] = now
            except Exception as exc:
                self.error = exc
            finally:
                self.condition.notify_all()

    def wait_ready(self, timeout):
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    self.error is not None
                    or (len(self.received) == len(self.IDS) and self.arm.get_fps() > 0)
                ),
                timeout,
            )
            self.check_error()
            return len(self.received) == len(self.IDS) and self.arm.get_fps() > 0

    def check_error(self):
        if self.error is not None:
            raise RuntimeError(f"SDK receive processing failed: {self.error}") from self.error

    def status(self):
        with self.condition:
            self.check_error()
            return copy.deepcopy(self.arm.get_arm_status()), self.received.get(0x2A1)

    def snapshot(self):
        with self.condition:
            self.check_error()
            if len(self.received) != len(self.IDS):
                raise RuntimeError("incomplete received feedback groups")
            frames = tuple(
                getattr(self.arm._parser, name, None)
                for name in ("joint_12", "joint_34", "joint_56")
            )
            if any(frame is None for frame in frames):
                raise RuntimeError("incomplete joint feedback groups")
            # The same lock encloses SDK parsing, values and receipt timestamps.
            return ReceivedFeedback(
                copy.deepcopy(self.arm.get_joint_angles()),
                copy.deepcopy(self.gripper.get_gripper_status()),
                copy.deepcopy(self.arm.get_arm_status()),
                copy.deepcopy(frames),
                tuple(float(self.arm._ctx.fps.get_fps(frame.msg_type)) for frame in frames),
                tuple(self.received[key] for key in self.IDS),
            )
