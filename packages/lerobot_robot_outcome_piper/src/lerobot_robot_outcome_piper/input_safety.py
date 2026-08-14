"""Scope Xbox input faults to one active PiPER motion session."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Protocol


class EmergencyStopTarget(Protocol):
    def request_emergency_stop(self, cause: BaseException | str) -> None: ...


_motion_session: ContextVar[EmergencyStopTarget | None] = ContextVar(
    "outcome_piper_motion_session", default=None
)
_scope_enabled: ContextVar[bool] = ContextVar(
    "outcome_piper_input_safety_scope_enabled", default=False
)


@contextmanager
def motion_input_safety_scope() -> Iterator[None]:
    """Enable binding only for one PiPER-owned official LeRobot workflow."""

    session_token = _motion_session.set(None)
    enabled_token = _scope_enabled.set(True)
    try:
        yield
    finally:
        _scope_enabled.reset(enabled_token)
        _motion_session.reset(session_token)


def register_active_motion_session(robot: EmergencyStopTarget) -> None:
    """Register the robot after and only after it reaches ``ACTIVE``."""

    if not _scope_enabled.get():
        return
    current = _motion_session.get()
    if current is not None and current is not robot:
        raise RuntimeError("an input-safety scope already owns another PiPER motion session")
    _motion_session.set(robot)


def request_input_emergency_stop(cause: BaseException | str) -> None:
    """Request the verified stop action when a bound Xbox input fails."""

    robot = _motion_session.get()
    if robot is not None:
        robot.request_emergency_stop(cause)
