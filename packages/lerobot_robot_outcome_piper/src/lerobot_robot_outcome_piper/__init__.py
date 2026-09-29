"""Auto-discoverable PiPER plugin for LeRobot."""

from .config import OutcomePiperConfig, OutcomePiperXboxConfig
from .processor import OutcomePiperXboxProcessor, make_xbox_processor
from .robot import FeedbackTelemetry, OutcomePiper, PiperState, ServoFeedback, ServoState
from .teleoperator import OutcomePiperXbox
from .workflows import record, teleoperate

__all__ = [
    "OutcomePiper",
    "PiperState",
    "ServoFeedback",
    "ServoState",
    "OutcomePiperConfig",
    "OutcomePiperXbox",
    "OutcomePiperXboxConfig",
    "OutcomePiperXboxProcessor",
    "FeedbackTelemetry",
    "make_xbox_processor",
    "record",
    "teleoperate",
]
