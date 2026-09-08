from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "infra/acceptance"))
import piper_motion_preflight as preflight  # noqa: E402


@dataclass
class Telemetry:
    timestamp_s: float = 1.0


class Robot:
    def __init__(self, missing=None):
        self.is_connected = False
        self.calls = []
        self.firmware_identity = {"software_version": "S-V1.9-0"}
        self.last_feedback_telemetry = Telemetry()
        self._gripper = object()

        def query(index, *, timeout, min_interval):
            self.calls.append(index)
            assert timeout == 1.0 and min_interval == 0.0
            if index == missing:
                return None
            return NS(
                msg=NS(min_angle_limit=-1.0, max_angle_limit=1.0, max_joint_spd=0.5), timestamp=1.0
            )

        self._arm = NS(get_joint_angle_vel_limits=query, has_comm_error=lambda: False)

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        return {"joint_1.pos": 0.0}


def test_each_controller_limit_is_queried_once_without_motion(monkeypatch):
    monkeypatch.setattr(preflight, "snapshot", lambda *_: {"enabled": False})
    robot = Robot()
    report = {}
    preflight.inspect_limits(robot, report)
    assert robot.calls == [1, 2, 3, 4, 5, 6]
    assert report["status"] == "read_complete"
    assert len(report["joint_limits"]) == 6
    assert not report["connected_after_disconnect"]


def test_missing_limit_stops_queries_and_preserves_completed_reads(monkeypatch):
    monkeypatch.setattr(preflight, "snapshot", lambda *_: {"enabled": False})
    robot = Robot(missing=3)
    report = {}
    with pytest.raises(RuntimeError, match="joint 3.*no retry"):
        preflight.inspect_limits(robot, report)
    assert robot.calls == [1, 2, 3]
    assert len(report["joint_limits"]) == 2
    assert not robot.is_connected
