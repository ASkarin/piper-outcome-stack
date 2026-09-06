from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "infra/acceptance"))
import piper_read_only_cycles as cycles  # noqa: E402


@dataclass
class Telemetry:
    timestamp_s: float = 100.0


class Thread:
    name = "fake-sdk-thread"
    alive = True

    def is_alive(self):
        return self.alive


class Robot:
    def __init__(self, *, fail_read=False, leak=False, firmware=None):
        self.threads = [Thread(), Thread(), Thread()]
        ctx = NS(
            _read_th=self.threads[0], _monitor_th=self.threads[1], fps=NS(thread=self.threads[2])
        )
        self._arm = NS(get_context=lambda: ctx)
        self._gripper = NS()
        self.config = NS(feedback_timeout_s=1.0)
        self.firmware_identity = firmware or {"software_version": "S-V1.9-0"}
        self.last_feedback_telemetry = Telemetry()
        self.is_connected = False
        self.state = NS(value="DISCONNECTED")
        self.fail_read, self.leak = fail_read, leak
        self.disconnect_count = 0

    def connect(self):
        self.is_connected = True
        self.state.value = "CONNECTED_DISABLED"

    def get_observation(self):
        if self.fail_read:
            raise RuntimeError("read failed")
        return {"gripper.pos": 0.0011}

    def disconnect(self):
        self.disconnect_count += 1
        self.is_connected = False
        self.state.value = "DISCONNECTED"
        if not self.leak:
            for thread in self.threads:
                thread.alive = False


def configure(monkeypatch):
    monkeypatch.setattr(cycles, "snapshot", lambda *_: {"host_wall_s": 100.0})
    monkeypatch.setattr(cycles, "validate_snapshot", lambda *_: None)
    monkeypatch.setattr(cycles.time, "sleep", lambda _: None)


def test_exactly_five_fresh_plugin_sessions_and_cleanup(monkeypatch):
    configure(monkeypatch)
    robots = []

    def factory():
        robot = Robot()
        robots.append(robot)
        return robot

    report = {}
    cycles.run_cycles(factory, {"software_version": "S-V1.9-0"}, report, sample_count=2)
    assert len(robots) == len(report["cycles"]) == 5
    assert report["status"] == "passed"
    assert all(robot.disconnect_count == 1 for robot in robots)
    assert all(cycle["sdk_threads_alive_after_disconnect"] == [] for cycle in report["cycles"])


@pytest.mark.parametrize("fault", ["read", "identity", "thread"])
def test_failure_stops_without_retry_and_disconnects(monkeypatch, fault):
    configure(monkeypatch)
    robots = []

    def factory():
        robot = Robot(
            fail_read=fault == "read",
            leak=fault == "thread",
            firmware={"software_version": "S-V1.8-8"} if fault == "identity" else None,
        )
        robots.append(robot)
        return robot

    report = {}
    with pytest.raises(RuntimeError):
        cycles.run_cycles(factory, {"software_version": "S-V1.9-0"}, report, sample_count=2)
    assert len(robots) == len(report["cycles"]) == 1
    assert robots[0].disconnect_count == 1
    assert report["cycles"][0]["status"] == "failed"
