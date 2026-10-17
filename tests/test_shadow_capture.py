from types import SimpleNamespace as NS
import numpy as np
import pytest

from piper_outcome_stack.shadow_capture import ACTION_KEYS, acquire_pairs


class FakeClock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, dt):
        self.now += dt


class ReadOnlyRobot:
    def __init__(self, clock):
        self.clock = clock
        self.sequence = 0
        self.config = NS(execution_mode="read_only", capture_timing=NS(observation_max_age_s=0.1))

    def get_observation(self):
        self.sequence += 1
        self.last_observation_telemetry = {
            "sequence": self.sequence,
            "quality": "checked",
            "oldest_received_monotonic_s": self.clock(),
        }
        return {
            **{k: 0.01 for k in ACTION_KEYS},
            "d435": np.full((480, 640, 3), self.sequence, np.uint8),
        }

    def send_action(self, *args):
        raise AssertionError("shadow must never dispatch")

    def enable(self, *args):
        raise AssertionError("shadow must never enable")


def test_capture_real_observation_pairs_without_dispatch(tmp_path):
    clock = FakeClock()
    robot = ReadOnlyRobot(clock)
    result = acquire_pairs(robot, tmp_path, pairs=2, stride=50, clock=clock, sleep=clock.sleep)
    assert result["motion_commands_sent"] == 0 and not result["motor_enable_executed"]
    assert result["observations"] == 76
    assert result["actual_spacing_s"] == pytest.approx([0.5, 0.5])
    first = np.load(tmp_path / "pairs/000.npz")
    assert first["first_image"][0, 0, 0] == 1 and first["second_image"][0, 0, 0] == 26
    assert "targets" not in first.files


def test_motion_configuration_rejected_before_observation(tmp_path):
    clock = FakeClock()
    robot = ReadOnlyRobot(clock)
    robot.config.execution_mode = "motion"
    with pytest.raises(ValueError, match="motion-configured"):
        acquire_pairs(robot, tmp_path)
    assert robot.sequence == 0


def test_stale_or_duplicate_observation_fails(tmp_path):
    clock = FakeClock()
    robot = ReadOnlyRobot(clock)
    original = robot.get_observation

    def stale():
        obs = original()
        robot.last_observation_telemetry["oldest_received_monotonic_s"] -= 0.2
        return obs

    robot.get_observation = stale
    with pytest.raises(ValueError, match="stale"):
        acquire_pairs(robot, tmp_path, pairs=1, clock=clock, sleep=clock.sleep)
    assert not (tmp_path / "pairs.json").exists()


def test_duplicate_sequence_fails(tmp_path):
    clock = FakeClock()
    robot = ReadOnlyRobot(clock)
    original = robot.get_observation

    def repeated():
        obs = original()
        robot.last_observation_telemetry["sequence"] = 1
        return obs

    robot.get_observation = repeated
    with pytest.raises(ValueError, match="sequence"):
        acquire_pairs(robot, tmp_path, pairs=1, clock=clock, sleep=clock.sleep)
