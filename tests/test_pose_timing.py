import math
from dataclasses import asdict, replace
import numpy as np
import pytest
from test_pose_lookahead import sequence
from test_plugin import xbox_config, make_robot
from lerobot_robot_outcome_piper.joint_pose import PoseTiming, JointPoseSequence
from lerobot_robot_outcome_piper.errors import OutcomePiperStateError


def timing():
    return PoseTiming((0.05,) * 6, (0.1,) * 6, 0.005, 0.01, 0.05)


def test_quintic_duration_bounds_and_same_path(tmp_path):
    old, safety, start = sequence(tmp_path)
    limits = timing()
    goal = [0.0] * 6 + [0.03]
    new = JointPoseSequence(start, 0.0, safety, old.settings, goal, "work", timing=limits)
    points = np.array([[*start, 0.0], *new.targets])
    velocity = np.diff(points, axis=0) / limits.period_s
    acceleration = np.diff(velocity, axis=0) / limits.period_s
    assert np.all(np.max(np.abs(velocity), axis=0) <= np.array([*0.05 * np.ones(6), 0.005]) + 1e-12)
    assert np.all(
        np.max(np.abs(acceleration), axis=0) <= np.array([*0.1 * np.ones(6), 0.01]) + 1e-12
    )
    assert new.targets[-1] == goal
    assert new.nominal_duration_s >= limits.minimum_duration(start, 0.0, goal, True)
    u = (points[:, 0] - start[0]) / -start[0]
    np.testing.assert_allclose(
        points,
        np.array([*start, 0.0]) + (np.array(goal) - np.array([*start, 0.0])) * u[:, None],
        atol=1e-14,
    )
    assert new.telemetry()["reference_profile"] == "quintic_timed_feedback_gated"


def test_time_gate_prevents_fast_caller_and_does_not_catch_up(tmp_path):
    old, safety, start = sequence(tmp_path)
    seq = JointPoseSequence(start, 0.0, safety, old.settings, [0.0] * 7, "home", timing=timing())
    seq.sent(1.0)
    seq.observe(start, (1.01,) * 3, 1.01, 0.0)
    seq.next_waypoint()
    assert seq.index == 0
    seq.sent(1.01)
    assert seq.window.after_s == 1.0 and seq.window.deadline == 11.0
    seq.observe(start, (1.05,) * 3, 1.05, 0.0)
    seq.next_waypoint()
    assert seq.index == 1
    seq.sent(1.05)
    seq.observe(seq.targets[1][:6], (2.0,) * 3, 2.0, 0.0)
    seq.next_waypoint()
    assert seq.index == 2  # Long scheduling delay never skips references.


def test_timed_stall_still_times_out(tmp_path):
    old, safety, start = sequence(tmp_path)
    seq = JointPoseSequence(start, 0.0, safety, old.settings, [0.0] * 7, "home", timing=timing())
    seq.sent(1.0)
    with pytest.raises(OutcomePiperStateError, match="timed out"):
        seq.observe(start, (11.0,) * 3, 11.0, 0.0)
    assert seq.index == 0


def test_joint_only_timing_keeps_gripper_uncommanded(tmp_path):
    old, safety, start = sequence(tmp_path)
    seq = JointPoseSequence(
        start, -0.0003, safety, old.settings, [0.0] * 6, "work", timing=timing()
    )
    assert "gripper.pos" not in seq.values
    assert all(row[6] == -0.0003 for row in seq.targets)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_timing_rejects_invalid_limits(value):
    with pytest.raises(ValueError):
        replace(timing(), joint_velocity_rad_s=(value,) * 6)


def test_xbox_config_and_processor_preserve_timing(tmp_path):
    from lerobot_robot_outcome_piper.workflows import _processor

    robot, _, _ = make_robot(tmp_path, mode="motion")
    cfg = replace(xbox_config(), pose_timing=asdict(timing()))
    pipeline = _processor(robot.config, cfg)
    assert pipeline.steps[0].get_config()["pose_timing"] == asdict(timing())
    with pytest.raises(ValueError, match="control_hz"):
        replace(cfg, pose_timing=asdict(replace(timing(), period_s=0.1)))


def test_sdk_completion_offset_does_not_halve_reference_rate(tmp_path):
    old, safety, start = sequence(tmp_path)
    seq = JointPoseSequence(start, 0.0, safety, old.settings, [0.0] * 7, "home", timing=timing())
    seq.sent(1.001)
    indices = []
    for i in range(1, 61):
        now = 1.0 + i * 0.05
        seq.observe(seq.targets[seq.index][:6], (now - 0.0001,) * 3, now, 0.0)
        seq.next_waypoint()
        indices.append(seq.index)
        seq.sent(now + 0.001)  # SDK returns after the control tick's observation time.
    assert indices[-1] == 59
    assert all(b - a == 1 for a, b in zip(indices[1:], indices[2:]))


def test_long_delay_rebases_schedule_without_catchup_burst(tmp_path):
    old, safety, start = sequence(tmp_path)
    seq = JointPoseSequence(start, 0.0, safety, old.settings, [0.0] * 7, "home", timing=timing())
    seq.sent(1.001)
    seq.observe(start, (2.0,) * 3, 2.0, 0.0)
    seq.next_waypoint()
    seq.sent(2.001)
    assert seq.index == 1
    for now in (2.01, 2.02, 2.04):
        seq.observe(seq.targets[seq.index][:6], (now,) * 3, now, 0.0)
        seq.next_waypoint()
        seq.sent(now + 0.001)
        assert seq.index == 1
    assert seq.next_reference_due_s == pytest.approx(2.05)


def test_feedback_wait_recovery_has_no_extra_cycle(tmp_path):
    old, safety, start = sequence(tmp_path)
    seq = JointPoseSequence(start, 0.0, safety, old.settings, [0.0] * 7, "home", timing=timing())
    seq.sent(1.001)
    current = seq.targets[0][:6]
    following = seq.targets[1][:6]
    delta = following[0] - current[0]
    lagged = list(current)
    lagged[0] = following[0] - math.copysign(
        safety.max_joint_step[0] - seq.settings.joint_tolerance_rad + abs(delta) / 2, delta
    )
    for now in (1.1, 1.15, 1.2):
        seq.observe(lagged, (now,) * 3, now, 0.0)
        seq.next_waypoint()
        seq.sent(now + 0.001)
        assert seq.index == 0 and seq.window.deadline == 11.001
    seq.observe(current, (1.25,) * 3, 1.25, 0.0)
    seq.next_waypoint()
    seq.sent(1.251)
    assert seq.index == 1 and seq.next_reference_due_s == pytest.approx(1.3)
    seq.observe(seq.targets[1][:6], (1.3,) * 3, 1.3, 0.0)
    seq.next_waypoint()
    assert seq.index == 2


def test_slow_sdk_completion_does_not_allow_immediate_catchup(tmp_path):
    old, safety, start = sequence(tmp_path)
    seq = JointPoseSequence(start, 0.0, safety, old.settings, [0.0] * 7, "home", timing=timing())
    seq.sent(1.001)
    seq.observe(start, (2.0,) * 3, 2.0, 0.0)
    seq.next_waypoint()
    seq.sent(2.1)
    assert seq.next_reference_due_s == pytest.approx(2.15)
    seq.observe(seq.targets[1][:6], (2.11,) * 3, 2.11, 0.0)
    seq.next_waypoint()
    assert seq.index == 1
