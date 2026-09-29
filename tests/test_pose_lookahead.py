"""Synthetic tracking tests; no hardware velocity or smoothness claim."""

from dataclasses import replace
import math
import pytest
from test_plugin import make_robot, xbox_config
from lerobot_robot_outcome_piper.joint_pose import JointPoseSequence
from lerobot_robot_outcome_piper.safety import load_motion_safety, step_within_limit
from lerobot_robot_outcome_piper.errors import OutcomePiperStateError


def sequence(tmp_path, start=None):
    robot, _, _ = make_robot(tmp_path, mode="motion")
    safety = load_motion_safety(robot.config.safety_path)
    safety = replace(safety, max_joint_step=(math.radians(5),) * 6, max_gripper_step=0.005)
    settings = replace(
        xbox_config().hold_settings(),
        stable_time_s=0.3,
        timeout_s=10,
        joint_tolerance_rad=math.radians(0.1),
    )
    start = [0.6, 0.5, -0.4, 0.3, -0.2, 0.1] if start is None else start
    return JointPoseSequence(start, 0.0, safety, settings, [0.0] * 7, "home"), safety, start


def test_advance_before_old_reference_arrival_without_stable_wait(tmp_path):
    seq, safety, start = sequence(tmp_path)
    seq.sent(0.0)
    old_target = seq.targets[0][:6]
    seq.observe(start, (0.05,) * 3, 0.05, 0.0)
    assert old_target != start
    assert seq.advance_ready and not seq.confirmed
    assert seq.window.stable_since is None
    seq.next_waypoint()
    assert seq.index == 1
    assert seq.telemetry()["tracking_error_rad"] is None
    assert all(
        step_within_limit(t, q, limit)
        for t, q, limit in zip(seq.targets[1][:6], start, safety.max_joint_step)
    )
    seq.sent(0.05)
    assert seq.window.deadline == 10.05  # One deadline for this newly admitted reference.
    seq.sent(1.0)
    assert seq.window.deadline == 10.05


def test_stationary_or_duplicate_feedback_cannot_run_reference_ahead(tmp_path):
    seq, safety, start = sequence(tmp_path)
    seq.sent(0.0)
    seq.observe(start, (0.05,) * 3, 0.05, 0.0)
    seq.next_waypoint()
    seq.sent(0.05)
    for i in range(2, 150):
        previous_index = seq.index
        seq.observe(start, (0.05 * i,) * 3, 0.05 * i, 0.0)
        seq.next_waypoint()
        seq.sent(0.05 * i)
        if seq.index == previous_index:
            break
    else:
        pytest.fail("stationary feedback did not bound lookahead")
    assert any(
        not step_within_limit(t, q, limit - seq.settings.joint_tolerance_rad)
        for t, q, limit in zip(seq.targets[seq.index + 1][:6], start, safety.max_joint_step)
    )
    deadline = seq.window.deadline
    # A different value with a duplicate receive stamp must not renew the timer.
    seq.observe(seq.targets[seq.index][:6], (0.05 * i,) * 3, 0.05 * i + 0.01, 0.0)
    assert not seq.advance_ready and seq.window.deadline == deadline
    with pytest.raises(OutcomePiperStateError, match="tracking timed out"):
        seq.observe(start, (deadline,) * 3, deadline, 0.0)


def test_tracking_error_is_checked_before_advancement(tmp_path):
    seq, _, start = sequence(tmp_path)
    seq.sent(0.0)
    wrong = list(start)
    wrong[0] += math.radians(6)
    with pytest.raises(OutcomePiperStateError, match="tracking error"):
        seq.observe(wrong, (0.05,) * 3, 0.05, 0.0)
    assert seq.index == 0 and not seq.advance_ready


def test_final_target_requires_original_stable_window_and_deadline(tmp_path):
    seq, _, _ = sequence(tmp_path, [0.0] * 6)
    seq.sent(1.0)
    for t in [1.01, 1.1, 1.2, 1.30]:
        seq.observe([0.0] * 6, (t,) * 3, t, 0.0)
        seq.sent(t)
        assert not seq.complete and seq.window.deadline == 11.0
    seq.observe([0.0] * 6, (1.32,) * 3, 1.32, 0.0)
    assert seq.complete and seq.telemetry()["joint_confirmed"]


def test_synthetic_following_stays_bounded_and_confirms_only_endpoint(tmp_path):
    seq, safety, q = sequence(tmp_path)
    seq.sent(0.0)
    early_advances = 0
    for i in range(1, 1000):
        now = i * 0.05
        reference = list(seq.targets[seq.index][:6])
        # Synthetic follower, not a PiPER dynamic model.
        q = [a + 0.2 * (b - a) for a, b in zip(q, reference)]
        seq.observe(q, (now,) * 3, now, 0.0)
        if seq.complete:
            break
        if seq.advance_ready:
            early_advances += (
                max(abs(a - b) for a, b in zip(q, reference)) > seq.settings.joint_tolerance_rad
            )
            assert not seq.confirmed
        seq.next_waypoint()
        assert all(
            step_within_limit(t, a, bound)
            for t, a, bound in zip(seq.targets[seq.index][:6], q, safety.max_joint_step)
        )
        seq.sent(now)
    else:
        pytest.fail("synthetic follower did not complete")
    assert early_advances > 1
    assert seq.targets[-1] == [0.0] * 7


def test_only_admitting_next_reference_starts_new_deadline(tmp_path):
    seq, _, start = sequence(tmp_path)
    seq.sent(0.0)
    progressed = [q * 0.9 for q in start]
    seq.observe(progressed, (8.0,) * 3, 8.0, 0.0)
    assert seq.window.deadline == 10.0
    seq.next_waypoint()
    seq.sent(8.0)
    assert seq.window.deadline == 18.0
    for t, q in [(9.0, [v * 0.95 for v in start]), (9.5, progressed)]:
        seq.observe(q, (t,) * 3, t, 0.0)
        assert seq.window.deadline == 18.0


def test_gripper_lag_also_bounds_lookahead(tmp_path):
    initial, safety, _ = sequence(tmp_path, [0.0] * 6)
    seq = JointPoseSequence([0.0] * 6, 0.03, safety, initial.settings, [0.0] * 7, "home")
    seq.sent(0.0)
    seq.observe([0.0] * 6, (0.05,) * 3, 0.05, 0.03)
    assert seq.advance_ready
    seq.next_waypoint()
    seq.sent(0.05)
    assert step_within_limit(seq.targets[seq.index][6], 0.03, safety.max_gripper_step)
    for i in range(2, 150):
        previous_index = seq.index
        seq.observe([0.0] * 6, (0.05 * i,) * 3, 0.05 * i, 0.03)
        seq.next_waypoint()
        seq.sent(0.05 * i)
        if seq.index == previous_index:
            break
    else:
        pytest.fail("gripper feedback did not bound lookahead")
    assert not seq.advance_ready
    assert step_within_limit(seq.targets[seq.index][6], 0.03, safety.max_gripper_step)
    assert not step_within_limit(seq.targets[seq.index + 1][6], 0.03, safety.max_gripper_step)


def test_dense_references_ramp_at_ends_and_keep_exact_goal(tmp_path):
    seq, safety, start = sequence(tmp_path)
    points = [[*start, 0.0], *seq.targets]
    deltas = [[abs(b - a) for a, b in zip(left, right)] for left, right in zip(points, points[1:])]
    for joint, bound in enumerate(safety.max_joint_step):
        assert (
            max(d[joint] for d in deltas)
            <= (bound - 2 * seq.settings.joint_tolerance_rad) / 5 + 1e-12
        )
        if start[joint] >= 0:
            assert all(right[joint] <= left[joint] for left, right in zip(points, points[1:]))
        else:
            assert all(right[joint] >= left[joint] for left, right in zip(points, points[1:]))
    middle = max(d[0] for d in deltas)
    assert deltas[0][0] < middle / 10
    assert deltas[-1][0] < middle / 10
    assert seq.targets[-1] == [0.0] * 7
    assert seq.telemetry()["reference_profile"] == "quintic_dense_feedback_gated"


def test_new_reference_not_committed_until_sdk_sent(tmp_path):
    seq, _, start = sequence(tmp_path)
    seq.observe(start, (0.05,) * 3, 0.05, 0.0)
    seq.next_waypoint()
    assert seq.index == 0 and seq.window is None
