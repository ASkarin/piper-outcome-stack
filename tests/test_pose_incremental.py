"""No physical hardware. Exercise planning on the ordinary hold/dispatch path."""

from dataclasses import asdict, replace
import pytest

from lerobot_robot_outcome_piper.joint_pose import JointPoseSequence
from lerobot_robot_outcome_piper.teleop_control import TeleopState
from lerobot_robot_outcome_piper.errors import OutcomePiperStateError, OutcomePiperIntentRejected
from test_xbox_home import home as home, base_session as base_session
from test_pose_lookahead import sequence
from test_pose_timing import timing
from test_xbox_pause import moves


def test_incremental_path_identical_and_not_executable_until_fully_checked(tmp_path):
    old, safety, start = sequence(tmp_path)
    args = (start, 0.0, safety, old.settings, [0.0] * 7, "home")
    expected = JointPoseSequence(*args, timing=timing())
    actual = JointPoseSequence(*args, timing=timing(), incremental=True)
    assert not actual.targets
    for operation in (lambda: actual.values, actual.next_waypoint, lambda: actual.sent(0)):
        with pytest.raises(OutcomePiperStateError, match="validation"):
            operation()
    batches = 0
    while not actual.planning_complete:
        prior = len(actual.targets)
        actual.plan_chunk()
        assert 0 < len(actual.targets) - prior <= 16
        batches += 1
    assert batches > 1
    assert actual.targets == expected.targets
    assert actual.nominal_duration_s == expected.nominal_duration_s
    assert actual.telemetry()["planning_batches"] == batches


def test_budget_yields_after_slow_waypoint(tmp_path, monkeypatch):
    old, safety, start = sequence(tmp_path)
    s = JointPoseSequence(
        start, 0.0, safety, old.settings, [0.0] * 7, "home", timing=timing(), incremental=True
    )
    clock = [0.0]
    from lerobot_robot_outcome_piper import joint_pose

    monkeypatch.setattr(joint_pose.time, "perf_counter", lambda: clock[0])
    fk = s._fk

    def slow(*args):
        clock[0] += 0.002
        return fk(*args)

    s._fk = slow
    s.plan_chunk()
    assert len(s.targets) == 2
    assert s.planning_max_batch_s == pytest.approx(0.004)


def begin_plan(home):
    (robot, arm, c, clock), p, tick, _ = home
    p.pose_timing = asdict(replace(timing(), period_s=0.02))
    tick()
    tick(y=True)
    for _ in range(10):
        clock.advance()
        tick()
        if c.pose_sequence is not None:
            break
    assert c.pose_sequence is not None and not c.pose_sequence.planning_complete
    return c.pose_sequence


def test_pending_lb_holds_until_complete_then_starts_on_next_fresh_tick(home):
    (robot, arm, c, clock), p, tick, _ = home
    s = begin_plan(home)
    held = len(moves(arm))
    assert tick(lb=True).intent == "hold"
    for _ in range(500):
        clock.advance()
        a = tick(lb=True)
        assert robot._watchdog_check_locked()
        if a.intent == "pose":
            break
        assert len(moves(arm)) == held
    else:
        pytest.fail("pending pose never started")
    assert s.planning_complete and s.window is not None
    assert c.state is TeleopState.POSE_MOVING
    assert len(moves(arm)) == held + 1
    assert robot.last_action_telemetry["values"] == dict(a)


@pytest.mark.parametrize("cancel", ["release", "axis", "B", "epoch"])
def test_cancel_during_planning_never_restarts_old_request(home, cancel):
    (robot, arm, c, clock), p, tick, _ = home
    s = begin_plan(home)
    tick(lb=True)
    if cancel == "B":
        n = len(moves(arm))
        with pytest.raises(Exception, match="emergency"):
            tick(lb=True, b=True)
        assert len(moves(arm)) == n
    elif cancel == "epoch":
        c.request_hold()
    else:
        tick(lb=cancel != "release", x=0.5 if cancel == "axis" else 0.0)
    assert c.pose_sequence is None
    assert not s.planning_complete
    if cancel != "B":
        for _ in range(4):
            clock.advance()
            a = tick(lb=True)
            assert a.intent != "pose"
        assert c.state is not TeleopState.POSE_MOVING


def test_completed_plan_waits_for_new_lb_and_rejects_moved_start(home):
    (robot, arm, c, clock), p, tick, _ = home
    s = begin_plan(home)
    while not s.planning_complete:
        clock.advance()
        tick()
    count = len(moves(arm))
    tick()
    assert c.state is TeleopState.POSE_READY and len(moves(arm)) == count
    arm.joints[0] += c.hold_settings.joint_tolerance_rad * 2
    action = tick(lb=True)
    assert action.intent == "hold" and "start changed" in action.rejection_reason
    assert c.pose_sequence is None
    assert all(call[1] == arm.joints for call in moves(arm)[count:])


def test_late_workspace_failure_sends_no_partial_trajectory(home, monkeypatch):
    (robot, arm, c, clock), p, tick, _ = home
    s = begin_plan(home)
    from lerobot_robot_outcome_piper import joint_pose

    monkeypatch.setattr(joint_pose, "workspace_pose_allowed", lambda *args, **kw: False)
    n = len(moves(arm))
    a = tick(lb=True)
    assert a.intent == "hold" and "workspace" in a.rejection_reason
    assert not s.planning_complete and c.pose_sequence is None
    assert all(call[1] == arm.joints for call in moves(arm)[n:])


def test_stale_feedback_during_plan_exits_without_pose(home):
    (robot, arm, c, clock), p, tick, _ = home
    begin_plan(home)
    n = len(moves(arm))
    robot._receiver.stale = True
    with pytest.raises(OutcomePiperStateError):
        tick(lb=True)
    assert len(moves(arm)) == n and c.state is TeleopState.FAULT


def test_gripper_start_drift_preserves_execution_step_limit(tmp_path):
    old, safety, start = sequence(tmp_path)
    s = JointPoseSequence(start, 0.0, safety, old.settings, [0.0] * 7, "home")
    with pytest.raises(OutcomePiperIntentRejected, match="gripper start changed"):
        s.validate_start(start, safety.max_gripper_step + 0.001)


def test_processor_reset_discards_prepared_route(home):
    (_, _, c, _), p, tick, _ = home
    begin_plan(home)
    tick(lb=True)
    p.reset()
    assert c.pose_sequence is None and c.state is TeleopState.HOLD_REQUESTED
    assert c.pose_cancel_reason == "processor_reset"


def test_b_during_batch_cannot_republish_cancelled_plan(home):
    (robot, arm, c, _), _, tick, _ = home
    seq = begin_plan(home)
    fk = seq._fk
    fired = []

    def stop(*args):
        if not fired:
            fired.append(True)
            robot.request_emergency_stop("B during planning")
        return fk(*args)

    seq._fk = stop
    count = len(moves(arm))
    with pytest.raises(OutcomePiperStateError, match="B during planning"):
        tick(lb=True)
    assert c.pose_sequence is None and c.state is TeleopState.E_STOP
    assert len(moves(arm)) == count


def test_reference_is_rechecked_against_feedback_at_dispatch(home):
    (robot, arm, c, clock), _, tick, _ = home
    seq = begin_plan(home)
    while not seq.planning_complete:
        clock.advance()
        tick()
    action = tick(lb=True, send=False)
    assert action.intent == "pose"
    # Drift the gripper after processing, while joint hold remains confirmed.
    arm.gripper.width += robot._safety.max_gripper_step + 0.001
    robot.send_action(action)
    assert "gripper start changed" in robot.last_action_telemetry["rejection_reason"]
    assert c.pose_sequence is None
