"""Xbox homing uses normal dispatch with fake devices, never actual hardware."""

from dataclasses import replace
import math

import pytest
from lerobot.types import TransitionKey
from lerobot_robot_outcome_piper.processor import make_xbox_processor
from lerobot_robot_outcome_piper.teleop_control import TeleopState, TeleopControl
from lerobot_robot_outcome_piper.joint_pose import JointPoseSequence
from lerobot_robot_outcome_piper.errors import OutcomePiperIntentRejected, OutcomePiperStateError
from test_xbox_pause import session as base_session, settle, moves  # noqa: F401


@pytest.fixture
def home(base_session, monkeypatch):  # noqa: F811
    robot, arm, c, clock = base_session
    arm.joints = [0.15, 0.12, -0.14, 0.13, -0.11, 0.10]
    robot._safety = replace(
        robot._safety,
        max_joint_step=(math.radians(5),) * 6,
        gripper_upper=0.03,
        max_gripper_step=0.005,
    )
    c.request_hold()
    settle(base_session)
    p = make_xbox_processor(
        robot._safety,
        max_xyz_step_m=0.0025,
        max_rotation_step_rad=0.0087,
        max_gripper_step_m=0.005,
        ik_max_nfev=100,
        ik_timeout_s=0.025,
        ik_residual_tolerance=1e-5,
        ik_min_singular_value=0.005,
    ).steps[0]
    p.control = c
    monkeypatch.setattr(p, "_solve", lambda *args: pytest.fail("homing must not use IK"))

    def tick(lb=False, y=False, a=False, x=0.0, b=False, send=True):
        raw = dict(
            stick_x=x,
            stick_y=0.0,
            stick_z=0.0,
            stick_yaw=0.0,
            left_trigger=0.0,
            right_trigger=0.0,
            hold=lb,
            home=y,
            work=a,
            neutral=x == 0.0,
            mode_switch=False,
            translation_switch=False,
            emergency_stop=b,
        )
        p._current_transition = {TransitionKey.OBSERVATION: robot.get_observation()}
        action = p.action(raw)
        action.generated_monotonic_s = clock.now
        if send:
            robot.send_action(action)
        return action

    def start():
        tick()
        tick(y=True)
        for _ in range(100):
            clock.advance()
            action = tick()
            if action.rejection_reason:
                return action
            if (
                c.state is TeleopState.POSE_READY
                and c.hold_confirmed
                and c.pose_sequence is not None
                and c.pose_sequence.planning_complete
            ):
                return tick(lb=True)
        pytest.fail("pose planning did not finish")

    return base_session, p, tick, start


def test_home_zero_sequence_closes_gripper_and_stays_enabled(home):
    (robot, arm, c, clock), p, tick, start = home
    start()
    targets = []
    for _ in range(100):
        sequence = c.pose_sequence
        if sequence is not None:
            arm.joints = sequence.targets[sequence.index][:6]
            arm.gripper.width = sequence.targets[sequence.index][6]
        clock.advance()
        action = tick(lb=True)
        assert robot._watchdog_check_locked()
        if action.intent == "pose":
            targets.append(dict(action))
            assert robot.last_action_telemetry["values"] == dict(action)
        if c.pose_event == "completed":
            break
    else:
        pytest.fail("homing did not complete")
    assert targets and c.state is TeleopState.HOLD_REQUESTED
    assert arm.gripper.commands[-1][0] == 0.0
    assert max(abs(q) for q in arm.joints) <= c.hold_settings.joint_tolerance_rad
    for _ in range(3):
        clock.advance()
        tick(lb=True)  # Holding LB after completion must not restart or run IK.
    assert c.state is TeleopState.PAUSED
    assert not any(call in ("disable", "reset", "home") for call in arm.calls)


def test_release_cancels_and_stale_waypoint_is_not_sent(home):
    (robot, arm, c, clock), p, tick, start = home
    start()
    stale = tick(lb=True, send=False)
    tick()
    assert c.pose_sequence is None and c.pose_event == "cancelled"
    count = len(moves(arm))
    robot.send_action(stale)
    assert len(moves(arm)) == count
    clock.advance()
    tick()
    tick(lb=True)
    assert c.state is not TeleopState.POSE_MOVING  # Needs a new Y request.


def test_stuck_home_waypoint_deadline_is_not_extended(home):
    (robot, arm, c, clock), p, tick, start = home
    start()
    # Dense references may initially fill the permitted lookahead. Once the
    # stationary arm prevents advancement, that pending reference must time out.
    for _ in range(len(c.pose_sequence.targets)):
        previous_index = c.pose_sequence.index
        clock.advance()
        tick(lb=True)
        if c.pose_sequence.index == previous_index:
            break
    else:
        pytest.fail("stationary arm did not bound pose lookahead")
    deadline = c.pose_sequence.window.deadline
    for _ in range(2):
        clock.advance()
        tick(lb=True)
        assert c.pose_sequence.window.deadline == deadline
    clock.advance(0.2)
    with pytest.raises(OutcomePiperStateError, match="pose tracking timed out"):
        tick(lb=True)
    assert c.state is TeleopState.FAULT


def test_home_b_preempts_and_axes_cancel(home):
    (robot, arm, c, clock), p, tick, start = home
    start()
    n = len(moves(arm))
    with pytest.raises(Exception, match="emergency stop"):
        tick(lb=True, y=True, b=True)
    assert c.state is TeleopState.E_STOP
    assert len(moves(arm)) == n


def test_home_route_must_remain_in_workspace(home):
    (robot, arm, c, clock), p, tick, start = home
    restricted = replace(
        robot._safety, workspace_lower=(5.0, 5.0, 5.0), workspace_upper=(6.0, 6.0, 6.0)
    )
    with pytest.raises(OutcomePiperIntentRejected, match="workspace"):
        JointPoseSequence(arm.joints, 0.03, restricted, c.hold_settings, [0.0] * 7, "home")


def test_y_startup_long_press_and_invalid_requests_do_not_queue():
    c = TeleopControl()
    c.confirm_hold()
    c.observe(False, True, home=True)
    assert c.state is TeleopState.WAITING
    c.observe(False, True, home=True)
    assert c.state is TeleopState.WAITING
    c.observe(False, True)
    c.observe(False, False, home=True)
    c.observe(False, True, home=True)
    assert c.state is TeleopState.WAITING
    c.observe(False, True)
    c.observe(False, True, home=True)
    assert c.state is TeleopState.POSE_READY
    epoch = c.epoch
    c.observe(False, True, home=True)
    assert c.epoch == epoch
    c.confirm_hold()
    c.observe(False, True)
    from types import SimpleNamespace

    c.pose_sequence = SimpleNamespace(planning_complete=True, control_epoch=c.epoch)
    c.observe(True, True)
    assert c.state is TeleopState.POSE_MOVING
    c.observe(True, False)
    assert c.state is TeleopState.HOLD_REQUESTED


def test_home_sdk_failure_does_not_commit_a_waypoint(home, monkeypatch):
    (robot, arm, c, clock), p, tick, start = home
    start()
    sequence = c.pose_sequence
    sequence.window = None
    monkeypatch.setattr(arm, "move_j", lambda q: (_ for _ in ()).throw(OSError("send failed")))
    with pytest.raises(OutcomePiperStateError, match="send failed"):
        tick(lb=True)
    assert sequence.window is None
    assert c.state is TeleopState.FAULT


def test_home_stale_feedback_stops_before_next_command(home):
    (robot, arm, c, clock), p, tick, start = home
    start()
    n = len(moves(arm))
    robot._receiver.stale = True
    with pytest.raises(OutcomePiperStateError):
        tick(lb=True)
    assert len(moves(arm)) == n


def test_home_button_mapping_conflict_rejected():
    from test_plugin import xbox_config

    with pytest.raises(ValueError, match="distinct"):
        xbox_config(home_button=1)


def test_zero_outside_limits_rejects_before_any_command(home):
    (robot, arm, c, clock), p, tick, start = home
    p.safety = replace(p.safety, joint_lower=(0.01, *p.safety.joint_lower[1:]))
    n = len(moves(arm))
    action = start()
    assert action.intent == "hold"
    assert "pose joint target is outside" in action.rejection_reason
    assert all(call[1] == arm.joints for call in moves(arm)[n:])


def test_home_boundary_feedback_within_arrival_tolerance_keeps_legal_hold(home):
    (robot, arm, c, clock), p, tick, start = home
    bounds = replace(
        robot._safety,
        joint_lower=(-2.0, 0.0, -2.0, -2.0, -2.0, -2.0),
        joint_upper=(2.0, 2.0, 0.0, 2.0, 2.0, 2.0),
    )
    robot._safety = p.safety = bounds
    start()
    saw_boundary = False
    for _ in range(100):
        sequence = c.pose_sequence
        if sequence is not None:
            arm.joints = list(sequence.targets[sequence.index][:6])
            arm.gripper.width = sequence.targets[sequence.index][6]
            if sequence.index == len(sequence.targets) - 1:
                arm.joints[1] = -c.hold_settings.joint_tolerance_rad / 2
                arm.joints[2] = c.hold_settings.joint_tolerance_rad / 2
        clock.advance()
        action = tick(lb=True)
        saw_boundary |= bool(action.feedback_limit_events)
        if c.pose_event == "completed":
            break
    else:
        pytest.fail("boundary feedback prevented homing completion")
    assert saw_boundary
    assert arm.joints[1] < 0 < arm.joints[2]  # Feedback was not clipped.
    assert robot._hold_window.target[1:3] == [0.0, 0.0]
    assert robot._hold_command["measured_joint_rad"][1] < 0
    for _ in range(3):
        clock.advance()
        tick(lb=True)
        assert robot._watchdog_check_locked()
    assert c.state is TeleopState.PAUSED
    for call in moves(arm):
        assert call[1][1] >= 0 and call[1][2] <= 0
    again = start()  # A new explicit Y return at the boundary also plans legal targets.
    assert again.intent == "pose" and again["joint_2.pos"] == again["joint_3.pos"] == 0.0


def test_feedback_tolerance_requires_legal_sent_target_and_reports_excess(home):
    from lerobot_robot_outcome_piper.safety import check_joint_feedback
    from lerobot_robot_outcome_piper.errors import OutcomePiperValidationError

    (robot, arm, c, clock), p, tick, start = home
    bounds = replace(robot._safety, joint_lower=(-2.0, 0.0, -2.0, -2.0, -2.0, -2.0))
    q = [0.0, -0.0005, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(OutcomePiperValidationError, match="last_target_rad"):
        check_joint_feedback(q, bounds, tolerance=0.001)
    with pytest.raises(OutcomePiperValidationError, match="excess_rad"):
        check_joint_feedback(q, bounds, target=[0.0, -0.0005, 0.0, 0.0, 0.0, 0.0], tolerance=0.001)
    with pytest.raises(OutcomePiperValidationError, match="measured_joint_rad"):
        check_joint_feedback(q, bounds, target=[0.0] * 6, tolerance=0.0001)
    events = check_joint_feedback(q, bounds, target=[0.0] * 6, tolerance=0.001)
    assert events[0]["joint"] == 2 and events[0]["excess_rad"] == 0.0005
    from lerobot_robot_outcome_piper.safety import JOINT_KEYS

    values = {**dict(zip(JOINT_KEYS, arm.joints)), "gripper.pos": 0.03}
    values["joint_2.pos"] = -0.0005
    robot._safety = bounds
    with pytest.raises(OutcomePiperIntentRejected, match="outside frozen limits"):
        robot._validate_action(values)


def test_zero_exit_boundary_excursion_is_not_tracking_lag(home):
    from lerobot_robot_outcome_piper.safety import check_joint_feedback

    (robot, arm, c, clock), p, tick, start = home
    bounds = replace(
        robot._safety,
        joint_lower=(-2.0, 0.0, -2.0, -2.0, -2.0, -2.0),
        joint_upper=(2.0, 2.0, 0.0, 2.0, 2.0, 2.0),
    )
    robot._safety = p.safety = bounds
    q = [0.0, math.radians(0.03), math.radians(0.062), 0.0, 0.0, 0.0]
    previous = [0.0, math.radians(0.03), math.radians(-0.060), 0.0, 0.0, 0.0]
    c.joint_target = tuple(previous)
    arm.joints = list(q)
    event = check_joint_feedback(
        q, bounds, target=previous, tolerance=c.hold_settings.joint_tolerance_rad
    )[0]
    assert math.degrees(event["excess_rad"]) == pytest.approx(0.062)
    assert math.degrees(event["tracking_error_rad"]) == pytest.approx(0.122)
    assert event["boundary_rad"] == 0.0
    # Both the processor and dispatch validator must accept the same fresh sample.
    action = tick(send=False)
    assert action.feedback_limit_events[0]["boundary_rad"] == 0.0
    values = {**dict(zip(robot.action_features, previous + [arm.gripper.width]))}
    joints, _ = robot._validate_action(values)
    assert joints[2] == previous[2] and arm.joints == q
    # Ordinary target-to-feedback step validation still rejects a large command.
    values["joint_3.pos"] = math.radians(-6.0)
    with pytest.raises(OutcomePiperIntentRejected, match="exceeds frozen step"):
        robot._validate_action(values)


def test_boundary_hold_uses_nearest_legal_boundary_not_advanced_target(home):
    (robot, arm, c, clock), p, tick, start = home
    robot._safety = p.safety = replace(
        robot._safety,
        joint_lower=(-2.0, 0.0, -2.0, -2.0, -2.0, -2.0),
        joint_upper=(2.0, 2.0, 0.0, 2.0, 2.0, 2.0),
    )
    arm.joints = [0.0, math.radians(0.03), math.radians(0.062), 0.0, 0.0, 0.0]
    c.joint_target = (0.0, math.radians(0.03), math.radians(-0.060), 0.0, 0.0, 0.0)
    robot._start_hold_locked()
    assert robot._hold_window.target[2] == 0.0
    assert robot._hold_command["measured_joint_rad"][2] == math.radians(0.062)
    assert arm.joints[2] == math.radians(0.062)


def test_actual_boundary_excess_still_fails_with_inward_target(home):
    from lerobot_robot_outcome_piper.safety import check_joint_feedback
    from lerobot_robot_outcome_piper.errors import OutcomePiperValidationError

    (robot, arm, c, clock), p, tick, start = home
    bounds = replace(robot._safety, joint_upper=(2.0, 2.0, 0.0, 2.0, 2.0, 2.0))
    q = [0.0, 0.0, math.radians(0.101), 0.0, 0.0, 0.0]
    previous = [0.0, 0.0, math.radians(-0.060), 0.0, 0.0, 0.0]
    with pytest.raises(OutcomePiperValidationError, match="excess_rad"):
        check_joint_feedback(q, bounds, target=previous, tolerance=math.radians(0.1))


def test_work_pose_uses_shared_executor_and_recording_rejects_requests(home):
    (robot, arm, c, clock), p, tick, start = home
    p.work_joint_rad = (0.1, 0.08, -0.07, 0.06, -0.05, 0.04)
    p.work_gripper_m = 0.0
    tick()
    tick(a=True, y=True)
    assert c.state is not TeleopState.POSE_READY
    tick()
    c.recording_phase = "recording"
    tick(a=True)
    assert c.pose_event == "request_rejected" and c.state is not TeleopState.POSE_READY
    c.recording_phase = "preparing"
    tick(a=True)  # Press made during recording must not be queued.
    assert c.state is not TeleopState.POSE_READY
    tick()
    tick(a=True)
    for _ in range(100):
        clock.advance()
        tick()
        if c.pose_sequence is not None and c.pose_sequence.planning_complete:
            break
    tick(lb=True)
    assert c.pose_kind == "work" and c.state is TeleopState.POSE_MOVING
    for _ in range(100):
        if c.pose_sequence:
            values = c.pose_sequence.values
            arm.joints = list(values.values())[:6]
            arm.gripper.width = values["gripper.pos"]
        clock.advance()
        tick(lb=True)
        if c.pose_event == "completed":
            break
    else:
        pytest.fail("work pose did not complete")
    assert arm.joints == list(p.work_joint_rad)
    assert arm.gripper.commands[-1][0] == 0.0
    assert c.state is TeleopState.HOLD_REQUESTED


def test_work_pose_config_requires_measured_complete_binding():
    from test_plugin import xbox_config

    with pytest.raises(ValueError, match="requires"):
        xbox_config(work_joint_rad=(0.0,) * 6)
    with pytest.raises(ValueError, match="distinct"):
        xbox_config(work_pose_button=1, work_joint_rad=(0.0,) * 6, work_gripper_m=0.0)
    cfg = xbox_config(work_pose_button=4, work_joint_rad=(0.0,) * 6, work_gripper_m=0.0)
    assert cfg.work_pose_button == 4


def test_rejected_replacement_cancels_old_pending_pose():
    c = TeleopControl()
    c.confirm_hold()
    c.observe(False, True)
    c.observe(False, True, work=True)
    assert c.state is TeleopState.POSE_READY and c.pose_kind == "work"
    c.confirm_hold()
    c.observe(False, True)
    c.observe(False, True, home=True, work=True)
    assert c.state is TeleopState.HOLD_REQUESTED
    assert c.pose_event == "request_rejected"
    c.confirm_hold()
    c.observe(False, True)
    assert c.observe(True, True)[0] != "pose"
