import pytest
from lerobot.types import TransitionKey
from lerobot_robot_outcome_piper.processor import make_xbox_processor
from lerobot_robot_outcome_piper.action_audit import holding_reference
from lerobot_robot_outcome_piper.teleop_control import TeleopState
from test_xbox_pause import session as base_session, settle, moves  # noqa: F401


@pytest.fixture
def setup(base_session, monkeypatch):  # noqa: F811
    robot, arm, c, clock = base_session
    settle(base_session)
    p = make_xbox_processor(
        robot._safety,
        max_xyz_step_m=0.0025,
        max_rotation_step_rad=0.0087,
        max_gripper_step_m=0.004,
        ik_max_nfev=100,
        ik_timeout_s=0.025,
        ik_residual_tolerance=1e-5,
        ik_min_singular_value=0.005,
    ).steps[0]
    p.control = c
    from lerobot_robot_outcome_piper.teleop_control import TranslationStrategy

    c.translation_strategy = TranslationStrategy.FIXED_ORIENTATION
    solves = []
    monkeypatch.setattr(
        p, "_solve", lambda q, pose: solves.append((q, pose)) or [x + 0.01 for x in q]
    )

    def tick(x=0.0, right=0.0, left=0.0, lb=True, send=True):
        raw = dict(
            stick_x=x,
            stick_y=0.0,
            stick_z=0.0,
            stick_yaw=0.0,
            left_trigger=left,
            right_trigger=right,
            hold=lb,
            neutral=x == right == left == 0.0,
            emergency_stop=False,
            mode_switch=False,
            translation_switch=False,
            home=False,
            work=False,
        )
        p._current_transition = {TransitionKey.OBSERVATION: robot.get_observation()}
        action = p.action(raw)
        action.generated_monotonic_s = clock.now
        if send:
            robot.send_action(action)
        return action

    tick()
    tick(x=0.1)
    arm.joints = [0.005] * 6  # Actual arm has not reached the previous waypoint.
    return base_session, p, solves, tick


def test_center_captures_once_gripper_independent_and_fresh_input_resumes(setup):
    (robot, arm, c, clock), p, solves, tick = setup
    n = len(moves(arm))
    initial_solves = len(solves)
    action = tick(right=1, send=False)
    assert c.state is TeleopState.CENTERING
    assert robot._watchdog_check_locked() and not c.hold_confirmed
    robot.send_action(action)
    assert len(moves(arm)) == n + 1 and moves(arm)[-1][1] == arm.joints
    target = robot._hold_window.target[:]
    reference = holding_reference(robot.last_action_telemetry)
    assert reference["values"]["gripper.pos"] == pytest.approx(0.034)
    tick(x=0.1)  # New stick input during unconfirmed hold is not queued.
    assert len(solves) == initial_solves
    clock.advance()
    tick()
    assert c.state is TeleopState.CENTERED
    arm.gripper.width = 0.031
    old_id = robot._hold_id
    tick(right=1)
    assert len(moves(arm)) == n + 1 and robot._hold_window.target == target
    assert arm.gripper.commands[-1][0] == pytest.approx(0.035)
    assert robot._hold_id != old_id
    holding_reference(robot.last_action_telemetry)
    grip_calls = list(arm.gripper.commands)
    for _ in range(80):
        clock.advance()
        tick()
        assert robot._watchdog_check_locked()
    assert arm.gripper.commands == grip_calls and len(moves(arm)) == n + 1
    assert len(solves) == initial_solves and c.state is TeleopState.CENTERED
    tick(x=-0.1)
    assert c.state is TeleopState.RUNNING and len(solves) == initial_solves + 1
    assert solves[-1][0] == arm.joints


def test_lb_release_disarms_even_when_trigger_remains_pressed(setup):
    (robot, arm, c, clock), p, solves, tick = setup
    tick()
    clock.advance()
    tick()
    calls = len(arm.gripper.commands)
    tick(right=1, lb=False)
    clock.advance()
    tick(right=1, lb=False)
    tick(right=1, lb=True)
    assert c.state is not TeleopState.RUNNING
    assert len(arm.gripper.commands) == calls


def test_center_drift_reconfirms_same_target_without_rewriting(setup):
    (robot, arm, c, clock), p, solves, tick = setup
    tick()
    clock.advance()
    tick()
    target = robot._hold_window.target[:]
    count = len(moves(arm))
    arm.joints[0] += 0.03
    clock.advance()
    tick()
    assert c.state is TeleopState.CENTERING and robot._hold_window.target == target
    arm.joints = target[:]
    clock.advance()
    tick()
    clock.advance()
    tick()
    assert c.state is TeleopState.CENTERED and len(moves(arm)) == count


def test_b_during_center_hold_prevents_gripper_command(setup):
    (robot, arm, c, clock), p, solves, tick = setup
    before = len(arm.gripper.commands)
    original = arm.move_j

    def stop(q):
        original(q)
        robot.request_emergency_stop("B during hold")

    arm.move_j = stop
    with pytest.raises(RuntimeError):
        tick(right=1)
    assert len(arm.gripper.commands) == before
    assert arm.calls.count("electronic_emergency_stop") == 1
    assert "disable" not in arm.calls


def test_input_loss_while_centered_holds_then_faults_without_estop(setup):
    (robot, arm, c, clock), p, solves, tick = setup
    tick()
    clock.advance()
    tick()
    assert c.state is TeleopState.CENTERED
    clock.auto = 0.001
    robot.request_input_fault("Xbox disconnected")
    assert robot.state.value == "FAULT" and robot.stop_outcome == "hold_confirmed"
    assert "electronic_emergency_stop" not in arm.calls


def test_stale_center_gripper_intent_cannot_cross_arm_resume(setup):
    (robot, arm, c, clock), p, solves, tick = setup
    tick()
    clock.advance()
    tick()
    stale = tick(right=1, send=False)
    tick(x=-0.1)
    before = list(arm.gripper.commands)
    robot.send_action(stale)
    assert arm.gripper.commands == before
    assert robot.last_action_telemetry["result"] == "discarded"


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("endpoint", ["lower", "upper"])
def test_gripper_endpoint_keeps_session_and_reverse_works(setup, moving, endpoint):
    (robot, arm, c, clock), p, solves, tick = setup
    from dataclasses import replace

    robot._safety = replace(robot._safety, gripper_upper=0.03, max_gripper_step=0.005)
    p.safety = robot._safety
    p.max_gripper_step_m = 0.005
    closing = endpoint == "lower"
    boundary = 0.0 if closing else 0.03
    arm.gripper.width = 0.0022 if closing else 0.028
    kwargs = dict(x=0.1 if moving else 0.0, left=float(closing), right=float(not closing))
    action = tick(**kwargs)
    assert action["gripper.pos"] == boundary
    assert action.gripper_plan["travel_shortened"]
    assert action.gripper_plan["boundary"] == endpoint
    assert robot.last_action_telemetry["gripper_plan"] == action.gripper_plan
    assert arm.gripper.commands[-1][0] == boundary
    arm.gripper.width = boundary
    clock.advance()
    tick(**kwargs)
    before = len(arm.gripper.commands)
    for _ in range(3):
        clock.advance()
        action = tick(**kwargs)
        assert action.gripper_plan["planned_delta_m"] == 0
        assert c.state is (TeleopState.RUNNING if moving else TeleopState.CENTERED)
    assert len(arm.gripper.commands) == before
    if not moving:
        assert holding_reference(robot.last_action_telemetry)["values"]["gripper.pos"] == boundary
    # No LB release/rearm between the endpoint and reversing the trigger.
    action = tick(x=kwargs["x"], left=float(not closing), right=float(closing))
    expected = 0.005 if closing else 0.025
    assert action["gripper.pos"] == pytest.approx(expected)
    assert arm.gripper.commands[-1][0] == pytest.approx(expected)


def test_absolute_gripper_target_still_rejected(setup):
    (robot, arm, c, clock), p, solves, tick = setup
    from lerobot_robot_outcome_piper.errors import OutcomePiperIntentRejected

    with pytest.raises(OutcomePiperIntentRejected):
        robot._validate_gripper_target(robot._safety.gripper_upper + 0.001, 0.03)
