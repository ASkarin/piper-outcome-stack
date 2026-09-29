import copy
import pytest
from test_xbox_pause import session as session, settle, tick, moves
from lerobot_robot_outcome_piper.action_audit import holding_reference
from lerobot_robot_outcome_piper.errors import OutcomePiperStateError, OutcomePiperIntentRejected


@pytest.mark.parametrize("width", [0.0, 0.012])
def test_start_acquires_measured_width_once_and_records_real_hold(session, width):
    robot, arm, control, clock = session
    control.recording_phase = "preparing"
    settle(session)
    arm.gripper.width = width
    before = len(moves(arm))
    reference = robot.prepare_recording_gripper(control.epoch)
    assert reference["initialized"]
    assert arm.gripper.commands == [(width, robot._safety.gripper_force_n)]
    assert len(moves(arm)) == before
    assert reference["command"]["result"] == "sdk_returned"
    tick(session)
    holding_reference(robot.last_action_telemetry)
    assert robot.last_action_telemetry["values"]["gripper.pos"] == width
    command = copy.deepcopy(robot._last_gripper_command)
    arm.gripper.width = max(
        0.0, width - 0.001
    )  # Actual compression must not reset the commanded grasp.
    assert not robot.prepare_recording_gripper(control.epoch)["initialized"]
    assert robot._last_gripper_command == command
    assert len(arm.gripper.commands) == 1


@pytest.mark.parametrize("width", [-0.001, 0.09])
def test_start_never_clips_invalid_width(session, width):
    robot, arm, control, _ = session
    control.recording_phase = "preparing"
    settle(session)
    arm.gripper.width = width
    with pytest.raises(OutcomePiperIntentRejected):
        robot.prepare_recording_gripper(control.epoch)
    assert not arm.gripper.commands and control.gripper_target is None


@pytest.mark.parametrize("reason", ["epoch", "unconfirmed", "running", "stale", "B", "sdk"])
def test_start_failure_does_not_create_gripper_reference(session, reason):
    robot, arm, control, _ = session
    control.recording_phase = "preparing"
    settle(session)
    epoch = control.epoch
    if reason == "epoch":
        epoch -= 1
    if reason == "unconfirmed":
        control.hold_confirmed = False
    if reason == "running":
        control.observe(True, True)
    if reason == "stale":
        robot._receiver.stale = True
    if reason == "B":
        robot.request_emergency_stop("test B")
    if reason == "sdk":
        arm.fail_command = True
    with pytest.raises((OutcomePiperStateError, OutcomePiperIntentRejected)):
        robot.prepare_recording_gripper(epoch)
    assert robot._last_gripper_command is None
    assert control.gripper_target is None
    if reason != "sdk":
        assert not arm.gripper.commands
