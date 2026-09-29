"""Servo lifecycle uses explicit operations and fresh feedback, never cleanup torque loss."""

import pytest
from test_plugin import make_robot
from test_xbox_pause import Clock, Receiver
from lerobot_robot_outcome_piper.robot import PiperState, ServoState
from lerobot_robot_outcome_piper.errors import OutcomePiperStateError


def setup_robot(tmp_path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    clock = Clock()
    robot._monotonic = clock
    robot._receiver_factory = Receiver
    robot._start_watchdog = lambda: None
    return robot, arm, clock


def test_connect_observes_already_enabled_without_control_commands(tmp_path):
    robot, arm, clock = setup_robot(tmp_path)
    arm.enabled = True
    arm.ctrl_mode = 0
    robot.connect()
    try:
        assert robot.state is PiperState.CONNECTED
        assert robot.get_servo_status().state is ServoState.ENABLED
        assert not any(
            (x if isinstance(x, str) else x[0])
            in {
                "enable",
                "disable",
                "motion_mode",
                "speed_percent",
                "move_j",
                "electronic_emergency_stop",
            }
            for x in arm.calls
        )
        arm.ctrl_mode = 1
        robot.enable()
        assert robot.state is PiperState.ACTIVE and "enable" not in arm.calls
        before = list(arm.calls)
        robot.enable()
        assert arm.calls == before
    finally:
        robot.disconnect()
    assert arm.enabled and "disable" not in arm.calls
    assert robot.get_servo_status().state is ServoState.UNKNOWN
    assert robot.last_servo_feedback.state is ServoState.ENABLED


@pytest.mark.parametrize("include_gripper", [False, True])
def test_explicit_disable_confirms_and_can_enable_again(tmp_path, include_gripper):
    robot, arm, clock = setup_robot(tmp_path)
    robot.connect()
    robot.enable()
    arm.gripper.enabled = True
    robot.disable(include_gripper=include_gripper)
    assert robot.state is PiperState.CONNECTED
    assert robot.get_servo_status().state is ServoState.DISABLED
    assert arm.gripper.enabled is (not include_gripper)
    assert arm.calls.count("disable") == 1
    assert robot.last_servo_command["result"] == "confirmed"
    robot.enable()
    assert robot.state is PiperState.ACTIVE
    assert arm.calls.count("enable") == 2
    robot.disconnect()
    assert arm.enabled


def test_disable_requires_gripper_choice_and_read_only_rejects(tmp_path):
    robot, arm, _ = make_robot(tmp_path)
    robot.connect()
    try:
        with pytest.raises(TypeError):
            robot.disable()
        with pytest.raises(OutcomePiperStateError):
            robot.disable(include_gripper=False)
        with pytest.raises(OutcomePiperStateError):
            robot.enable()
        assert "disable" not in arm.calls and "enable" not in arm.calls
    finally:
        robot.disconnect()


def test_disable_timeout_latches_without_extra_stop_or_resend(tmp_path):
    robot, arm, clock = setup_robot(tmp_path)
    robot.connect()
    robot.enable()
    arm.disable = lambda: arm.calls.append("disable")
    clock.auto = 0.001
    with pytest.raises(OutcomePiperStateError, match="disable confirmation timed out"):
        robot.disable(include_gripper=False)
    assert robot.state is PiperState.FAULT
    assert arm.calls.count("disable") == 1
    assert "electronic_emergency_stop" not in arm.calls
    robot.disconnect()


def test_partial_and_stale_feedback_are_not_reported_disabled(tmp_path):
    robot, arm, clock = setup_robot(tmp_path)
    robot.connect()
    original = robot._receiver.driver_states

    def mixed():
        states = original()
        states[0][0].msg.foc_status.driver_enable_status = True
        return states

    robot._receiver.driver_states = mixed
    assert robot.get_servo_status().state is ServoState.PARTIAL
    robot._receiver.driver_states = lambda: tuple((s, clock.now - 1) for s, t in original())
    assert robot.get_servo_status().state is ServoState.UNKNOWN
    assert robot.last_servo_feedback.state is ServoState.PARTIAL
    robot.disconnect()


def test_stop_during_joint_disable_prevents_gripper_disable(tmp_path):
    robot, arm, clock = setup_robot(tmp_path)
    robot.connect()
    robot.enable()
    arm.gripper.enabled = True

    def disable_and_request_stop():
        arm.calls.append("disable")
        arm.enabled = False
        robot._emergency_stop_cause = "operator B"
        robot._emergency_stop_requested.set()

    arm.disable = disable_and_request_stop
    with pytest.raises(OutcomePiperStateError, match="operator B"):
        robot.disable(include_gripper=True)
    assert "disable" not in arm.gripper.commands
    assert arm.gripper.enabled
    assert arm.calls.count("electronic_emergency_stop") == 1
    robot.disconnect()
    assert arm.calls.count("electronic_emergency_stop") == 1


def test_replay_explicit_enable_precedes_dispatch_and_cleanup(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS
    from lerobot.scripts import lerobot_replay as official
    from lerobot_robot_outcome_piper import workflows
    from lerobot_robot_outcome_piper.safety import ACTION_KEYS

    robot, arm, clock = setup_robot(tmp_path)
    values = [0.0] * 6 + [0.03]
    dataset = NS(
        features={"action": {"names": list(ACTION_KEYS)}},
        select_columns=lambda _: [{"action": values}],
        num_frames=1,
        fps=30,
    )
    monkeypatch.setattr(official, "LeRobotDataset", lambda *a, **kw: dataset)
    monkeypatch.setattr(workflows, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(official, "log_say", lambda *a, **kw: None)
    workflows.replay(
        NS(
            robot=robot.config,
            dataset=NS(repo_id="local/test", root=tmp_path, episode=0),
            play_sounds=False,
        )
    )
    names = [c if isinstance(c, str) else c[0] for c in arm.calls]
    assert (
        names.index("connect")
        < names.index("enable")
        < names.index("move_j")
        < names.index("disconnect")
    )
    assert "disable" not in names and arm.enabled


@pytest.mark.parametrize("teach_status", [1, 3, 4])
def test_startup_teaching_preflight_never_writes_control(tmp_path, teach_status):
    robot, arm, clock = setup_robot(tmp_path)
    robot.connect()
    arm.teach_status = teach_status
    arm.ctrl_mode = 2
    before = list(arm.calls)
    with pytest.raises(OutcomePiperStateError, match="stop teaching first"):
        robot.enable()
    assert arm.calls == before
    assert not robot._control_started
    robot.disconnect()
    assert "electronic_emergency_stop" not in arm.calls


def test_startup_stale_status_never_writes_control(tmp_path):
    robot, arm, clock = setup_robot(tmp_path)
    robot.connect()
    original = robot._receiver.status
    robot._receiver.status = lambda: (original()[0], clock.now - 1)
    before = list(arm.calls)
    with pytest.raises(OutcomePiperStateError, match="stale"):
        robot.enable()
    assert arm.calls == before
    robot.disconnect()


def test_stopped_teaching_can_transition_to_control(tmp_path):
    robot, arm, _ = setup_robot(tmp_path)
    robot.connect()
    arm.teach_status = 2
    arm.ctrl_mode = 2
    original = arm.set_motion_mode

    def switch(mode):
        original(mode)
        arm.ctrl_mode = 1

    arm.set_motion_mode = switch
    robot.enable()
    assert robot.state is PiperState.ACTIVE
    robot.disconnect()
