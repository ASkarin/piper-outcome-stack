import pytest
from test_plugin import make_robot, connect_for_test
from test_xbox_pause import Clock, Receiver
from lerobot_robot_outcome_piper.robot import PiperState


def setup(tmp_path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    clock = Clock()
    robot._monotonic = clock
    robot._receiver_factory = Receiver
    robot._start_watchdog = lambda: None
    connect_for_test(robot)
    return robot, arm, clock


def test_fresh_observation_keeps_loop_alive_without_commands(tmp_path):
    robot, arm, clock = setup(tmp_path)
    before = list(arm.calls)
    grip = list(arm.gripper.commands)
    for _ in range(3):
        clock.advance(robot._safety.watchdog_timeout_s * 0.75)
        robot.observe_control()
        assert robot._watchdog_check_locked()
    assert arm.calls == before and arm.gripper.commands == grip
    clock.advance(robot._safety.watchdog_timeout_s + 0.01)
    assert not robot._watchdog_check_locked() and robot.state is PiperState.FAULT
    robot.disconnect()


def test_stale_feedback_does_not_refresh_heartbeat(tmp_path):
    robot, arm, clock = setup(tmp_path)
    previous = robot._last_control_tick_s
    robot._receiver.stale = True
    with pytest.raises(Exception):
        robot.observe_control()
    assert robot._last_control_tick_s == previous and robot.state is PiperState.FAULT
    robot.disconnect()


def test_read_only_session_cannot_claim_motion_heartbeat(tmp_path):
    robot, arm, _ = make_robot(tmp_path, mode="read_only")
    robot.connect()
    before = list(arm.calls)
    with pytest.raises(Exception, match="requires ACTIVE"):
        robot.observe_control()
    assert arm.calls == before
    robot.disconnect()


def test_faulted_session_cannot_be_revived_by_observation(tmp_path):
    robot, arm, clock = setup(tmp_path)
    robot.request_emergency_stop("test")
    previous = robot._last_control_tick_s
    with pytest.raises(Exception):
        robot.observe_control()
    assert robot.state is PiperState.E_STOP and robot._last_control_tick_s == previous
    robot.disconnect()


def test_observation_only_wait_keeps_joint_limit_check(tmp_path):
    robot, arm, clock = setup(tmp_path)
    previous = robot._last_control_tick_s
    arm.joints[3] = 100.0
    with pytest.raises(Exception, match="joint feedback outside"):
        robot.observe_control()
    assert robot.state is PiperState.FAULT and robot._last_control_tick_s == previous
    robot.disconnect()


def test_late_tick_cannot_hide_stall_before_watchdog_thread_runs(tmp_path):
    robot, arm, clock = setup(tmp_path)
    previous = robot._last_control_tick_s
    clock.advance(robot._safety.watchdog_timeout_s + 0.01)
    with pytest.raises(Exception, match="watchdog"):
        robot.observe_control()
    assert robot.state is PiperState.FAULT and robot._last_control_tick_s == previous
    robot.disconnect()
