"""Xbox control transitions and SDK dispatch with synthetic devices only."""

from dataclasses import replace
import json
from types import SimpleNamespace as NS
import threading

import pytest

from test_plugin import make_robot, FakeReceiver, valid_action, xbox_config, FakeJoystick
from lerobot_robot_outcome_piper.teleop_control import TeleopControl, TeleopState
from lerobot_robot_outcome_piper.processor import OutcomePiperAction
from lerobot_robot_outcome_piper.robot import PiperState
from lerobot_robot_outcome_piper.teleoperator import OutcomePiperXbox
from lerobot_robot_outcome_piper.errors import OutcomePiperStateError, OutcomePiperValidationError


class Clock:
    def __init__(self):
        self.now = 100.0
        self.auto = 0.0

    def __call__(self):
        self.now += self.auto
        return self.now

    def advance(self, delta=0.02):
        self.now += delta


class Receiver(FakeReceiver):
    stale = False

    def status(self):
        return self.arm.get_arm_status(), self.clock()

    def snapshot(self):
        result = super().snapshot()
        return replace(result, received_s=(self.clock() - (1 if self.stale else 0),) * 5)


@pytest.fixture
def session(tmp_path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    clock = Clock()
    robot._monotonic = clock
    robot._receiver_factory = Receiver
    robot._start_watchdog = lambda: None
    control = TeleopControl()
    robot.configure_teleoperation(control, xbox_config().hold_settings())
    robot.connect()
    yield robot, arm, control, clock
    robot.disconnect()


def tick(session, hold=False, neutral=True, values=None):
    robot, arm, control, clock = session
    obs = robot.get_observation()
    intent, epoch = control.observe(hold, neutral)
    action = OutcomePiperAction(obs if values is None else values, intent=intent, epoch=epoch)
    action.generated_monotonic_s = clock.now
    result = robot.send_action(action)
    return action, result


def settle(session):
    tick(session)
    session[3].advance()
    tick(session)
    assert session[2].hold_confirmed


def moves(arm):
    return [x for x in arm.calls if isinstance(x, tuple) and x[0] == "move_j"]


def test_startup_requires_release_neutral_and_new_press(session):
    robot, arm, control, clock = session
    tick(session, True, True)
    clock.advance()
    tick(session, True, True)
    assert control.state is TeleopState.WAITING
    tick(session, False, False)
    tick(session, True, False)
    tick(session, True, True)
    assert len(moves(arm)) == 1 and not arm.gripper.commands
    tick(session, False, True)
    tick(session, True, True)
    assert control.state is TeleopState.RUNNING and len(moves(arm)) == 2
    assert "electronic_emergency_stop" not in arm.calls


def test_release_captures_once_retains_grasp_and_resumes_from_feedback(session):
    robot, arm, control, clock = session
    settle(session)
    tick(session, True, True, valid_action(0.01, 0.035))
    arm.joints = [0.008] * 6
    arm.gripper.width = 0.02  # Measured opening must not replace the grasp command.
    tick(session, False, True)
    captured = robot._hold_target[:]
    count = len(moves(arm))
    gripper_commands = arm.gripper.commands[:]
    clock.advance()
    _, values = tick(session)
    assert control.state is TeleopState.PAUSED
    assert values["gripper.pos"] == 0.035 and arm.gripper.commands == gripper_commands
    assert robot.last_action_telemetry["result"] == "holding"
    assert robot.last_action_telemetry["commands"] == []
    for _ in range(4):
        clock.advance()
        tick(session)
    assert len(moves(arm)) == count and robot._hold_target == captured
    tick(session, True, True, {**valid_action(0.009, 0.035)})
    assert control.state is TeleopState.RUNNING
    assert arm.gripper.commands == gripper_commands
    assert "electronic_emergency_stop" not in arm.calls


def test_transient_drift_resets_stable_window_not_target_or_deadline(session):
    robot, arm, control, clock = session
    tick(session)
    deadline = robot._hold_deadline
    target = robot._hold_target[:]
    clock.advance(0.005)
    arm.joints = [0.011] * 6
    tick(session)
    assert not control.hold_confirmed and robot._hold_deadline == deadline
    clock.advance(0.005)
    arm.joints = [0.0] * 6
    tick(session)
    clock.advance(0.011)
    tick(session)
    assert control.hold_confirmed and robot._hold_target == target and len(moves(arm)) == 1
    arm.joints = [0.011] * 6
    clock.advance()
    tick(session)
    assert control.state is TeleopState.HOLD_REQUESTED
    assert robot._hold_target == target and len(moves(arm)) == 1


def test_hold_timeout_emergency_stops_once_without_disable_or_reset(session):
    robot, arm, control, clock = session
    arm.joints = [0.02] * 6
    tick(session)
    clock.advance(0.11)
    with pytest.raises(OutcomePiperStateError, match="hold confirmation timed out"):
        tick(session)
    robot._set_latch(PiperState.FAULT, "cleanup")
    robot.disconnect()
    assert arm.calls.count("electronic_emergency_stop") == 1
    assert "disable" not in arm.calls and "reset" not in arm.calls


def test_input_loss_holds_then_latches_without_later_duplicate_stop(session):
    robot, arm, control, clock = session
    settle(session)
    tick(session, True, True, valid_action(0.01, 0.035))
    clock.auto = 0.001
    robot.request_input_fault("selected Xbox disconnected")
    assert robot.state is PiperState.FAULT and robot._stop_outcome == "hold_confirmed"
    robot._set_latch(PiperState.FAULT, "outer handler")
    assert "electronic_emergency_stop" not in arm.calls
    robot.request_emergency_stop("B pressed")
    assert robot.state is PiperState.E_STOP
    assert arm.calls.count("electronic_emergency_stop") == 1


def test_pause_ticks_keep_watchdog_alive_but_stalled_loop_faults(session):
    robot, arm, control, clock = session
    settle(session)
    for _ in range(8):
        clock.advance(0.05)
        tick(session)
        assert robot._watchdog_check_locked()
    clock.advance(robot._safety.watchdog_timeout_s + 0.01)
    assert not robot._watchdog_check_locked()
    assert robot.state is PiperState.FAULT and robot._stop_outcome == "hold_confirmed"
    assert "electronic_emergency_stop" not in arm.calls


def test_stale_feedback_while_paused_takes_existing_emergency_path(session):
    robot, arm, control, clock = session
    settle(session)
    robot._receiver.stale = True
    assert not robot._watchdog_check_locked()
    assert robot.state is PiperState.FAULT
    assert arm.calls.count("electronic_emergency_stop") == 1


def test_delayed_action_cannot_cross_pause_or_rearm_epoch(session):
    robot, arm, control, clock = session
    settle(session)
    robot.get_observation()
    intent, epoch = control.observe(True, True)
    old = OutcomePiperAction(valid_action(0.04), intent=intent, epoch=epoch)
    old.generated_monotonic_s = clock.now
    control.observe(False, True)
    robot.send_action(old)
    count = len(moves(arm))
    clock.advance()
    tick(session)
    tick(session, True, True, valid_action(0.01))
    robot.get_observation()
    robot.send_action(old)
    assert len(moves(arm)) == count + 1
    assert robot.last_action_telemetry["result"] == "discarded"


def test_b_button_preempts_axis_parsing(session):
    from lerobot_robot_outcome_piper.input_safety import (
        motion_input_safety_scope,
        register_active_motion_session,
    )

    robot, arm, control, clock = session
    joystick = FakeJoystick()
    joystick.buttons[0] = 1
    joystick.get_axis = lambda _: (_ for _ in ()).throw(AssertionError("axes should not be read"))
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    with motion_input_safety_scope():
        register_active_motion_session(robot)
        xbox.connect()
        result = xbox.get_action()
    assert result["emergency_stop"] is True and robot.state is PiperState.E_STOP
    assert arm.calls.count("electronic_emergency_stop") == 1
    xbox.disconnect()


def test_raw_neutral_is_calculated_before_release_zeroing():
    joystick = FakeJoystick()
    joystick.buttons[1] = 0
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    xbox.connect()
    a = xbox.get_action()
    assert a["delta_z"] == 0 and a["neutral"] is False
    joystick.axes = [0, 0, 0, 0, -1, -1]
    assert xbox.get_action()["neutral"] is True
    xbox.disconnect()


def test_hold_configuration_is_verified_before_can_construction(tmp_path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    path = robot.config.hardware_acceptance_path
    r = json.loads(path.read_text())
    r["teleoperation_hold"]["verified"] = False
    path.write_text(json.dumps(r))
    with pytest.raises(OutcomePiperValidationError, match="hold acceptance"):
        robot.configure_teleoperation(TeleopControl(), xbox_config().hold_settings())
    assert not arm.calls


def test_selected_device_removal_routes_to_hold_not_generic_stop(session):
    from lerobot_robot_outcome_piper.errors import OutcomePiperInputDisconnected
    from lerobot_robot_outcome_piper.input_safety import (
        motion_input_safety_scope,
        register_active_motion_session,
    )

    robot, arm, control, clock = session
    settle(session)
    joystick = FakeJoystick()
    joystick.get_instance_id = lambda: 42
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    xbox.connect()
    events = [NS(instance_id=99)]
    xbox._pygame = NS(JOYDEVICEREMOVED=9, event=NS(pump=lambda: None, get=lambda _: events))
    with motion_input_safety_scope():
        register_active_motion_session(robot)
        xbox.get_action()  # Removing a different device is not this input fault.
        events[:] = [NS(instance_id=42)]
        with pytest.raises(OutcomePiperInputDisconnected):
            xbox.get_action()
    xbox._pygame = None
    xbox.disconnect()
    assert robot.state is PiperState.FAULT
    assert robot.stop_outcome == "hold_confirmed"
    assert "electronic_emergency_stop" not in arm.calls


@pytest.mark.parametrize("interruption", ["release", "disconnect", "B"])
def test_inflight_arm_dispatch_never_sends_late_gripper(session, interruption):
    robot, arm, control, clock = session
    settle(session)
    tick(session, True, True, valid_action(0.01, 0.035))
    original = arm.move_j
    entered, finish = threading.Event(), threading.Event()
    errors = []
    first = True

    def blocked_move(target):
        nonlocal first
        if first:
            first = False
            entered.set()
            assert finish.wait(2)
        original(target)

    arm.move_j = blocked_move
    robot.get_observation()
    action = OutcomePiperAction(valid_action(0.02, 0.038), intent="run", epoch=control.epoch)
    action.generated_monotonic_s = clock.now

    def send():
        try:
            robot.send_action(action)
        except OutcomePiperStateError as exc:
            errors.append(exc)

    worker = threading.Thread(target=send)
    worker.start()
    assert entered.wait(2)
    count = len(arm.gripper.commands)
    fault_worker = None
    if interruption == "release":
        control.observe(False, True)
    elif interruption == "disconnect":
        fault_worker = threading.Thread(
            target=robot.request_input_fault, args=("Xbox disconnected",)
        )
        clock.auto = 0.001
        fault_worker.start()
        assert robot._input_fault_requested.wait(2)
    else:
        fault_worker = threading.Thread(target=robot.request_emergency_stop, args=("B pressed",))
        fault_worker.start()
        assert robot._emergency_stop_requested.wait(2)
    finish.set()
    worker.join(2)
    if fault_worker:
        fault_worker.join(2)
        assert not fault_worker.is_alive()
    assert not worker.is_alive()
    assert len(arm.gripper.commands) == count
    assert robot._last_gripper_target == 0.035
    assert ("disable" not in arm.calls) and ("reset" not in arm.calls)
    assert arm.calls.count("electronic_emergency_stop") == (1 if interruption == "B" else 0)
    assert bool(errors) == (interruption == "B")


def test_failed_hold_and_failed_stop_preserve_unknown_outcome(session):
    robot, arm, control, clock = session
    settle(session)
    tick(session, True, True, valid_action(0.01))
    arm.move_j = lambda _: (_ for _ in ()).throw(OSError("CAN unavailable"))
    arm.electronic_emergency_stop = lambda: (_ for _ in ()).throw(OSError("stop send failed"))
    robot.request_input_fault("Xbox disconnected")
    assert robot.state is PiperState.FAULT
    assert robot.stop_outcome == "stop_unknown"
    assert "stop send failed" in robot.stop_error
    assert robot.last_action_telemetry["stop_outcome"] == "stop_unknown"
    assert robot.last_action_telemetry["hold_command"]["result"] == "failed"


def test_old_epoch_does_not_feed_watchdog_and_old_generation_faults(session):
    robot, arm, control, clock = session
    settle(session)
    old, _ = tick(session, True, True, valid_action(0.01))
    tick(session, False, True)
    previous = robot._last_control_tick_s
    clock.advance(0.02)
    robot.get_observation()
    robot.send_action(old)
    assert robot._last_control_tick_s == previous
    intent, epoch = control.observe(False, True)
    expired = OutcomePiperAction(valid_action(), intent=intent, epoch=epoch)
    expired.generated_monotonic_s = clock.now - robot.config.feedback_timeout_s - 1
    before = len(moves(arm))
    with pytest.raises(OutcomePiperStateError, match="control tick expired"):
        robot.send_action(expired)
    assert len(moves(arm)) == before
    assert robot.stop_outcome == "hold_confirmed"


def test_cached_feedback_cannot_confirm_continuous_hold(session):
    robot, arm, control, clock = session
    snapshot = robot._receiver.snapshot
    frozen = snapshot()
    robot._receiver.snapshot = lambda: frozen
    tick(session)
    clock.advance(0.02)
    tick(session)
    assert not control.hold_confirmed
    robot._receiver.snapshot = snapshot
    tick(session)
    assert control.hold_confirmed
