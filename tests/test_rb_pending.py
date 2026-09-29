import pytest
from lerobot_robot_outcome_piper.teleop_control import TeleopControl, TeleopMode, TeleopState
from lerobot_robot_outcome_piper.processor import OutcomePiperAction
from test_xbox_pause import session as session, settle, moves
from test_plugin import valid_action


def select_during_hold(c):
    c.confirm_hold()
    c.observe(False, True, False)
    c.observe(True, True, False)
    intent, epoch = c.observe(False, True, True)
    assert intent == "hold"
    assert c.pending_mode is TeleopMode.ORIENTATION
    assert c.mode is TeleopMode.TRANSLATION and not c.hold_confirmed
    return epoch


def test_select_same_tick_as_lb_release_and_finish_only_on_fresh_neutral_input():
    c = TeleopControl()
    epoch = select_during_hold(c)
    c.observe(False, True, True)  # Held RB does not flip twice.
    assert c.pending_mode is TeleopMode.ORIENTATION
    c.confirm_hold()
    assert c.mode is TeleopMode.TRANSLATION
    intent, ready_epoch = c.observe(False, True, False)
    assert intent == "hold" and ready_epoch > epoch
    assert c.mode is TeleopMode.ORIENTATION and c.pending_mode is None
    assert c.mode_event["ready"] is True
    assert c.observe(True, True, False)[0] == "run"


@pytest.mark.parametrize(
    "change,reason",
    [
        (dict(hold=True), "LB pressed before mode ready"),
        (dict(neutral=False), "inputs not neutral"),
        (dict(home=True), "conflicting button"),
        (dict(work=True), "conflicting button"),
        (dict(translation_switch=True), "conflicting button"),
    ],
)
def test_pending_selection_cancelled_without_later_motion(change, reason):
    c = TeleopControl()
    select_during_hold(c)
    args = dict(hold=False, neutral=True, mode_switch=False)
    args.update(change)
    assert c.observe(**args)[0] == "hold"
    assert c.pending_mode is None and c.mode is TeleopMode.TRANSLATION
    assert c.mode_event["reason"] == reason
    c.confirm_hold()
    assert c.observe(**args)[0] != "run"
    assert c.mode is TeleopMode.TRANSLATION


@pytest.mark.parametrize("phase", ["review", "saving", "finalizing"])
def test_phase_change_cancels_selection(phase):
    c = TeleopControl()
    select_during_hold(c)
    c.recording_phase = phase
    c.observe(False, True)
    assert c.pending_mode is None and c.mode is TeleopMode.TRANSLATION


def test_rb_again_cancels_and_startup_press_is_not_queued():
    c = TeleopControl()
    c.observe(False, True, True)
    assert c.pending_mode is None
    c.confirm_hold()
    c.observe(False, True, True)
    assert c.mode is TeleopMode.TRANSLATION
    c.observe(False, True)
    c.hold_confirmed = False
    c.observe(False, True, True)
    c.observe(False, True)
    c.observe(False, True, True)
    assert c.pending_mode is None
    assert c.mode_event["reason"] == "RB pressed again"


def test_old_mode_does_not_start_on_simultaneous_rb_lb_from_pause():
    c = TeleopControl()
    c.confirm_hold()
    c.observe(False, True)
    assert c.observe(True, True, True)[0] != "run"
    assert c.mode is TeleopMode.TRANSLATION


@pytest.mark.parametrize("operation", ["B", "fault", "reset", "enable"])
def test_lifecycle_invalidates_pending(operation):
    from test_processor import processor

    p = processor()
    c = p.control
    c.request_hold()
    c.observe(False, True, False)
    c.observe(False, True, True)
    assert c.pending_mode is not None
    if operation in ("B", "fault"):
        c.stop(operation == "B")
    elif operation == "reset":
        p.reset()
    else:
        c.prepare_enable()
    assert c.pending_mode is None and c.mode is TeleopMode.TRANSLATION


def send(session, lb=False, rb=False):
    robot, arm, c, clock = session
    obs = robot.get_observation()
    intent, epoch = c.observe(lb, True, rb)
    a = OutcomePiperAction(obs, intent=intent, epoch=epoch)
    a.generated_monotonic_s = clock.now
    robot.send_action(a)
    return a


def test_hold_timer_target_and_gripper_are_not_restarted_by_selection(session):
    robot, arm, c, clock = session
    settle(session)
    send(session, lb=True)
    send(session, lb=False)
    original_window = robot._hold_window
    original_time = original_window.after_s
    commands = len(moves(arm))
    grip = list(arm.gripper.commands)
    send(session, rb=True)
    assert c.pending_mode is TeleopMode.ORIENTATION
    assert robot._hold_window is original_window
    assert original_window.after_s == original_time
    assert len(moves(arm)) == commands and arm.gripper.commands == grip
    assert robot.last_action_telemetry["pending_mode"] == "ORIENTATION"
    clock.advance(0.02)
    send(session)
    # Sending the hold action confirms it; the following sampled input commits mode.
    send(session)
    assert c.mode is TeleopMode.ORIENTATION
    assert robot._hold_window is original_window
    assert len(moves(arm)) == commands and arm.gripper.commands == grip
    assert c.state in (TeleopState.WAITING, TeleopState.PAUSED)


def test_stale_action_and_b_cannot_dispatch_after_selection(session):
    robot, arm, c, clock = session
    settle(session)
    send(session, lb=True)
    old = OutcomePiperAction(valid_action(0.01, 0.035), intent="run", epoch=c.epoch)
    old.generated_monotonic_s = clock.now
    send(session, lb=False, rb=True)
    count = len(moves(arm))
    robot.send_action(old)
    assert len(moves(arm)) == count
    robot.request_emergency_stop("B while mode pending")
    assert c.pending_mode is None
    with pytest.raises(RuntimeError):
        robot.send_action(old)
    assert len(moves(arm)) == count


def test_starting_episode_does_not_carry_pending_mode_selection():
    c = TeleopControl()
    c.recording_phase = "preparing"
    select_during_hold(c)
    c.confirm_hold()
    c.recording_phase = "recording"
    assert c.observe(False, True)[0] == "hold"
    assert c.pending_mode is None and c.mode is TeleopMode.TRANSLATION
    assert c.mode_event["reason"] == "recording phase changed"
