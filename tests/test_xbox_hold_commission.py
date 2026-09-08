"""One-joint real-input commissioning exercised with moving fake feedback."""

import math
from types import SimpleNamespace as NS
import pytest
from test_joint_commission import Clock, Arm, Receiver, Gripper, LIMITS
from piper_xbox_hold_commission import XboxHoldRun, InputLost, OperatorStop
from lerobot_robot_outcome_piper.teleop_control import TeleopState


def setup_case():
    clock, arm, report = Clock(), Arm(), {}
    arm.q = [0, 0, 0, 0, math.radians(2.932), 0]
    arm.flags = [True] * 6
    arm.ctrl = 1
    arm.goal = arm.q[:]
    arm.stops = 0

    def move(q):
        arm.moves.append(q[:])
        arm.goal = q[:]

    def stop():
        arm.stops += 1

    arm.move_j = move
    arm.electronic_emergency_stop = stop
    rx = Receiver(arm, clock)
    rx.gripper.foc_status.driver_enable_status = True
    snapshot = rx.snapshot
    last = [clock()]

    def moving_snapshot():
        delta = 0.04 * (clock() - last[0])
        last[0] = clock()
        arm.q = [q + max(-delta, min(delta, g - q)) for q, g in zip(arm.q, arm.goal)]
        return snapshot()

    rx.snapshot = moving_snapshot
    run = XboxHoldRun(
        arm,
        rx,
        LIMITS,
        report,
        goal=[0] * 6,
        clock=clock,
        sleep=clock.sleep,
        gripper=Gripper(rx, arm),
        gripper_target=0,
        speed_percent=1,
    )
    run.expected_start = arm.q[:]
    return run, arm, rx, clock


def release_reader(run, arm, *, fault=None):
    phase = [0]

    def read():
        neutral = True
        hold = False
        if phase[0] == 0 and run.control.hold_confirmed:
            phase[0] = 1
        if phase[0] == 1:
            hold = True
            if arm.q[4] < run.expected_start[4] - math.radians(0.2):
                if fault == "disconnect":
                    raise InputLost("Xbox disconnected")
                if fault == "B":
                    return NS(emergency=True, hold=True, neutral=False)
                phase[0] = 2
                hold = False
        elif phase[0] == 2 and run.control.state is TeleopState.PAUSED:
            phase[0] = 3
        elif phase[0] == 3:
            hold = True
        return NS(emergency=False, hold=hold, neutral=neutral)

    return read


def test_real_hold_primitive_reused_for_release_and_resume():
    run, arm, rx, clock = setup_case()
    run.run_xbox(release_reader(run, arm), confirm=lambda _: "")
    assert run.report["status"] == "release_resume_complete"
    assert run.report["released_before_arrival"]
    assert run.report["final_hold_confirmed"]
    commands = run.report["commands"]
    assert sum(c["name"] == "capture_hold_once" for c in commands) == 3
    assert sum(c["name"] == "return_zero_once" for c in commands) == 2
    assert sum(c["name"] == "set_gripper_width" for c in commands) == 1
    assert arm.enables == 0 and arm.stops == 0
    targets = [r["hold_target_rad"] for r in run.report["input_samples"] if r["state"] == "PAUSED"]
    assert targets and all(q == targets[0] for q in targets)


@pytest.mark.parametrize("fault", ["disconnect", "B"])
def test_input_fault_routes_hold_or_explicit_estop(fault):
    run, arm, rx, clock = setup_case()
    with pytest.raises((InputLost, OperatorStop)) as error:
        run.run_xbox(release_reader(run, arm, fault=fault), confirm=lambda _: "")
    run.stop_after_failure(error.value)
    assert arm.enables == 0
    if fault == "disconnect":
        assert arm.stops == 0 and run.report["stop_result"] == "hold_confirmed_then_fault"
    else:
        assert arm.stops == 1 and run.control.state is TeleopState.E_STOP


def test_input_fault_with_missing_feedback_records_stop_failure():
    run, arm, rx, clock = setup_case()
    with pytest.raises(InputLost) as error:
        run.run_xbox(release_reader(run, arm, fault="disconnect"), confirm=lambda _: "")
    rx.stale = True
    arm.electronic_emergency_stop = lambda: (_ for _ in ()).throw(OSError("CAN lost"))
    run.stop_after_failure(error.value)
    assert run.report["stop_result"] == "unknown" and "hold_error" in run.report


def test_no_release_is_not_reported_as_success():
    run, arm, rx, clock = setup_case()

    def read():
        return NS(emergency=False, neutral=True, hold=run.control.hold_confirmed)

    run.run_xbox(read, confirm=lambda _: "")
    assert run.report["status"] == "pause_not_demonstrated"
    assert run.report["final_hold_confirmed"] and arm.stops == 0


def test_changed_start_pose_rejected_without_sends():
    run, arm, rx, clock = setup_case()
    arm.q[4] += math.radians(1)
    with pytest.raises(RuntimeError, match="pose differs"):
        run.run_xbox(lambda: NS(emergency=False, hold=False, neutral=True), confirm=lambda _: "")
    assert not arm.moves and arm.enables == 0
