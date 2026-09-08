"""Exercise single-target commissioning with simulated feedback, never CAN."""

import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "infra/acceptance"))
from piper_joint_commission import STEP, TOLERANCE, JointRun, joint_waypoints, trace_transmissions  # noqa: E402

LIMITS = [
    (math.radians(a), math.radians(b))
    for a, b in [(-150, 150), (0, 180), (-170, 0), (-100, 100), (-70, 70), (-180, 180)]
]
START = list(map(math.radians, [-0.656, -1.477, 1.991, -3.425, 21.325, 4.708]))


class Clock:
    now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Arm:
    OPTIONS = NS(MOTION_MODE=NS(J="j"))

    def __init__(self):
        self.q, self.flags, self.ctrl = list(START), [False] * 6, 0
        self.moves, self.enables, self.stall = [], 0, False
        self.comm_error = False

    def has_comm_error(self):
        return self.comm_error

    def get_comm_error(self):
        return "simulated communication failure"

    def get_driver_states(self, i):
        return NS(
            msg=NS(foc_status=NS(driver_enable_status=self.flags[i - 1], driver_error_status=False))
        )

    def set_auto_set_motion_mode_enabled(self, value):
        assert value is False

    def set_joint_limits_enabled(self, value):
        assert value is False

    def set_speed_percent(self, value):
        assert value in (1, 5)
        self.speed = value

    def set_motion_mode(self, value):
        assert value == "j"
        self.ctrl = 1

    def enable(self):
        self.enables += 1
        self.flags = [True] * 6
        return False  # Cached API return is not the fresh enable acknowledgement.

    def move_j(self, goal):
        self.moves.append(list(goal))
        if all(self.flags) and not self.stall:
            self.q = list(goal)


class Receiver:
    DRIVER_IDS = tuple(range(0x261, 0x267))

    def __init__(self, arm, clock):
        self.arm, self.clock, self.condition = arm, clock, threading.RLock()
        self.received, self.stale = {}, False
        self.gripper = NS(
            mode="width", value=0.0, status_code=0, foc_status=NS(driver_enable_status=False)
        )

    def snapshot(self):
        t = self.clock() - (1 if self.stale else 0)
        self.received = dict.fromkeys(self.DRIVER_IDS, t)
        return NS(
            received_s=(t,) * 5,
            joints=NS(msg=self.arm.q),
            status=NS(
                msg=NS(
                    ctrl_mode=self.arm.ctrl,
                    mode_feedback=self.arm.ctrl,
                    arm_status=0,
                    err_code=0,
                    teach_status=0,
                    motion_status=0,
                    trajectory_num=0,
                )
            ),
            gripper=NS(msg=self.gripper),
        )


def setup():
    clock, arm, report = Clock(), Arm(), {}
    rx = Receiver(arm, clock)
    return (
        JointRun(arm, rx, LIMITS, report, goal=[0] * 6, clock=clock, sleep=clock.sleep),
        arm,
        rx,
        report,
    )


def test_waypoints_enter_legal_range_and_end_at_zero_under_five_degrees():
    previous = START
    for point in joint_waypoints(START, [0] * 6, LIMITS):
        assert all(abs(a - b) <= STEP - TOLERANCE + 1e-12 for a, b in zip(previous, point))
        assert all(lo <= q <= hi for q, (lo, hi) in zip(point, LIMITS))
        previous = point
    assert previous == [0] * 6


def test_zero_only_keeps_all_enabled_and_does_not_run_second_target():
    run, arm, _, report = setup()
    run.run(lambda _: "")
    assert arm.enables == 1 and all(arm.flags)
    assert arm.q == [0] * 6
    assert report["status"] == "target_feedback_confirmed_enabled"
    assert arm.moves == joint_waypoints(START, [0] * 6, LIMITS)
    commands = [c["name"] for c in report["commands"]]
    assert (
        commands.index("enable_all_joints")
        < commands.index("speed_percent")
        < commands.index("CAN_J_mode")
        < commands.index("joint_waypoint")
    )
    assert not any(
        "disable" in c["name"] and c["name"] not in ("disable_auto_mode", "disable_sdk_clipping")
        for c in report["commands"]
    )


@pytest.mark.parametrize("fault", ["cancel", "stale", "comm", "partial_enable", "outside_entry"])
def test_invalid_start_sends_no_motion_or_enable(fault):
    run, arm, rx, report = setup()
    if fault == "stale":
        rx.stale = True
    if fault == "comm":
        arm.comm_error = True
    if fault == "partial_enable":
        arm.flags[0] = True
    if fault == "outside_entry":
        arm.q[1] = math.radians(-8)
    with pytest.raises((RuntimeError, ValueError)):
        run.run(lambda _: "cancel" if fault == "cancel" else "")
    assert arm.moves == [] and arm.enables == 0


def test_stalled_motion_sends_no_later_waypoints_or_disable():
    run, arm, _, report = setup()
    # Use a legal start to allow a measured hold on failure.
    arm.q[1:3] = [0, 0]
    arm.stall = True
    with pytest.raises(RuntimeError, match="timed out"):
        run.run(lambda _: "")
    assert len(arm.moves) == 1
    run.hold_on_failure()
    assert arm.moves[-1] == arm.q and all(arm.flags)
    assert report["failure_hold"] == "command sent; physical stop not verified"


def test_stale_failure_cannot_send_a_guessed_hold():
    run, arm, rx, report = setup()
    report["commands"].append({"name": "enable_all_joints"})
    rx.stale = True
    with pytest.raises(RuntimeError, match="stale"):
        run.hold_on_failure()
    assert not arm.moves


def test_target_outside_zero_range_is_rejected():
    with pytest.raises(ValueError, match="target is outside"):
        joint_waypoints(START, [0] * 6, [(0.1, 1), *LIMITS[1:]])


def test_transmission_trace_preserves_packet_and_does_not_resend_on_error():
    calls, report = [], {}

    def original(frame):
        calls.append(frame)
        raise OSError("simulated send failure")

    comm = NS(send=original)
    trace_transmissions(comm, report, clock=lambda: 123)
    frame = NS(arbitration_id=0x151, data=bytes.fromhex("0101010000000000"))
    with pytest.raises(OSError):
        comm.send(frame)
    assert calls == [frame]
    assert report["tx_frames"] == [
        {"can_id": 0x151, "data_hex": "0101010000000000", "started_monotonic_s": 123}
    ]


def test_continue_enabled_zero_allows_bounded_transient_without_reenable():
    run, arm, rx, report = setup()
    arm.q = [0, 0, 0, 0, math.radians(16.002), 0]
    arm.flags, arm.ctrl = [True] * 6, 1
    original = rx.snapshot
    moves = arm.move_j
    pending = []

    def move(goal):
        before = list(arm.q)
        moves(goal)
        pending.append((before, list(goal)))

    arm.move_j = move

    def snapshot():
        if pending:
            before, goal = pending.pop(0)
            arm.q = list(before)
            arm.q[4] += math.radians(0.4)
            frame = original()
            arm.q = list(goal)
            return frame
        return original()

    rx.snapshot = snapshot
    run.run(lambda _: "")
    assert arm.enables == 0 and all(arm.flags) and arm.q == [0] * 6
    assert report["status"] == "target_feedback_confirmed_enabled"
    assert report["plan"]["initially_enabled"] is True


def test_excursion_over_five_degrees_stops_before_another_waypoint():
    run, arm, rx, report = setup()
    arm.q = [0, 0, 0, 0, math.radians(16.002), 0]
    arm.flags, arm.ctrl = [True] * 6, 1
    original = arm.move_j

    def move(goal):
        before = list(arm.q)
        original(goal)
        arm.q = before
        arm.q[4] += math.radians(5.1)

    arm.move_j = move
    with pytest.raises(RuntimeError, match="excursion exceeded"):
        run.run(lambda _: "")
    assert len(arm.moves) == 1 and arm.enables == 0


def test_requested_second_pose_keeps_enabled_and_respects_every_step():
    run, arm, rx, report = setup()
    start = [0.0] * 6
    arm.q = start[:]
    arm.flags, arm.ctrl = [True] * 6, 1
    run.goal = list(map(math.radians, [25, 45, -30, 60, -10, -34]))
    run.run(lambda _: "")
    assert arm.enables == 0 and all(arm.flags)
    assert arm.q == pytest.approx(run.goal)
    previous = start
    for point in arm.moves:
        assert all(abs(a - b) <= STEP for a, b in zip(previous, point))
        assert all(lo <= v <= hi for v, (lo, hi) in zip(point, LIMITS))
        previous = point
    assert report["status"] == "target_feedback_confirmed_enabled"


def test_rejects_original_wrong_sign_second_pose_before_any_commands():
    run, arm, rx, report = setup()
    run.goal = list(map(math.radians, [25, -45, 30, 60, -10, -34]))
    with pytest.raises(ValueError, match="target is outside"):
        run.run(lambda _: "")
    assert arm.enables == 0 and arm.moves == []


def test_step_reserves_both_endpoint_errors_without_relaxing_five_degree_limit():
    goal = list(map(math.radians, [25, 45, -30, 60, -10, -34]))
    previous = [0.0] * 6
    for target in joint_waypoints(previous, goal, LIMITS):
        for a, b in zip(previous, target):
            if a != b:
                direction = math.copysign(1, b - a)
                measured_start = a - direction * TOLERANCE
                measured_end = b + direction * TOLERANCE
                assert abs(measured_end - measured_start) <= STEP + 1e-12
        previous = target


def test_flange_result_units_and_missing_feedback():
    from piper_joint_commission import read_flange_result

    receiver = NS(condition=threading.RLock())
    arm = NS(
        get_flange_pose=lambda: NS(
            msg=[0.056127, 0, 0.233266, 0, math.radians(84.999), 0], timestamp=123
        )
    )
    result = read_flange_result(arm, receiver)
    assert result["xyz_mm"] == pytest.approx([56.127, 0, 233.266])
    assert result["rpy_deg"] == pytest.approx([0, 84.999, 0])
    assert result["sdk_timestamp_s"] == 123
    arm.get_flange_pose = lambda: None
    with pytest.raises(ValueError, match="unavailable"):
        read_flange_result(arm, receiver)


def test_z20_joint_target_from_zero_is_one_bounded_waypoint():
    goal = list(map(math.radians, [0, 2.257, -4.323, 0, 2.065, 0]))
    assert joint_waypoints([0] * 6, goal, LIMITS) == [goal]


def test_stability_window_restarts_after_transient_without_more_commands():
    run, arm, rx, report = setup()
    arm.flags, arm.ctrl = [True] * 6, 1
    arm.q = [0.0] * 6
    run.active = run.mode_required = True
    requested = run.clock()
    original = rx.snapshot

    def snapshot():
        elapsed = run.clock() - requested
        arm.q[2] = math.radians(0.107 if 0.1 <= elapsed < 0.2 else 0)
        return original()

    rx.snapshot = snapshot
    run.wait_stable(requested, [0.0] * 6)
    assert 0.5 <= run.clock() - requested < 0.6
    assert arm.moves == []


def test_stability_does_not_extend_original_deadline_or_accept_in_progress():
    run, arm, rx, report = setup()
    arm.flags, arm.ctrl = [True] * 6, 1
    arm.q = [0.0] * 6
    run.active = run.mode_required = True
    requested = run.clock() - 9.9
    original = rx.snapshot

    def snapshot():
        frame = original()
        frame.status.msg.motion_status = 1
        return frame

    rx.snapshot = snapshot
    with pytest.raises(RuntimeError, match="original waypoint deadline"):
        run.wait_stable(requested, [0.0] * 6)
    assert 10 <= run.clock() - requested < 10.02
    assert arm.moves == []


class Gripper:
    def __init__(self, rx, arm):
        self.rx, self.arm = rx, arm
        self.calls = []
        self.maximum = 0.07
        self.stuck = False

    def get_gripper_teaching_pendant_param(self, **kwargs):
        return NS(msg=NS(max_range_config=self.maximum))

    def move_gripper_m(self, width, force):
        self.calls.append((width, force, list(self.arm.q)))
        self.rx.gripper.foc_status.driver_enable_status = True
        self.rx.gripper.status_code = 0x40
        if not self.stuck:
            self.rx.gripper.value = width


def final_profile():
    run, arm, rx, report = setup()
    arm.q = list(map(math.radians, [0, 2.268, -4.216, 0, 2.08, 0]))
    arm.flags, arm.ctrl = [True] * 6, 1
    run.goal = list(map(math.radians, [-42.488, 132.772, -106.826, -4.781, 43.002, 66.067]))
    run.step, run.speed, run.via_zero = math.radians(15), 5, True
    run.gripper, run.gripper_target = Gripper(rx, arm), 0.065
    return run, arm, rx, report


def test_final_profile_zero_closed_then_target_open_with_one_confirmation():
    run, arm, rx, report = final_profile()
    confirmations = []
    run.run(lambda prompt: confirmations.append(prompt) or "")
    assert len(confirmations) == 1 and arm.enables == 0 and arm.speed == 5
    assert run.gripper.calls == [(0.0, 1.0, [0.0] * 6), (0.065, 1.0, run.goal)]
    assert arm.q == pytest.approx(run.goal) and all(arm.flags)
    assert rx.gripper.value == 0.065 and rx.gripper.foc_status.driver_enable_status
    assert len(report["completed_stages"]) == 2 and report["gripper_open_confirmed"]
    previous = [0.0] * 6
    for goal in report["plan"]["stage_waypoints_deg"][1]:
        assert all(abs(a - b) <= 15 for a, b in zip(previous, goal))
        previous = goal


def test_gripper_range_failure_prevents_both_stages():
    run, arm, rx, report = final_profile()
    run.gripper.maximum = 0.06
    with pytest.raises(ValueError, match="exceeds reported range"):
        run.run(lambda _: "")
    assert not arm.moves and not run.gripper.calls


def test_gripper_open_failure_does_not_complete_target_stage():
    run, arm, rx, report = final_profile()
    run.gripper.stuck = True
    with pytest.raises(RuntimeError, match="timed out"):
        run.run(lambda _: "")
    assert arm.q == run.goal and len(report["completed_stages"]) == 1
    assert len(run.gripper.calls) == 2
    run.hold_on_failure()
    assert arm.q == run.goal and all(arm.flags)


def test_gripper_fault_does_not_block_healthy_joint_hold():
    run, arm, rx, report = final_profile()
    arm.q = [0.0] * 6
    run.active = run.mode_required = True
    report["commands"].append({"name": "joint_waypoint"})
    rx.gripper.status_code = 0x20
    run.hold_on_failure()
    assert arm.moves == [[0.0] * 6]
    with pytest.raises(RuntimeError, match="gripper reports a fault"):
        run.read(True)


def test_official_gripper_packet_opens_without_calibration_or_reset():
    from lerobot_robot_outcome_piper.sdk import create_piper

    arm = create_piper("not-opened", "v189")
    gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
    packets = []
    gripper._send_msg = lambda message: packets.append(gripper._parser.pack(message))
    gripper.move_gripper_m(0.065, 1.0)
    assert len(packets) == 1 and packets[0].arbitration_id == 0x159
    assert bytes(packets[0].data) == bytes.fromhex("0000fde803e80100")


def test_normal_enabled_standby_resumes_without_reenable_or_reset():
    run, arm, rx, report = final_profile()
    arm.ctrl = 0
    run.run(lambda _: "")
    assert arm.enables == 0 and all(arm.flags)
    assert report["status"] == "target_feedback_confirmed_enabled"
    assert report["completed_stages"][0] == [0.0] * 6
    assert report["gripper_open_confirmed"]


def test_enabled_standby_emergency_stop_still_blocks_all_commands():
    run, arm, rx, report = final_profile()
    arm.ctrl = 0
    original = rx.snapshot

    def snapshot():
        frame = original()
        frame.status.msg.arm_status = 1
        return frame

    rx.snapshot = snapshot
    with pytest.raises(RuntimeError, match="controller fault"):
        run.run(lambda _: "")
    assert report["commands"] == [] and arm.enables == 0 and not arm.moves


def test_roundtrip_returns_to_zero_closed_after_open_target():
    run, arm, rx, report = final_profile()
    arm.q = run.goal[:]
    run.return_zero = True
    run.run(lambda _: "")
    assert report["completed_stages"] == [
        [0.0] * 6,
        pytest.approx(list(map(math.degrees, run.goal))),
        [0.0] * 6,
    ]
    assert arm.q == [0.0] * 6 and all(arm.flags) and arm.enables == 0
    assert run.gripper.calls == [
        (0.0, 1.0, [0.0] * 6),
        (0.065, 1.0, run.goal),
        (0.0, 1.0, [0.0] * 6),
    ]
    assert rx.gripper.value == 0.0 and rx.gripper.foc_status.driver_enable_status


def test_roundtrip_fault_does_not_attempt_return_trajectory():
    run, arm, rx, report = final_profile()
    arm.q = [0.0] * 6
    run.return_zero = True
    original = run.gripper.move_gripper_m

    def open_and_stall(width, force):
        original(width, force)
        arm.stall = True

    run.gripper.move_gripper_m = open_and_stall
    with pytest.raises(RuntimeError, match="timed out"):
        run.run(lambda _: "")
    assert report["completed_stages"] == [[0.0] * 6]
    assert len(arm.moves) == 2


def test_zero_close_failure_blocks_departure_from_zero():
    run, arm, rx, report = final_profile()
    run.return_zero = True
    rx.gripper.value = 0.065
    rx.gripper.foc_status.driver_enable_status = True
    run.gripper.stuck = True
    with pytest.raises(RuntimeError, match="timed out"):
        run.run(lambda _: "")
    assert arm.q == [0.0] * 6
    assert report["completed_stages"] == []
    assert run.gripper.calls == [(0.0, 1.0, [0.0] * 6)]


def test_zero_width_command_does_not_calibrate_the_gripper():
    from lerobot_robot_outcome_piper.sdk import create_piper

    arm = create_piper("not-opened", "v189")
    gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
    packets = []
    gripper._send_msg = lambda message: packets.append(gripper._parser.pack(message))
    gripper.move_gripper_m(0.0, 1.0)
    assert len(packets) == 1 and packets[0].arbitration_id == 0x159
    assert bytes(packets[0].data) == bytes.fromhex("0000000003e80100")
