"""Synthetic receive events and clocks; no hardware timing values are approved here."""

import copy
import threading
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip("lerobot")
from test_plugin import FakeArm, NOW, make_robot, valid_action, d435_config
from lerobot_robot_outcome_piper.camera import CameraTelemetry, validate_frame, read_new_frame
from lerobot_robot_outcome_piper.timing import FeedbackReceiver, CaptureTiming
from lerobot_robot_outcome_piper.robot import PiperState


def test_receiver_binds_monotonic_time_to_atomic_official_parse_result():
    arm = FakeArm()
    clock = [10.0]
    comm = NS()

    def parse(packet):
        if packet.arbitration_id == 0x2A5:
            arm.joints[:2] = packet.data[:2]

    comm.callback = parse
    comm.get_callback = lambda: comm.callback
    comm.set_callback = lambda cb: setattr(comm, "callback", cb)
    arm.get_context = lambda: NS(get_comm=lambda: comm)
    arm.get_fps = lambda: 200.0
    rx = FeedbackReceiver(arm, arm.gripper, lambda: clock[0])
    assert not rx.wait_ready(0)
    for ident in rx.IDS:
        comm.callback(
            NS(
                arbitration_id=ident,
                data=bytes([1, 2, 0, 0, 0, 0, 0, 0]),
                is_extended_id=False,
                is_error_frame=False,
            )
        )
        clock[0] += 0.001
    assert rx.wait_ready(0)
    first = rx.snapshot()
    assert first.joints.msg[:2] == [1, 2]
    assert first.received_s[0] == 10.0
    arm.joints[0] = 99
    arm._parser.joint_12.timestamp = NOW - 10000  # UTC is provenance only.
    assert first.joints.msg[0] == 1
    assert rx.snapshot().received_s == first.received_s
    clock[0] = 9.0
    comm.callback(
        NS(arbitration_id=0x2A1, data=bytes(8), is_extended_id=False, is_error_frame=False)
    )
    with pytest.raises(RuntimeError, match="backwards"):
        rx.snapshot()


def test_receiver_parser_failure_is_visible_without_polling_old_cache():
    arm = FakeArm()

    def broken(packet):
        raise ValueError("decode failed")

    comm = NS(get_callback=lambda: broken, set_callback=lambda cb: None)
    arm.get_context = lambda: NS(get_comm=lambda: comm)
    rx = FeedbackReceiver(arm, arm.gripper, lambda: 10.0)
    rx.receive(NS(arbitration_id=0x2A5))
    with pytest.raises(RuntimeError, match="decode failed"):
        rx.wait_ready(0)


def test_initial_wait_accepts_delayed_frames_without_reissuing_query(tmp_path):
    robot, arm, _ = make_robot(tmp_path)
    receiver = robot._receiver_factory(arm, arm.gripper, lambda: 100.0)
    waits = []

    def ready(timeout):
        waits.append(timeout)
        return len(waits) > 1

    receiver.wait_ready = ready
    robot._receiver_factory = lambda *args: receiver
    try:
        robot.connect()
        assert len(waits) == 2
        assert len([c for c in arm.calls if isinstance(c, tuple) and c[0] == "get_firmware"]) == 1
        assert "enable" not in arm.calls
    finally:
        robot.disconnect()


def test_initial_timeout_does_not_enable_or_retry(tmp_path):
    robot, arm, _ = make_robot(tmp_path)
    receiver = robot._receiver_factory(arm, arm.gripper, lambda: 100.0)
    receiver.wait_ready = lambda timeout: False
    robot._receiver_factory = lambda *args: receiver
    with pytest.raises(RuntimeError, match="initial complete feedback timed out"):
        robot.connect()
    assert arm.calls.count("connect") == arm.calls.count("disconnect") == 1
    assert "enable" not in arm.calls


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"frame_number": 1}, "duplicate"),
        ({"device_timestamp_ms": 1.0}, "did not advance"),
        ({"timestamp_domain": "system_time"}, "domain"),
        ({"received_monotonic_s": 0.5}, "backwards"),
        ({"device_timestamp_ms": float("nan")}, "invalid"),
    ],
)
def test_device_metadata_rejects_invalid_transitions(changes, message):
    previous = CameraTelemetry(1, 1.0, "hardware_clock", 1.0, 1.01)
    current = CameraTelemetry(2, 2.0, "hardware_clock", 2.0, 2.01)
    with pytest.raises(RuntimeError, match=message):
        validate_frame(previous, replace(current, **changes))


def test_camera_consumption_never_reuses_a_frame():
    camera = NS(
        metadata_condition=threading.Condition(),
        capture_error=None,
        latest_metadata=CameraTelemetry(1, 10.0, "hardware_clock", 1.0, 1.0),
        consumed_frame_number=None,
        latest_color_frame=np.zeros((2, 3, 3), dtype=np.uint8),
        thread=NS(is_alive=lambda: True),
    )
    image, metadata = read_new_frame(camera, 0)
    camera.latest_color_frame[:] = 99
    assert image.max() == 0
    assert metadata.frame_number == 1
    with pytest.raises(TimeoutError, match="no new D435"):
        read_new_frame(camera, 0)


def test_fresh_feedback_cannot_hide_expired_policy_observation(tmp_path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    try:
        robot.connect()
        old = copy.deepcopy(robot.last_observation_telemetry)
        robot._monotonic = lambda: 100.5
        fresh = replace(robot._receiver.snapshot(), received_s=(100.5,) * 5)
        robot._receiver.snapshot = lambda: fresh
        with pytest.raises(RuntimeError, match="observation expired"):
            robot.send_action(valid_action())
        assert robot.last_observation_telemetry == old
        assert not any(isinstance(c, tuple) and c[0] == "move_j" for c in arm.calls)
        assert robot.state is PiperState.FAULT
    finally:
        robot.disconnect()


@pytest.mark.parametrize(
    "camera_t, message", [(99.0, "stale"), (99.85, "skew"), (100.1, "invalid")]
)
def test_camera_state_quality_gates(tmp_path, camera_t, message):
    robot, arm, _ = make_robot(tmp_path)
    robot.config = replace(robot.config, capture_timing=CaptureTiming(0.2, 0.1, 0.05, 0.2))
    try:
        robot.connect()
        robot.cameras = {
            "d435": NS(
                is_connected=True,
                disconnect=lambda: None,
                read_with_metadata=lambda timeout: (
                    np.zeros((2, 3, 3), dtype=np.uint8),
                    CameraTelemetry(1, 1.0, "hardware_clock", camera_t, camera_t),
                ),
            )
        }
        with pytest.raises(RuntimeError, match=message):
            robot.get_observation()
    finally:
        robot.disconnect()


def test_motion_camera_requires_explicit_timing(tmp_path):
    robot, _, _ = make_robot(tmp_path, mode="motion")
    with pytest.raises(ValueError, match="capture_timing"):
        replace(robot.config, cameras={"d435": d435_config()})


def test_realsense_publication_preserves_frame_and_stops_on_duplicate():
    from lerobot_robot_outcome_piper.realsense import TimedRealSenseCamera

    camera = TimedRealSenseCamera.__new__(TimedRealSenseCamera)
    camera.stop_event = threading.Event()
    camera.new_frame_event = threading.Event()
    camera.frame_lock = threading.Lock()
    camera.metadata_condition = threading.Condition(camera.frame_lock)
    camera.latest_metadata = None
    camera.capture_error = None
    camera._postprocess_image = lambda image: image
    raw = NS(
        get_data=lambda: np.full((2, 3, 3), 42, dtype=np.uint8),
        get_frame_number=lambda: 1,
        get_timestamp=lambda: 10.0,
        get_frame_timestamp_domain=lambda: "hardware_clock",
    )
    camera._read_from_hardware = lambda: NS(get_color_frame=lambda: raw)
    camera._read_loop()
    assert camera.latest_metadata.frame_number == 1
    assert camera.latest_color_frame.min() == 42
    assert "duplicate" in str(camera.capture_error)


def test_pinned_sdk_parsers_keep_radians_and_total_gripper_metres():
    pytest.importorskip("pyAgxArm")
    import math
    import struct
    import can
    from lerobot_robot_outcome_piper.sdk import create_piper

    arm = create_piper("unused-no-can-open", "v189")
    gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

    def parse(packet):
        arm._parser.parse_packet(packet)
        gripper._parser.parse_packet(packet)

    comm = NS(get_callback=lambda: parse, set_callback=lambda cb: None)
    arm.get_context().comm = comm
    rx = FeedbackReceiver(arm, gripper, lambda: 10.0)
    for ident in (0x2A5, 0x2A6, 0x2A7):
        rx.receive(
            can.Message(
                arbitration_id=ident, is_extended_id=False, data=struct.pack(">ii", 90000, -90000)
            )
        )
    rx.receive(can.Message(arbitration_id=0x2A1, is_extended_id=False, data=bytes(8)))
    rx.receive(
        can.Message(
            arbitration_id=0x2A8, is_extended_id=False, data=struct.pack(">ihBB", 30000, 0, 0, 0)
        )
    )
    snap = rx.snapshot()
    assert snap.joints.msg == pytest.approx([math.pi / 2, -math.pi / 2] * 3)
    assert snap.gripper.msg.value == pytest.approx(0.03)
    assert snap.received_s == (10.0,) * 5
    arm.get_context().comm = None
    assert arm.is_connected() is False
