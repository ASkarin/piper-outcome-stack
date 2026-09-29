# ruff: noqa: F811
"""Camera quality must not masquerade as a fault of healthy holding control."""

from types import SimpleNamespace as NS
import threading
import time
import numpy as np
import pytest
from test_xbox_pause import session, settle, tick, moves  # noqa: F401, F811
from lerobot_robot_outcome_piper.camera import CameraTelemetry, read_new_frame
from lerobot_robot_outcome_piper.errors import OutcomePiperCameraError, OutcomePiperStateError
from lerobot_robot_outcome_piper.robot import PiperState
from lerobot_robot_outcome_piper.teleop_control import TeleopState


def camera(read):
    return NS(is_connected=True, disconnect=lambda: None, read_with_metadata=read)


def fail_read(timeout):
    raise OutcomePiperCameraError("camera missing fresh frame")


@pytest.mark.parametrize("phase", ["review", "saving", "finalizing"])
def test_holding_phases_do_not_read_or_require_camera(session, phase):
    robot, arm, control, clock = session
    settle(session)
    robot.cameras = {"d435": camera(lambda _: pytest.fail("camera read during holding-only phase"))}
    robot.cameras["d435"].is_connected = False
    control.recording_phase = phase
    robot.get_observation()
    assert robot.last_observation_telemetry["quality"] == "control_only"
    assert robot.last_observation_telemetry["cameras"] == {}
    assert robot.last_depth_frames == {}
    assert "electronic_emergency_stop" not in arm.calls
    assert robot.is_connected
    # The next preparation phase must check the camera again.
    control.recording_phase = "preparing"
    with pytest.raises(OutcomePiperCameraError, match="disconnected"):
        robot.get_observation()
    assert robot.stop_outcome == "hold_confirmed"
    assert "electronic_emergency_stop" not in arm.calls


def test_recording_camera_fault_holds_and_latches_without_new_gripper_command(session):
    robot, arm, control, clock = session
    settle(session)
    tick(session, True, True)
    gripper = list(arm.gripper.commands)
    control.recording_phase = "recording"
    robot.cameras = {"d435": camera(fail_read)}
    clock.auto = 0.001
    with pytest.raises(OutcomePiperCameraError):
        robot.get_observation()
    assert robot.state is PiperState.FAULT
    assert robot.stop_outcome == "hold_confirmed"
    assert arm.gripper.commands == gripper
    assert "electronic_emergency_stop" not in arm.calls
    count = len(moves(arm))
    robot.request_input_fault("cleanup after camera error")
    assert len(moves(arm)) == count
    assert "disable" not in arm.calls and "reset" not in arm.calls


def test_camera_fault_and_bad_robot_feedback_still_stop(session):
    robot, arm, control, clock = session
    settle(session)
    robot.cameras = {"d435": camera(fail_read)}
    robot._receiver.stale = True
    with pytest.raises(OutcomePiperCameraError):
        robot.get_observation()
    assert "electronic_emergency_stop" in arm.calls
    assert robot.stop_outcome == "electronic_stop_sent_unverified"


def test_b_preempts_camera_fault_hold_confirmation(session):
    robot, arm, control, clock = session
    settle(session)
    tick(session, True, True)
    robot.cameras = {"d435": camera(fail_read)}
    robot.emergency_stop_poll = lambda: True
    with pytest.raises(OutcomePiperCameraError):
        robot.get_observation()
    assert robot.state is PiperState.E_STOP
    assert arm.calls.count("electronic_emergency_stop") == 1


def waiting_camera():
    now = time.monotonic()
    return NS(
        metadata_condition=threading.Condition(),
        capture_error=None,
        latest_metadata={"color": CameraTelemetry(1, 1.0, "hardware", now - 1, now - 1)},
        latest_frames={"color": np.zeros((2, 3, 3), np.uint8)},
        consumed_frame_number=None,
        max_frame_age_s=0.05,
        thread=NS(is_alive=lambda: True),
    )


def test_stale_cached_frame_is_skipped_and_service_runs_until_fresh():
    cam = waiting_camera()
    calls = []

    def service():
        calls.append(1)
        if len(calls) == 2:
            now = time.monotonic()
            cam.latest_metadata = {"color": CameraTelemetry(2, 2.0, "hardware", now, now)}

    cam.wait_service = service
    _, meta = read_new_frame(cam, 0.1)
    assert meta["color"].frame_number == 2
    assert cam.last_read_diagnostics["stale_frames_skipped"] == 1
    assert len(calls) == 2


def test_camera_wait_has_one_deadline_and_can_be_interrupted():
    cam = waiting_camera()
    calls = []
    cam.wait_service = lambda: calls.append(time.monotonic())
    started = time.monotonic()
    with pytest.raises(OutcomePiperCameraError, match="stale frame sequence"):
        read_new_frame(cam, 0.025)
    assert len(calls) > 1
    assert time.monotonic() - started < 0.15
    cam.wait_service = lambda: (_ for _ in ()).throw(OutcomePiperStateError("B stop"))
    with pytest.raises(OutcomePiperStateError, match="B stop"):
        read_new_frame(cam, 0.1)


def test_camera_wait_releases_command_lock_and_services_lb(session):
    robot, arm, control, clock = session
    settle(session)
    tick(session, True, True)
    owned = []

    def read(timeout):
        acquired = []

        def check():
            with robot._command_lock:
                acquired.append(True)

        t = threading.Thread(target=check)
        t.start()
        t.join(0.2)
        owned.extend(acquired)
        robot._service_camera_wait()
        raise OutcomePiperCameraError("test ends after servicing input")

    robot.camera_input_poll = lambda: {"hold": False, "emergency_stop": False}
    robot.cameras = {"d435": camera(read)}
    clock.auto = 0.001
    with pytest.raises(OutcomePiperCameraError):
        robot.get_observation()
    assert owned == [True]
    assert robot.stop_outcome == "hold_confirmed"
    assert control.state is TeleopState.FAULT
    assert "electronic_emergency_stop" not in arm.calls


def test_camera_fault_hold_timeout_escalates(session):
    robot, arm, control, clock = session
    settle(session)
    robot.cameras = {"d435": camera(fail_read)}
    update = robot._update_hold_locked

    def drift():
        arm.joints[0] += 0.02
        update()

    robot._update_hold_locked = drift
    clock.auto = 0.002
    with pytest.raises(OutcomePiperCameraError):
        robot.get_observation()
    assert robot.stop_outcome == "electronic_stop_sent_unverified"
    assert arm.calls.count("electronic_emergency_stop") == 1


def test_control_only_observation_cannot_enter_dataset(tmp_path):
    from lerobot_robot_outcome_piper.recording import TelemetryDataset

    robot = NS(
        config=NS(),
        last_observation_telemetry={"quality": "control_only"},
        last_action_telemetry={},
    )
    audit = TelemetryDataset(NS(root=tmp_path), robot)
    try:
        with pytest.raises(RuntimeError, match="control-only observations"):
            audit.add_frame({})
        assert audit.pending == []
    finally:
        audit.close()
