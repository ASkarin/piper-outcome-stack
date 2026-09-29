from __future__ import annotations

import importlib.metadata
import itertools
import json
import math
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("lerobot")

PLUGIN_SRC = Path(__file__).parents[1] / "packages" / "lerobot_robot_outcome_piper" / "src"
sys.path.insert(0, str(PLUGIN_SRC))

from lerobot.types import TransitionKey  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.robots import make_robot_from_config  # noqa: E402
from lerobot.teleoperators import make_teleoperator_from_config  # noqa: E402
from lerobot_robot_outcome_piper.config import (  # noqa: E402
    OutcomePiperConfig,
    OutcomePiperXboxConfig,
)
from lerobot_robot_outcome_piper.errors import (  # noqa: E402
    OutcomePiperStateError,
    OutcomePiperValidationError,
)
from lerobot_robot_outcome_piper.input_safety import (  # noqa: E402
    motion_input_safety_scope,
)
from lerobot_robot_outcome_piper.processor import (  # noqa: E402
    OutcomePiperAction,
    OutcomePiperXboxProcessor,
)
from lerobot_robot_outcome_piper.timing import CaptureTiming, ReceivedFeedback  # noqa: E402
from lerobot_robot_outcome_piper.camera import CameraTelemetry  # noqa: E402
from lerobot_robot_outcome_piper.robot import OutcomePiper, PiperState  # noqa: E402
from lerobot_robot_outcome_piper.safety import ACTION_KEYS, JOINT_KEYS, MotionSafety  # noqa: E402
from lerobot_robot_outcome_piper.teleoperator import OutcomePiperXbox  # noqa: E402
from lerobot_robot_outcome_piper import cli, workflows  # noqa: E402


NOW = 1_800_000_000.0


def d435_config(**kwargs):
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    values = dict(serial_number_or_name="123456789012", width=640, height=480, fps=30)
    values.update(kwargs)
    return RealSenseCameraConfig(**values)


@pytest.mark.parametrize(
    "cameras, message",
    [
        (
            {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)},
            "RealSense backend",
        ),
        ({"front": d435_config(serial_number_or_name="")}, "select a RealSense"),
        ({"front": d435_config(width=None, height=None, fps=None)}, "Specifying"),
        ({"front": d435_config(use_depth=True), "front.depth": d435_config()}, "collision"),
    ],
)
def test_camera_contract_rejects_invalid_payload_configuration(tmp_path, cameras, message):
    with pytest.raises(ValueError, match=message):
        replace(config(tmp_path), cameras=cameras)


@pytest.mark.parametrize(
    "cameras",
    [
        {"front": d435_config(serial_number_or_name="Intel RealSense D435", use_depth=True)},
        {"left": d435_config(), "right": d435_config(serial_number_or_name="another-camera")},
        {"range": d435_config(use_rgb=False, use_depth=True)},
        {"front": d435_config(color_mode="bgr", use_depth=True)},
    ],
)
def test_camera_names_count_and_depth_are_configuration_choices(tmp_path, cameras):
    cfg = replace(config(tmp_path), cameras=cameras)
    assert cfg.cameras == cameras


def test_single_d435_observation_preserves_seven_action_fields(tmp_path):
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera = SimpleNamespace(
        is_connected=True,
        connect=lambda: None,
        disconnect=lambda: None,
        read_with_metadata=lambda timeout: (
            {"color": frame},
            {"color": CameraTelemetry(1, 1.0, "hardware_clock", 100.0, 100.0)},
        ),
    )
    robot = OutcomePiper(
        replace(config(tmp_path), cameras={"d435": d435_config()}),
        piper_factory=lambda *_: FakeArm(),
        camera_factory=lambda _: {"d435": camera},
        monotonic=lambda: 100.0,
        wall_time=lambda: NOW,
        receiver_factory=FakeReceiver,
    )
    try:
        connect_for_test(robot)
        observation = robot.get_observation()
        assert set(observation) == {*ACTION_KEYS, "d435"}
        assert observation["d435"] is frame
        assert robot.observation_features["d435"] == frame.shape
        assert tuple(robot.action_features) == ACTION_KEYS
    finally:
        robot.disconnect()


class FakeReceiver:
    """Explicit receive events simulated in host monotonic time, separate from UTC."""

    def __init__(self, arm, gripper, clock):
        self.arm, self.gripper, self.clock = arm, gripper, clock

    def wait_ready(self, timeout):
        return True

    def status(self):
        status = self.arm.get_arm_status()
        return status, None if status is None else 100.0 + (status.timestamp - NOW)

    def driver_states(self):
        return tuple(
            (
                SimpleNamespace(
                    msg=SimpleNamespace(
                        foc_status=SimpleNamespace(
                            driver_enable_status=self.arm.enabled, driver_error_status=False
                        )
                    )
                ),
                self.clock(),
            )
            for _ in range(6)
        )

    def snapshot(self):
        frames = tuple(
            getattr(self.arm._parser, name, None) for name in ("joint_12", "joint_34", "joint_56")
        )
        if any(f is None for f in frames):
            raise RuntimeError("incomplete joint feedback groups")
        status, gripper = self.arm.get_arm_status(), self.gripper.get_gripper_status()
        return ReceivedFeedback(
            self.arm.get_joint_angles(),
            gripper,
            status,
            frames,
            tuple(self.arm._ctx.fps.get_fps(f.msg_type) for f in frames),
            tuple(
                100.0 + (t - NOW)
                for t in (*[f.timestamp for f in frames], status.timestamp, gripper.timestamp)
            ),
        )


class FakeFps:
    def __init__(self, values=(11.0, 12.0, 13.0)):
        self.values = dict(zip((12, 34, 56), values, strict=True))

    def get_fps(self, message_type):
        return self.values[message_type]


class FakeGripper:
    def __init__(self, arm):
        self.arm = arm
        self.width = 0.03
        self.enabled = False
        self.status_code = 0
        self.timestamp = NOW - 0.02
        self.hz = 50.0
        self.commands = []

    def get_gripper_status(self):
        if self.arm.fail_feedback:
            raise OSError("feedback failed")
        return SimpleNamespace(
            msg=SimpleNamespace(
                value=self.width,
                mode="width",
                status_code=self.status_code,
                foc_status=SimpleNamespace(driver_enable_status=self.enabled),
            ),
            timestamp=self.timestamp,
            hz=self.hz,
        )

    def disable_gripper(self):
        self.commands.append("disable")
        self.enabled = False

    def move_gripper_m(self, width, *, force):
        self.commands.append((width, force))
        if self.arm.fail_command:
            raise OSError("gripper failed")


class FakeArm:
    class OPTIONS:
        class EFFECTOR:
            AGX_GRIPPER = "official-gripper"

        class MOTION_MODE:
            J = "J"

    def __init__(self):
        self.calls = []
        self.connected = False
        self.comm_error = False
        self.fail_feedback = False
        self.fail_command = False
        self.enable_result = True
        self.enabled = False
        self.joints = [0.0] * 6
        self.status = 0
        self.error_code = 0
        self.teach_status = 0
        self.ctrl_mode = 1
        self.mode_feedback = 1
        self.firmware = {
            "hardware_version": "H-V1.2-1",
            "motor_ratio_and_batch": "10",
            "node_type": "ARM_MC",
            "software_version": "S-V1.8-9",
            "production_date": "260813",
            "node_number": "15",
        }
        self.status_timestamp = NOW - 0.03
        self.status_hz = 40.0
        self._parser = SimpleNamespace(
            joint_12=SimpleNamespace(timestamp=NOW - 0.01, msg_type=12),
            joint_34=SimpleNamespace(timestamp=NOW - 0.02, msg_type=34),
            joint_56=SimpleNamespace(timestamp=NOW - 0.03, msg_type=56),
        )
        self._ctx = SimpleNamespace(fps=FakeFps())
        self.gripper = FakeGripper(self)
        self.move_started = threading.Event()
        self.release_move = threading.Event()
        self.block_move = False
        self.stop_started = threading.Event()
        self.release_stop = threading.Event()
        self.block_stop = False
        self.fail_stop = False

    def connect(self):
        self.calls.append("connect")
        self.connected = True

    def disconnect(self):
        self.calls.append("disconnect")
        self.connected = False

    def is_connected(self):
        return self.connected

    def has_comm_error(self):
        return self.comm_error

    def get_comm_error(self):
        return "fake CAN error"

    def init_effector(self, effector):
        self.calls.append(("init_effector", effector))
        return self.gripper

    def get_firmware(self, *, timeout, min_interval):
        self.calls.append(("get_firmware", timeout, min_interval))
        return self.firmware

    def set_auto_set_motion_mode_enabled(self, value):
        self.calls.append(("auto_mode", value))

    def set_joint_limits_enabled(self, value):
        self.calls.append(("sdk_limits", value))

    def set_motion_mode(self, value):
        self.calls.append(("motion_mode", value))
        self.status_timestamp = NOW

    def set_speed_percent(self, value):
        self.calls.append(("speed_percent", value))

    def enable(self):
        self.calls.append("enable")
        self.enabled = True
        return self.enable_result

    def disable(self):
        self.calls.append("disable")
        self.enabled = False

    def electronic_emergency_stop(self):
        self.calls.append("electronic_emergency_stop")
        self.stop_started.set()
        if self.fail_stop:
            raise OSError("electronic stop failed")
        if self.block_stop:
            if not self.release_stop.wait(timeout=2):
                raise TimeoutError("test did not release electronic emergency stop")

    def get_joint_angles(self):
        if self.fail_feedback:
            raise OSError("feedback failed")
        return SimpleNamespace(msg=self.joints)

    def get_arm_status(self):
        if self.fail_feedback:
            raise OSError("feedback failed")
        return SimpleNamespace(
            msg=SimpleNamespace(
                ctrl_mode=self.ctrl_mode,
                teach_status=self.teach_status,
                arm_status=self.status,
                mode_feedback=self.mode_feedback,
                err_code=self.error_code,
            ),
            timestamp=self.status_timestamp,
            hz=self.status_hz,
        )

    def move_j(self, joints):
        self.calls.append(("move_j", joints))
        if self.block_move:
            self.move_started.set()
            if not self.release_move.wait(timeout=2):
                raise TimeoutError("test did not release move_j")
        if self.fail_command:
            raise OSError("move_j failed")


class FakeCamera:
    def __init__(self, *, connected=True, fail_probe=False):
        self.connected = connected
        self.fail_probe = fail_probe

    @property
    def is_connected(self):
        if self.fail_probe:
            self.fail_probe = False
            raise OSError("camera connection probe failed")
        return self.connected

    def disconnect(self):
        self.connected = False


def safety() -> MotionSafety:
    return MotionSafety(
        joint_lower=(-1.0,) * 6,
        joint_upper=(1.0,) * 6,
        max_joint_step=(0.1,) * 6,
        gripper_lower=0.0,
        gripper_upper=0.08,
        max_gripper_step=0.01,
        workspace_lower=(-1.0, -1.0, -1.0),
        workspace_upper=(1.0, 1.0, 1.0),
        feedback_timeout_s=0.2,
        watchdog_timeout_s=10.0,
        motion_speed_percent=5,
        gripper_force_n=0.5,
        stop_strategy="electronic_emergency_stop",
    )


def write_safety(tmp_path: Path) -> Path:
    """Create synthetic runtime limits, without acceptance attestations."""
    safety_path = tmp_path / "safety.json"
    safety_path.write_text(
        json.dumps(
            {
                "schema_version": "outcome-piper-safety-v1",
                "joint_lower_rad": [-1.0] * 6,
                "joint_upper_rad": [1.0] * 6,
                "max_joint_step_rad": [0.1] * 6,
                "gripper_lower_m": 0.0,
                "gripper_upper_m": 0.08,
                "max_gripper_step_m": 0.01,
                "workspace_lower_m": [-1.0] * 3,
                "workspace_upper_m": [1.0] * 3,
                "feedback_timeout_s": 0.2,
                "watchdog_timeout_s": 10.0,
                "motion_speed_percent": 5,
                "gripper_force_n": 0.5,
                "stop_strategy": "electronic_emergency_stop",
            }
        ),
        encoding="utf-8",
    )
    return safety_path


def config(tmp_path: Path, *, mode="read_only"):
    kwargs = {}
    if mode == "motion":
        kwargs = {"safety_path": write_safety(tmp_path)}
    return OutcomePiperConfig(
        can_interface="can-test",
        firmware="v189",
        feedback_timeout_s=0.2,
        execution_mode=mode,
        calibration_dir=tmp_path / "calibration",
        **kwargs,
    )


def make_robot(tmp_path: Path, *, mode="read_only"):
    arm = FakeArm()
    factory_calls = []

    def factory(can_interface, firmware):
        factory_calls.append((can_interface, firmware))
        return arm

    robot = OutcomePiper(
        config(tmp_path, mode=mode),
        piper_factory=factory,
        camera_factory=lambda _: {},
        monotonic=lambda: 100.0,
        wall_time=lambda: NOW,
        receiver_factory=FakeReceiver,
    )
    return robot, arm, factory_calls


def valid_action(value=0.0, gripper=0.03):
    return {**dict.fromkeys(JOINT_KEYS, value), "gripper.pos": gripper}


def test_import_and_construction_have_no_can_io(tmp_path: Path):
    robot, arm, factory_calls = make_robot(tmp_path)
    assert factory_calls == []
    assert arm.calls == []
    assert robot.state is PiperState.DISCONNECTED


def test_plugin_distribution_discovery_and_lerobot_factories(tmp_path: Path):
    distribution = importlib.metadata.distribution("lerobot_robot_outcome_piper")
    assert distribution.version == "0.1.0"
    robot = make_robot_from_config(config(tmp_path))
    teleop = make_teleoperator_from_config(xbox_config())
    assert isinstance(robot, OutcomePiper)
    assert isinstance(teleop, OutcomePiperXbox)


def test_read_only_connect_has_zero_motion_configuration_and_enable(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path)
    connect_for_test(robot)
    assert robot.state is PiperState.CONNECTED
    assert "enable" not in arm.calls
    assert not any(
        isinstance(call, tuple) and call[0] in {"auto_mode", "sdk_limits", "motion_mode"}
        for call in arm.calls
    )
    with pytest.raises(OutcomePiperStateError):
        robot.send_action(valid_action())


def test_invalid_runtime_limits_reject_before_sdk_construction(tmp_path):
    robot, _, factory_calls = make_robot(tmp_path, mode="motion")
    path = robot.config.safety_path
    values = json.loads(path.read_text())
    values["max_joint_step_rad"] = [0] * 6
    path.write_text(json.dumps(values))
    with pytest.raises(OutcomePiperValidationError, match="joint step limits"):
        connect_for_test(robot)
    assert not factory_calls


def test_motion_connect_configures_one_mode_and_enables(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    assert robot.state is PiperState.ACTIVE
    assert ("motion_mode", "J") in arm.calls
    assert ("speed_percent", 5) in arm.calls
    assert "enable" in arm.calls


@pytest.mark.parametrize("software_version", ["S-V1.6-2", "S-V1.6-3"])
def test_motion_firmware_must_match_pinned_kinematics(tmp_path: Path, software_version):
    robot, arm, factory_calls = make_robot(tmp_path, mode="motion")
    robot.config = replace(robot.config, firmware="default")
    arm.firmware["software_version"] = software_version
    try:
        if software_version == "S-V1.6-2":
            with pytest.raises(OutcomePiperStateError, match="pinned PiPER MDH model"):
                connect_for_test(robot)
            assert factory_calls
            assert "enable" not in arm.calls
        else:
            connect_for_test(robot)
            assert robot.state is PiperState.ACTIVE
            assert "enable" in arm.calls
    finally:
        robot.disconnect()


def test_read_only_can_inspect_firmware_before_kinematics_change(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path)
    robot.config = replace(robot.config, firmware="default")
    arm.firmware["software_version"] = "S-V1.6-2"
    try:
        connect_for_test(robot)
        assert robot.get_observation() == valid_action()
        assert "enable" not in arm.calls
        assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)
    finally:
        robot.disconnect()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hardware_version", None, "hardware_version"),
        ("node_type", "PIPER_X", "PiPER arm controller"),
        ("software_version", "S-V1.8-8", "software_version"),
    ],
)
def test_live_firmware_validation_still_precedes_enable(tmp_path: Path, field, value, message):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    arm.firmware[field] = value
    with pytest.raises(OutcomePiperStateError, match=message):
        connect_for_test(robot)
    assert "enable" not in arm.calls


@pytest.mark.parametrize(("ctrl_mode", "mode_feedback"), [(0, 1), (1, 0)])
def test_motion_mode_feedback_must_confirm_can_move_j_before_enable(
    tmp_path: Path, ctrl_mode, mode_feedback
):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    arm.ctrl_mode = ctrl_mode
    arm.mode_feedback = mode_feedback
    robot.connect()
    robot._monotonic = itertools.count(100.0, 0.02).__next__
    with pytest.raises(OutcomePiperStateError, match="motion-mode feedback confirmation timed out"):
        robot.enable()
    assert "enable" not in arm.calls
    assert arm.calls.count(("motion_mode", "J")) == 1


def test_mode_confirmation_waits_for_fresh_matching_status_before_enable(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    original_get_status = arm.get_arm_status
    old_matching = original_get_status()
    new_transition = original_get_status()
    new_transition.timestamp = NOW
    new_transition.msg.mode_feedback = 0
    new_matching = original_get_status()
    new_matching.timestamp = NOW
    pending = [None, old_matching, new_transition, new_matching]

    def get_status():
        if ("motion_mode", "J") not in arm.calls:
            return original_get_status()
        if pending:
            assert "enable" not in arm.calls
            result = pending.pop(0)
            if result is new_matching or result is new_transition:
                result.timestamp = NOW + (robot._monotonic() - 100.0)
            return result
        return original_get_status()

    arm.get_arm_status = get_status
    robot._monotonic = itertools.count(100.0, 0.001).__next__
    try:
        connect_for_test(robot)
        assert not pending
        assert robot.state is PiperState.ACTIVE
        assert arm.calls.count(("motion_mode", "J")) == 1
        assert arm.calls.count("enable") == 1
    finally:
        robot.disconnect()


@pytest.mark.parametrize("missing", [False, True])
def test_stale_or_missing_mode_confirmation_never_enables(tmp_path: Path, missing):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    old = arm.get_arm_status()
    arm.get_arm_status = lambda: None if missing and ("motion_mode", "J") in arm.calls else old
    robot.connect()
    robot._monotonic = itertools.count(100.0, 0.02).__next__
    with pytest.raises(OutcomePiperStateError, match="confirmation timed out"):
        robot.enable()
    assert robot.state is PiperState.FAULT
    assert arm.calls.count(("motion_mode", "J")) == 1
    assert "enable" not in arm.calls
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)
    assert arm.gripper.commands == []


@pytest.mark.parametrize("timestamp", [0.0, math.nan, NOW + 0.01])
def test_invalid_mode_timestamp_fails_before_enable(tmp_path: Path, timestamp):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    status = arm.get_arm_status()
    status.timestamp = timestamp
    arm.get_arm_status = lambda: status
    with pytest.raises(OutcomePiperStateError, match="timestamp"):
        connect_for_test(robot)
    assert "enable" not in arm.calls


def test_mode_confirmation_does_not_accept_a_result_after_its_deadline(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    clock = [100.0]
    robot._monotonic = lambda: clock[0]
    original_get_status = arm.get_arm_status

    def late_status():
        if ("motion_mode", "J") in arm.calls:
            clock[0] += 0.3
        return original_get_status()

    robot.connect()
    arm.get_arm_status = late_status
    with pytest.raises(OutcomePiperStateError, match="confirmation timed out"):
        robot.enable()
    assert "enable" not in arm.calls


@pytest.mark.parametrize(
    ("status", "error_code", "expected_state"),
    [(1, 0, PiperState.E_STOP), (5, 0, PiperState.FAULT), (0, 0x0100, PiperState.FAULT)],
)
def test_controller_fault_aborts_mode_confirmation(tmp_path, status, error_code, expected_state):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    arm.status = status
    arm.error_code = error_code
    with pytest.raises(OutcomePiperStateError, match="controller"):
        connect_for_test(robot)
    assert robot.state is expected_state
    assert "enable" not in arm.calls
    assert ("speed_percent", 5) not in arm.calls
    assert arm.calls.count("electronic_emergency_stop") == 0


def test_stop_request_interrupts_mode_confirmation(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")

    def stop_while_waiting():
        robot._emergency_stop_cause = "operator stop during connection"
        robot._emergency_stop_requested.set()
        return None

    arm.get_arm_status = stop_while_waiting
    with pytest.raises(OutcomePiperStateError, match="operator stop during connection"):
        connect_for_test(robot)
    assert robot.state is PiperState.E_STOP
    assert "enable" not in arm.calls


def test_wall_clock_rollback_does_not_change_receive_age(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot._wall_time = lambda: NOW - 1000.0
    try:
        connect_for_test(robot)
        robot.send_action(valid_action())
        assert robot.state is PiperState.ACTIVE
    finally:
        robot.disconnect()


@pytest.mark.parametrize(("ctrl_mode", "mode_feedback"), [(2, 1), (7, 1), (1, 6)])
def test_motion_session_mode_change_stops_before_next_action(tmp_path, ctrl_mode, mode_feedback):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    try:
        connect_for_test(robot)
        arm.ctrl_mode = ctrl_mode
        arm.mode_feedback = mode_feedback
        with pytest.raises(OutcomePiperStateError, match="left CAN joint position-velocity mode"):
            robot.send_action(valid_action())
        assert robot.state is PiperState.FAULT
        assert arm.calls.count("electronic_emergency_stop") == 1
        assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)
        assert arm.gripper.commands == []
    finally:
        robot.disconnect()


def test_mode_change_after_confirmation_is_rejected_before_enable(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    original_confirm = robot._confirm_motion_mode_locked

    def confirm(requested):
        original_confirm(requested)
        arm.ctrl_mode = 2

    robot._confirm_motion_mode_locked = confirm
    with pytest.raises(OutcomePiperStateError, match="left CAN joint position-velocity mode"):
        connect_for_test(robot)
    assert "enable" not in arm.calls


def test_read_only_observation_does_not_require_motion_mode(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path)
    arm.ctrl_mode = 2
    arm.mode_feedback = 0
    try:
        connect_for_test(robot)
        assert robot.get_observation() == valid_action()
        assert "enable" not in arm.calls
        assert "electronic_emergency_stop" not in arm.calls
    finally:
        robot.disconnect()


@pytest.mark.parametrize("failed_call", ["auto_mode", "sdk_limits", "speed_percent", "motion_mode"])
def test_motion_configure_checks_each_sdk_step_immediately(tmp_path: Path, failed_call):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    original_has_comm_error = arm.has_comm_error

    def has_comm_error():
        return any(isinstance(call, tuple) and call[0] == failed_call for call in arm.calls) or (
            original_has_comm_error()
        )

    arm.has_comm_error = has_comm_error
    with pytest.raises(OutcomePiperStateError, match="latched FAULT"):
        connect_for_test(robot)
    configured = [call[0] for call in arm.calls if isinstance(call, tuple)]
    expected = {
        "auto_mode": ["init_effector", "get_firmware", "auto_mode"],
        "sdk_limits": ["init_effector", "get_firmware", "auto_mode", "sdk_limits"],
        "speed_percent": [
            "init_effector",
            "get_firmware",
            "auto_mode",
            "sdk_limits",
            "speed_percent",
        ],
        "motion_mode": [
            "init_effector",
            "get_firmware",
            "auto_mode",
            "sdk_limits",
            "speed_percent",
            "motion_mode",
        ],
    }
    assert configured == expected[failed_call]
    assert "enable" not in arm.calls


@pytest.mark.parametrize(
    "action, message",
    [
        ({key: 0.0 for key in JOINT_KEYS}, "keys mismatch"),
        ({**valid_action(), "extra": 0.0}, "keys mismatch"),
        ({**valid_action(), "joint_1.pos": math.nan}, "finite"),
        ({**valid_action(), "joint_1.pos": math.inf}, "finite"),
        ({**valid_action(), "joint_1.pos": 1.1}, "outside frozen limits"),
        ({**valid_action(), "joint_1.pos": 0.11}, "step limit"),
    ],
)
def test_action_schema_and_limits_fail_without_move(tmp_path: Path, action, message):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    with pytest.raises(OutcomePiperValidationError, match=message):
        robot.send_action(action)
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)


def test_direct_joint_action_cannot_bypass_frozen_workspace(tmp_path: Path):
    kinematics = pytest.importorskip("pyAgxArm.utiles.mdh_kinematics")
    mdh = list(kinematics.get_mdh("piper"))
    initial_pose = kinematics.fk_from_mdh(mdh, [0.0] * 6)
    target_joints = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
    target_pose = kinematics.fk_from_mdh(mdh, target_joints)
    assert target_pose[1] > initial_pose[1]

    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    assert robot._safety is not None
    robot._safety = replace(
        robot._safety,
        workspace_upper=(1.0, (initial_pose[1] + target_pose[1]) / 2, 1.0),
    )
    action = valid_action()
    action["joint_1.pos"] = target_joints[0]

    with pytest.raises(OutcomePiperValidationError, match="outside the frozen workspace"):
        robot.send_action(action)

    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)
    assert arm.gripper.commands == []


def test_feedback_uses_three_groups_and_separate_status_gripper_telemetry(tmp_path: Path):
    robot, _, _ = make_robot(tmp_path)
    connect_for_test(robot)
    telemetry = robot.last_feedback_telemetry
    assert telemetry is not None
    assert telemetry.joint_group_timestamps_s == (NOW - 0.01, NOW - 0.02, NOW - 0.03)
    assert telemetry.joint_group_hz == (11.0, 12.0, 13.0)
    assert telemetry.arm_status_timestamp_s == NOW - 0.03
    assert telemetry.arm_status_hz == 40.0
    assert telemetry.gripper_timestamp_s == NOW - 0.02
    assert telemetry.gripper_hz == 50.0
    assert telemetry.ctrl_mode == 1
    assert telemetry.mode_feedback == 1


def test_send_action_uses_only_frozen_speed_and_gripper_force(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    result = robot.send_action(valid_action())
    assert result == valid_action()
    assert ("speed_percent", 5) in arm.calls
    assert arm.gripper.commands == [(0.03, 0.5)]


@pytest.mark.parametrize("timestamp", [NOW - 0.21, NOW + 0.01])
def test_stale_or_future_feedback_latches_fault(tmp_path: Path, timestamp):
    robot, arm, _ = make_robot(tmp_path)
    arm._parser.joint_12.timestamp = timestamp
    with pytest.raises(OutcomePiperStateError):
        connect_for_test(robot)
    assert robot.state is PiperState.FAULT


def test_sdk_feedback_failure_latches_and_disconnect_cannot_clear_session(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path)
    connect_for_test(robot)
    arm.fail_feedback = True
    with pytest.raises(OutcomePiperStateError, match="latched FAULT"):
        robot.get_observation()
    robot.disconnect()
    assert robot.state is PiperState.FAULT
    with pytest.raises(OutcomePiperStateError):
        connect_for_test(robot)


@pytest.mark.parametrize("malformation", ["gripper_status", "frame_frequency"])
def test_feedback_parse_failure_stops_and_terminally_latches_session(tmp_path: Path, malformation):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    if malformation == "gripper_status":
        arm.gripper.status_code = None
    else:
        arm._ctx.fps.get_fps = lambda _: (_ for _ in ()).throw(OSError("fps read failed"))

    with pytest.raises(OutcomePiperStateError, match="latched FAULT"):
        robot.get_observation()

    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.FAULT
    robot.disconnect()
    with pytest.raises(OutcomePiperStateError, match="terminally latched FAULT"):
        connect_for_test(robot)


def test_active_arm_disconnect_stops_and_terminally_latches_session(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    arm.connected = False

    with pytest.raises(OutcomePiperStateError, match="connection lost: arm"):
        robot.get_observation()

    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.FAULT
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected
    with pytest.raises(OutcomePiperStateError, match="terminally latched FAULT"):
        connect_for_test(robot)


def test_active_camera_disconnect_stops_and_terminally_latches_session(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    robot.cameras["d435"] = FakeCamera(connected=False)

    with pytest.raises(OutcomePiperStateError, match="camera 'd435'"):
        robot.get_observation()

    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.FAULT
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected
    with pytest.raises(OutcomePiperStateError, match="terminally latched FAULT"):
        connect_for_test(robot)


@pytest.mark.parametrize("probe_target", ["arm", "camera"])
def test_active_connection_probe_error_stops_and_latches_session(tmp_path: Path, probe_target):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    if probe_target == "arm":
        arm.is_connected = lambda: (_ for _ in ()).throw(OSError("arm probe failed"))
    else:
        robot.cameras["d435"] = FakeCamera(fail_probe=True)

    with pytest.raises(OutcomePiperStateError, match="latched FAULT"):
        robot.get_observation()

    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.FAULT
    robot.disconnect()


def test_disconnect_never_homes_resets_disables_or_stops(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    robot.disconnect()
    assert not any(
        (isinstance(call, str) and call in {"home", "reset", "disable"})
        or (isinstance(call, tuple) and call[0] in {"home", "reset", "disable"})
        for call in arm.calls
    )


def test_disconnect_is_serialized_after_inflight_action(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    arm.block_move = True
    action_errors = []
    disconnect_errors = []

    action_thread = threading.Thread(
        target=lambda: _capture_error(action_errors, robot.send_action, valid_action())
    )
    action_thread.start()
    assert arm.move_started.wait(timeout=1)
    disconnect_thread = threading.Thread(
        target=lambda: _capture_error(disconnect_errors, robot.disconnect)
    )
    disconnect_thread.start()
    assert disconnect_thread.is_alive()
    arm.release_move.set()
    action_thread.join(timeout=2)
    disconnect_thread.join(timeout=2)
    assert not action_thread.is_alive()
    assert not disconnect_thread.is_alive()
    assert action_errors == []
    assert disconnect_errors == []
    assert arm.calls.index("disconnect") > next(
        index for index, call in enumerate(arm.calls) if call == ("move_j", [0.0] * 6)
    )


def test_expired_inflight_action_does_not_send_gripper(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    arm.block_move = True
    action_errors = []
    action_thread = threading.Thread(
        target=lambda: _capture_error(action_errors, robot.send_action, valid_action())
    )
    action_thread.start()
    assert arm.move_started.wait(timeout=1)
    robot._monotonic = lambda: 111.0
    assert not robot._watchdog_stop.wait(0.1)
    arm.release_move.set()
    action_thread.join(timeout=2)
    assert len(action_errors) == 1
    assert "observation expired" in str(action_errors[0])
    assert robot.state is PiperState.FAULT
    assert arm.gripper.commands == []
    robot.disconnect()
    assert "electronic_emergency_stop" in arm.calls


def test_watchdog_stop_failure_is_exposed_to_the_control_loop(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    arm.fail_stop = True
    robot._monotonic = lambda: 111.0
    assert arm.stop_started.wait(timeout=1)

    with pytest.raises(OutcomePiperStateError, match="stop action failed: OSError"):
        robot.get_observation()

    assert robot.state is PiperState.FAULT
    assert robot.stop_error == "OSError: electronic stop failed"
    robot.disconnect()


def test_disconnect_waits_for_watchdog_stop_before_releasing_sdk(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    arm.block_stop = True
    robot._monotonic = lambda: 111.0
    assert arm.stop_started.wait(timeout=1)
    disconnect_errors = []
    disconnect_thread = threading.Thread(
        target=lambda: _capture_error(disconnect_errors, robot.disconnect)
    )
    disconnect_thread.start()
    assert disconnect_thread.is_alive()
    arm.release_stop.set()
    disconnect_thread.join(timeout=2)
    assert disconnect_errors == []
    assert robot.state is PiperState.FAULT
    assert arm.calls.index("electronic_emergency_stop") < arm.calls.index("disconnect")


def test_input_stop_during_inflight_move_prevents_gripper_and_latches_e_stop(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    with motion_input_safety_scope():
        connect_for_test(robot)
        arm.block_move = True
        action_errors = []
        stop_errors = []
        action_thread = threading.Thread(
            target=lambda: _capture_error(action_errors, robot.send_action, valid_action())
        )
        action_thread.start()
        assert arm.move_started.wait(timeout=1)
        stop_thread = threading.Thread(
            target=lambda: _capture_error(
                stop_errors, robot.request_emergency_stop, "Xbox input failed"
            )
        )
        stop_thread.start()
        assert robot._emergency_stop_requested.wait(timeout=1)
        assert stop_thread.is_alive()
        arm.release_move.set()
        action_thread.join(timeout=2)
        stop_thread.join(timeout=2)

    assert len(action_errors) == 1
    assert isinstance(action_errors[0], OutcomePiperStateError)
    assert stop_errors == []
    assert arm.gripper.commands == []
    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.E_STOP


def test_input_stop_wins_over_concurrent_move_failure_and_session_cannot_recover(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    with motion_input_safety_scope():
        connect_for_test(robot)
        arm.block_move = True
        arm.fail_command = True
        action_errors = []
        stop_errors = []
        action_thread = threading.Thread(
            target=lambda: _capture_error(action_errors, robot.send_action, valid_action())
        )
        action_thread.start()
        assert arm.move_started.wait(timeout=1)
        stop_thread = threading.Thread(
            target=lambda: _capture_error(
                stop_errors, robot.request_emergency_stop, "Xbox input failed"
            )
        )
        stop_thread.start()
        assert robot._emergency_stop_requested.wait(timeout=1)
        arm.release_move.set()
        action_thread.join(timeout=2)
        stop_thread.join(timeout=2)

    assert len(action_errors) == 1
    assert isinstance(action_errors[0], OutcomePiperStateError)
    assert stop_errors == []
    assert arm.gripper.commands == []
    assert robot.state is PiperState.E_STOP
    assert robot.latched_cause == "Xbox input failed"
    robot.disconnect()
    assert robot.state is PiperState.E_STOP
    with pytest.raises(OutcomePiperStateError, match="terminally latched E_STOP"):
        connect_for_test(robot)


def _capture_error(target, operation, *args):
    try:
        operation(*args)
    except Exception as exc:
        target.append(exc)


class FakeJoystick:
    def __init__(self, guid="measured-guid"):
        self.guid = guid
        self.axes = [0.0, 0.05, -0.5, 0.25, -1.0, 1.0]
        self.buttons = [0, 1, 0, 0, 0, 0]
        self.initialized = True

    def get_init(self):
        return self.initialized

    def get_guid(self):
        return self.guid

    def get_numaxes(self):
        return len(self.axes)

    def get_numbuttons(self):
        return len(self.buttons)

    def get_axis(self, index):
        return self.axes[index]

    def get_button(self, index):
        return self.buttons[index]

    def quit(self):
        self.initialized = False


def xbox_config(**overrides):
    values = {
        "device_guid": "measured-guid",
        "axis_x": 0,
        "axis_y": 1,
        "axis_z": 2,
        "axis_yaw": 3,
        "axis_left_trigger": 4,
        "axis_right_trigger": 5,
        "hold_button": 1,
        "emergency_stop_button": 0,
        "mode_switch_button": 2,
        "translation_switch_button": 5,
        "home_button": 3,
        "hold_joint_tolerance_rad": 0.01,
        "hold_stable_time_s": 0.01,
        "hold_timeout_s": 0.1,
        "deadzone": 0.1,
        "control_hz": 20,
        "xyz_step_m": 0.01,
        "rotation_step_rad": 0.02,
        "gripper_step_m": 0.004,
        "axis_signs": (1, -1, 1, -1),
        "trigger_rest_values": (-1.0, -1.0),
        "trigger_pressed_values": (1.0, 1.0),
        "ik_max_nfev": 20,
        "ik_timeout_s": 0.1,
        "ik_residual_tolerance": 0.001,
        "ik_min_singular_value": 0.001,
    }
    values.update(overrides)
    return OutcomePiperXboxConfig(**values)


def test_xbox_mapping_deadzone_triggers_hold_and_disconnect():
    joystick = FakeJoystick()
    xbox = OutcomePiperXbox(
        xbox_config(), joystick_factory=lambda guid: joystick if guid == "measured-guid" else None
    )
    xbox.connect()
    action = xbox.get_action()
    assert action == {
        "stick_x": 0.0,
        "stick_y": 0.0,
        "stick_z": -0.5,
        "stick_yaw": -0.25,
        "left_trigger": 0.0,
        "right_trigger": 1.0,
        "hold": True,
        "neutral": False,
        "emergency_stop": False,
        "mode_switch": False,
        "translation_switch": False,
        "home": False,
        "work": False,
    }
    xbox.disconnect()
    with pytest.raises(OutcomePiperStateError, match="disconnected"):
        xbox.get_action()


def test_unbound_xbox_disconnect_only_fails_fast():
    joystick = FakeJoystick()
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    xbox.connect()
    joystick.initialized = False
    with pytest.raises(OutcomePiperStateError, match="disconnected"):
        xbox.get_action()


def test_xbox_rejects_mismatched_guid_without_motion_session():
    joystick = FakeJoystick(guid="different-guid")
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)

    with pytest.raises(OutcomePiperStateError, match="GUID does not match"):
        xbox.connect()
    assert not xbox.is_connected


def test_xbox_initially_disconnected_device_fails_fast_without_motion_session():
    joystick = FakeJoystick()
    joystick.initialized = False
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)

    with pytest.raises(OutcomePiperStateError, match="disconnected during connection"):
        xbox.connect()
    assert not xbox.is_connected


def test_bound_xbox_disconnect_immediately_stops_and_latches_e_stop(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    joystick = FakeJoystick()
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    xbox.connect()
    with motion_input_safety_scope():
        connect_for_test(robot)
        joystick.initialized = False
        with pytest.raises(OutcomePiperStateError, match="disconnected"):
            xbox.get_action()
    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.E_STOP
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)


def test_bound_xbox_axis_error_immediately_stops_and_latches_e_stop(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    joystick = FakeJoystick()
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    xbox.connect()
    joystick.get_axis = lambda _: (_ for _ in ()).throw(OSError("USB read failed"))
    with motion_input_safety_scope():
        connect_for_test(robot)
        with pytest.raises(OSError, match="USB read failed"):
            xbox.get_action()
    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.E_STOP
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)


@pytest.mark.parametrize(
    ("failure", "expected_stop_error"),
    [
        ("command", "OSError: electronic stop failed"),
        ("communication_probe", "OSError: stop communication probe failed"),
    ],
)
def test_bound_xbox_failure_reports_electronic_stop_failure(
    tmp_path: Path, failure, expected_stop_error
):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    joystick = FakeJoystick()
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    xbox.connect()
    with motion_input_safety_scope():
        connect_for_test(robot)
        if failure == "command":
            arm.fail_stop = True
        else:
            arm.has_comm_error = lambda: (_ for _ in ()).throw(
                OSError("stop communication probe failed")
            )
        joystick.initialized = False
        with pytest.raises(OutcomePiperStateError, match="stop action failed: OSError"):
            xbox.get_action()

    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.E_STOP
    assert robot.stop_error == expected_stop_error
    robot.disconnect()


def test_xbox_released_hold_preserves_raw_input_for_processor():
    joystick = FakeJoystick()
    joystick.buttons[1] = 0
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    xbox.connect()
    assert xbox.get_action() == {
        "stick_x": 0.0,
        "stick_y": 0.0,
        "stick_z": -0.5,
        "stick_yaw": -0.25,
        "left_trigger": 0.0,
        "right_trigger": 1.0,
        "hold": False,
        "neutral": False,
        "emergency_stop": False,
        "mode_switch": False,
        "translation_switch": False,
        "home": False,
        "work": False,
    }


def test_xbox_trigger_outside_measured_range_fails():
    joystick = FakeJoystick()
    joystick.axes[5] = 1.1
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    xbox.connect()
    with pytest.raises(OutcomePiperValidationError, match="measured range"):
        xbox.get_action()


def test_processor_hold_output_has_only_canonical_seven_fields():
    processor = OutcomePiperXboxProcessor(
        safety=safety(),
        max_xyz_step_m=0.01,
        max_rotation_step_rad=0.02,
        max_gripper_step_m=0.004,
        ik_max_nfev=10,
        ik_timeout_s=0.1,
        ik_residual_tolerance=0.001,
        ik_min_singular_value=0.001,
    )
    processor._current_transition = {TransitionKey.OBSERVATION: valid_action()}
    result = processor.action(
        {
            "stick_x": 0.0,
            "stick_y": 0.0,
            "stick_z": 0.0,
            "stick_yaw": 0.0,
            "left_trigger": 0.0,
            "right_trigger": 0.0,
            "hold": False,
            "neutral": False,
            "emergency_stop": False,
            "mode_switch": False,
            "translation_switch": False,
            "home": False,
            "work": False,
        }
    )
    assert tuple(result) == ACTION_KEYS
    assert isinstance(result, OutcomePiperAction)
    assert result.intent == "hold"


def attach_hold(robot, control):
    from lerobot_robot_outcome_piper.teleop_control import HoldSettings

    settings = HoldSettings(0.01, 0.01, 0.1)
    robot.configure_teleoperation(control, settings)


def test_startup_wait_through_no_dataset_record_loop_does_not_estop(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot_robot_outcome_piper.processor.time.monotonic", lambda: 100.0)
    from lerobot.processor import make_default_processors
    from lerobot.scripts import lerobot_record as official

    robot, arm, _ = make_robot(tmp_path, mode="motion")
    processor = workflows._processor(robot.config, xbox_config())
    attach_hold(robot, processor.steps[0].control)
    connect_for_test(robot)

    joystick = FakeJoystick()
    joystick.axes = [0.0, 0.0, 0.0, 0.0, -1.0, -1.0]
    joystick.buttons[1] = 0
    teleop = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    teleop.connect()

    monkeypatch.setattr(official, "precise_sleep", lambda _: None)
    _, ap, op = make_default_processors()
    try:
        official.record_loop(
            teleop=teleop,
            events={"exit_early": False},
            dataset=None,
            robot=robot,
            fps=20,
            teleop_action_processor=processor,
            robot_action_processor=ap,
            robot_observation_processor=op,
            control_time_s=0.001,
        )
        assert [c for c in arm.calls if isinstance(c, tuple) and c[0] == "move_j"] == [
            ("move_j", [0.0] * 6)
        ]
        assert not arm.gripper.commands and "electronic_emergency_stop" not in arm.calls
        assert robot.state is PiperState.ACTIVE
    finally:
        teleop.disconnect()
        robot.disconnect()


def test_startup_wait_through_official_record_loop_emits_no_training_frame(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot_robot_outcome_piper.processor.time.monotonic", lambda: 100.0)
    from lerobot.processor import make_default_processors
    from lerobot.scripts import lerobot_record as official
    from lerobot_robot_outcome_piper.recording import TelemetryDataset

    robot, arm, _ = make_robot(tmp_path, mode="motion")
    processor = workflows._processor(robot.config, xbox_config())
    attach_hold(robot, processor.steps[0].control)
    connect_for_test(robot)
    frames = []

    class Dataset:
        root = tmp_path
        num_episodes = 0
        fps = 20
        features = {"action": {"dtype": "float32", "shape": (7,), "names": list(ACTION_KEYS)}}

        def add_frame(self, frame):
            frames.append(frame)

    audit = TelemetryDataset(Dataset(), robot)
    _, ap, op = make_default_processors()
    joystick = FakeJoystick()
    joystick.buttons[1] = 0
    teleop = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    teleop.connect()
    monkeypatch.setattr(official, "precise_sleep", lambda _: None)
    try:
        official.record_loop(
            robot=robot,
            events={"exit_early": False},
            fps=20,
            teleop_action_processor=processor,
            robot_action_processor=ap,
            robot_observation_processor=op,
            dataset=audit,
            teleop=teleop,
            control_time_s=0.001,
            single_task="startup wait",
        )
        assert frames == [] and not arm.gripper.commands
        assert "electronic_emergency_stop" not in arm.calls
    finally:
        audit.close()
        teleop.disconnect()
        robot.disconnect()


@pytest.mark.parametrize(
    "observation, message",
    [
        ({**valid_action(), "joint_1.pos": 1.1}, "joint limits"),
        ({**valid_action(), "gripper.pos": -0.01}, "negative"),
    ],
)
def test_processor_rejects_observation_outside_frozen_limits(observation, message):
    processor = OutcomePiperXboxProcessor(
        safety=safety(),
        max_xyz_step_m=0.01,
        max_rotation_step_rad=0.02,
        max_gripper_step_m=0.004,
        ik_max_nfev=10,
        ik_timeout_s=0.1,
        ik_residual_tolerance=0.001,
        ik_min_singular_value=0.001,
    )
    processor._current_transition = {TransitionKey.OBSERVATION: observation}
    with pytest.raises(OutcomePiperValidationError, match=message):
        processor.action(
            {
                "stick_x": 0.0,
                "stick_y": 0.0,
                "stick_z": 0.0,
                "stick_yaw": 0.0,
                "left_trigger": 0.0,
                "right_trigger": 0.0,
                "hold": False,
                "neutral": False,
                "emergency_stop": False,
                "mode_switch": False,
                "translation_switch": False,
                "home": False,
                "work": False,
            }
        )


def test_processor_ik_failure_does_not_call_robot_sdk(tmp_path: Path, monkeypatch):
    pytest.importorskip("pyAgxArm")
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    processor = OutcomePiperXboxProcessor(
        safety=safety(),
        max_xyz_step_m=0.01,
        max_rotation_step_rad=0.02,
        max_gripper_step_m=0.004,
        ik_max_nfev=10,
        ik_timeout_s=0.1,
        ik_residual_tolerance=0.001,
        ik_min_singular_value=0.001,
    )
    processor._current_transition = {TransitionKey.OBSERVATION: valid_action()}
    from lerobot_robot_outcome_piper.teleop_control import TranslationStrategy

    processor.control.translation_strategy = TranslationStrategy.FIXED_ORIENTATION
    processor.control.confirm_hold()
    processor.control.observe(False, True)
    processor.control.observe(True, True)
    monkeypatch.setattr(
        processor,
        "_solve",
        lambda *_: (_ for _ in ()).throw(OutcomePiperValidationError("IK failed")),
    )
    with pytest.raises(OutcomePiperValidationError, match="IK failed"):
        processor.action(
            {
                "stick_x": 0.001,
                "stick_y": 0.0,
                "stick_z": 0.0,
                "stick_yaw": 0.0,
                "left_trigger": 0.0,
                "right_trigger": 0.0,
                "hold": True,
                "neutral": False,
                "emergency_stop": False,
                "mode_switch": False,
                "translation_switch": False,
                "home": False,
                "work": False,
            }
        )
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)


def test_bound_processor_ik_failure_stops_without_motion(tmp_path: Path, monkeypatch):
    pytest.importorskip("pyAgxArm")
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    processor = OutcomePiperXboxProcessor(
        safety=safety(),
        max_xyz_step_m=0.01,
        max_rotation_step_rad=0.02,
        max_gripper_step_m=0.004,
        ik_max_nfev=10,
        ik_timeout_s=0.1,
        ik_residual_tolerance=0.001,
        ik_min_singular_value=0.001,
    )
    processor._current_transition = {TransitionKey.OBSERVATION: valid_action()}
    from lerobot_robot_outcome_piper.teleop_control import TranslationStrategy

    processor.control.translation_strategy = TranslationStrategy.FIXED_ORIENTATION
    processor.control.confirm_hold()
    processor.control.observe(False, True)
    processor.control.observe(True, True)
    monkeypatch.setattr(
        processor,
        "_solve",
        lambda *_: (_ for _ in ()).throw(OutcomePiperValidationError("IK failed")),
    )

    with motion_input_safety_scope():
        connect_for_test(robot)
        with pytest.raises(OutcomePiperValidationError, match="IK failed"):
            processor.action(
                {
                    "stick_x": 0.001,
                    "stick_y": 0.0,
                    "stick_z": 0.0,
                    "stick_yaw": 0.0,
                    "left_trigger": 0.0,
                    "right_trigger": 0.0,
                    "hold": True,
                    "neutral": False,
                    "emergency_stop": False,
                    "mode_switch": False,
                    "translation_switch": False,
                    "home": False,
                    "work": False,
                }
            )

    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.E_STOP
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)
    assert arm.gripper.commands == []
    robot.disconnect()


def test_cli_registers_plugins_and_forwards_arguments(monkeypatch):
    events = []
    monkeypatch.setattr(cli, "register_third_party_plugins", lambda: events.append("plugins"))
    monkeypatch.setattr(cli, "_teleoperate_from_cli", lambda: events.append(tuple(sys.argv[1:])))
    cli.teleoperate_main(["--robot.type=outcome_piper"])
    assert events == ["plugins", ("--robot.type=outcome_piper",)]


def test_record_injects_canonical_processor_into_official_recorder(monkeypatch):
    robot_config = SimpleNamespace(
        execution_mode="motion",
        cameras={"d435": SimpleNamespace(fps=20)},
        capture_timing=object(),
        scene=object(),
    )
    teleop_config = SimpleNamespace(control_hz=20)
    cfg = SimpleNamespace(
        robot=robot_config,
        teleop=teleop_config,
        dataset=SimpleNamespace(fps=20, push_to_hub=False),
    )
    sentinel = object()
    captured = {}
    monkeypatch.setattr(
        workflows, "_validate_workflow_configs", lambda *_: (robot_config, teleop_config)
    )
    monkeypatch.setattr(workflows, "_processor", lambda *_: sentinel)

    def fake_record(received_cfg, *, teleop_action_processor):
        captured["cfg"] = received_cfg
        captured["processor"] = teleop_action_processor
        return "dataset"

    from lerobot_robot_outcome_piper import recording

    monkeypatch.setattr(recording, "record_with_telemetry", fake_record)
    assert workflows.record(cfg) == "dataset"
    assert captured == {"cfg": cfg, "processor": sentinel}


def test_record_rejects_hub_push_inside_isolated_can_namespace(monkeypatch):
    robot_config = SimpleNamespace(
        execution_mode="motion", cameras={"d435": SimpleNamespace(fps=20)}, capture_timing=object()
    )
    teleop_config = SimpleNamespace(control_hz=20)
    cfg = SimpleNamespace(
        robot=robot_config,
        teleop=teleop_config,
        dataset=SimpleNamespace(fps=20, push_to_hub=True),
    )
    monkeypatch.setattr(
        workflows, "_validate_workflow_configs", lambda *_: (robot_config, teleop_config)
    )
    with pytest.raises(ValueError, match="dataset.push_to_hub=false"):
        workflows.record(cfg)


@pytest.mark.parametrize(
    "cameras, message", [({"front": SimpleNamespace(fps=15)}, "exceeds a configured camera")]
)
def test_record_rejects_missing_camera_or_rate_mismatch(monkeypatch, cameras, message):
    robot_config = SimpleNamespace(cameras=cameras)
    teleop_config = SimpleNamespace(control_hz=20)
    cfg = SimpleNamespace(
        robot=robot_config,
        teleop=teleop_config,
        dataset=SimpleNamespace(fps=20, push_to_hub=False),
    )
    monkeypatch.setattr(
        workflows, "_validate_workflow_configs", lambda *_: (robot_config, teleop_config)
    )
    with pytest.raises(ValueError, match=message):
        workflows.record(cfg)


def test_record_does_not_connect_robot_if_teleop_connect_fails(tmp_path: Path, monkeypatch):
    import lerobot.scripts.lerobot_record as official
    from lerobot_robot_outcome_piper.scene import SceneContext

    robot, arm, _ = make_robot(tmp_path, mode="motion")

    class Dataset:
        writer = None
        root = tmp_path / "dataset"

        def finalize(self):
            pass

    class DatasetConfig:
        fps = 20
        video = False
        repo_id = "test/piper-record-input-failure"
        root = tmp_path / "dataset"
        image_writer_processes = 0
        num_image_writer_processes = 0
        num_image_writer_threads_per_camera = 0
        video_encoding_batch_size = 1
        rgb_encoder = None
        depth_encoder = None
        encoder_threads = 0
        streaming_encoding = False
        encoder_queue_maxsize = 1
        push_to_hub = False

        def stamp_repo_id(self):
            pass

    failing_teleop = OutcomePiperXbox(
        xbox_config(),
        joystick_factory=lambda _: (_ for _ in ()).throw(OSError("Xbox enumeration failed")),
    )

    cfg = SimpleNamespace(
        robot=replace(
            robot.config,
            cameras={"d435": d435_config(fps=20)},
            capture_timing=CaptureTiming(0.1, 0.1, 0.1, 0.1),
            scene=SceneContext("synthetic", "base", "view", "area"),
        ),
        teleop=xbox_config(),
        dataset=DatasetConfig(),
        raw_root=str(tmp_path / "raw"),
        display_data=False,
        display_compressed_images=False,
        play_sounds=False,
        resume=False,
    )
    from lerobot_robot_outcome_piper.teleop_control import TeleopControl

    monkeypatch.setattr(
        workflows,
        "_processor",
        lambda *_: SimpleNamespace(steps=[SimpleNamespace(control=TeleopControl())]),
    )
    monkeypatch.setattr(official, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(official, "make_teleoperator_from_config", lambda _: failing_teleop)
    monkeypatch.setattr(official.LeRobotDataset, "create", lambda *args, **kwargs: Dataset())
    monkeypatch.setattr(official, "aggregate_pipeline_dataset_features", lambda **kwargs: {})
    monkeypatch.setattr(official, "create_initial_features", lambda **kwargs: {})
    monkeypatch.setattr(official, "combine_feature_dicts", lambda *args: {})
    monkeypatch.setattr(official, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(official, "asdict", lambda _: {})

    with pytest.raises(OSError, match="Xbox enumeration failed"):
        workflows.record(cfg)

    assert arm.calls == []
    assert robot.state is PiperState.DISCONNECTED


def test_speed_configuration_unknown_echo_is_replaced_by_confirmed_j(tmp_path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    speed, mode = arm.set_speed_percent, arm.set_motion_mode

    def set_speed(value):
        speed(value)
        arm.mode_feedback = 255

    def set_mode(value):
        mode(value)
        arm.mode_feedback = 1

    arm.set_speed_percent = set_speed
    arm.set_motion_mode = set_mode
    try:
        connect_for_test(robot)
        assert arm.mode_feedback == 1 and robot.state is PiperState.ACTIVE
        assert arm.calls.index(("speed_percent", 5)) < arm.calls.index(("motion_mode", "J"))
    finally:
        robot.disconnect()


def connect_for_test(robot):
    robot.connect()
    if robot.config.execution_mode == "motion":
        robot.enable()
