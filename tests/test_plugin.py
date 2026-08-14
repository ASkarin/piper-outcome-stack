from __future__ import annotations

import importlib.metadata
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
from lerobot_robot_outcome_piper.robot import OutcomePiper, PiperState  # noqa: E402
from lerobot_robot_outcome_piper.safety import ACTION_KEYS, JOINT_KEYS, MotionSafety  # noqa: E402
from lerobot_robot_outcome_piper.teleoperator import OutcomePiperXbox  # noqa: E402
from lerobot_robot_outcome_piper import cli, workflows  # noqa: E402


NOW = 1_800_000_000.0


class FakeFps:
    def __init__(self, values=(11.0, 12.0, 13.0)):
        self.values = dict(zip((12, 34, 56), values, strict=True))

    def get_fps(self, message_type):
        return self.values[message_type]


class FakeGripper:
    def __init__(self, arm):
        self.arm = arm
        self.width = 0.03
        self.status_code = 0
        self.timestamp = NOW - 0.02
        self.hz = 50.0
        self.commands = []

    def get_gripper_status(self):
        if self.arm.fail_feedback:
            raise OSError("feedback failed")
        return SimpleNamespace(
            msg=SimpleNamespace(value=self.width, mode="width", status_code=self.status_code),
            timestamp=self.timestamp,
            hz=self.hz,
        )

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
        self.joints = [0.0] * 6
        self.status = 0
        self.error_code = 0
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

    def set_speed_percent(self, value):
        self.calls.append(("speed_percent", value))

    def enable(self):
        self.calls.append("enable")
        return self.enable_result

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


def write_gate(tmp_path: Path, *, nameplate_model="PiPER") -> tuple[Path, Path]:
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
    import hashlib

    digest = "sha256:" + hashlib.sha256(safety_path.read_bytes()).hexdigest()
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "schema_version": "outcome-piper-hardware-acceptance-v1",
                "standard_piper_verified": True,
                "official_gripper_verified": True,
                "official_power_and_harness_verified": True,
                "official_usb_can_verified": True,
                "physical_emergency_stop_verified": True,
                "five_read_only_cycles_verified": True,
                "communication_loss_stop_verified": True,
                "watchdog_stop_verified": True,
                "hold_to_run_stop_verified": True,
                "electronic_emergency_stop_verified": True,
                "no_drop_stop_verified": True,
                "stop_strategy_verified": True,
                "stop_strategy": "electronic_emergency_stop",
                "acceptance_id": "acceptance-test-001",
                "validated_at_utc": "2026-08-31T00:00:00Z",
                "validated_by": "operator-test",
                "nameplate_model": nameplate_model,
                "robot_serial_number": "PIPER-TEST-001",
                "gripper_identifier": "AGX-GRIPPER-TEST-001",
                "usb_can_identifier": "USB-CAN-TEST-001",
                "physical_emergency_stop_identifier": "ESTOP-TEST-001",
                "can_interface": "can-test",
                "firmware": "v189",
                "firmware_identity": {
                    "hardware_version": "H-V1.2-1",
                    "motor_ratio_and_batch": "10",
                    "node_type": "ARM_MC",
                    "software_version": "S-V1.8-9",
                    "production_date": "260813",
                    "node_number": "15",
                },
                "safety_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    return safety_path, acceptance_path


def config(tmp_path: Path, *, mode="read_only", accepted=True):
    kwargs = {}
    if mode == "motion":
        safety_path, acceptance_path = write_gate(tmp_path)
        if not accepted:
            document = json.loads(acceptance_path.read_text(encoding="utf-8"))
            document["physical_emergency_stop_verified"] = False
            acceptance_path.write_text(json.dumps(document), encoding="utf-8")
        kwargs = {"safety_path": safety_path, "hardware_acceptance_path": acceptance_path}
    return OutcomePiperConfig(
        can_interface="can-test",
        firmware="v189",
        feedback_timeout_s=0.2,
        execution_mode=mode,
        calibration_dir=tmp_path / "calibration",
        **kwargs,
    )


def make_robot(tmp_path: Path, *, mode="read_only", accepted=True):
    arm = FakeArm()
    factory_calls = []

    def factory(can_interface, firmware):
        factory_calls.append((can_interface, firmware))
        return arm

    robot = OutcomePiper(
        config(tmp_path, mode=mode, accepted=accepted),
        piper_factory=factory,
        camera_factory=lambda _: {},
        monotonic=lambda: 100.0,
        wall_time=lambda: NOW,
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
    robot.connect()
    assert robot.state is PiperState.CONNECTED_DISABLED
    assert "enable" not in arm.calls
    assert not any(
        isinstance(call, tuple) and call[0] in {"auto_mode", "sdk_limits", "motion_mode"}
        for call in arm.calls
    )
    with pytest.raises(OutcomePiperStateError):
        robot.send_action(valid_action())


def test_motion_gate_rejects_before_sdk_construction(tmp_path: Path):
    robot, _, factory_calls = make_robot(tmp_path, mode="motion", accepted=False)
    with pytest.raises(OutcomePiperValidationError, match="gate is incomplete"):
        robot.connect()
    assert factory_calls == []


@pytest.mark.parametrize("nameplate_model", ["PiPER-H", "PiPER-L", "PiPER-X"])
def test_motion_gate_rejects_nonstandard_nameplate_before_sdk_construction(
    tmp_path: Path, nameplate_model
):
    safety_path, acceptance_path = write_gate(tmp_path, nameplate_model=nameplate_model)
    factory_calls = []
    robot = OutcomePiper(
        OutcomePiperConfig(
            can_interface="can-test",
            firmware="v189",
            feedback_timeout_s=0.2,
            execution_mode="motion",
            safety_path=safety_path,
            hardware_acceptance_path=acceptance_path,
        ),
        piper_factory=lambda *args: factory_calls.append(args),
        camera_factory=lambda _: {},
    )
    with pytest.raises(OutcomePiperValidationError, match="exactly 'PiPER'"):
        robot.connect()
    assert factory_calls == []


def test_motion_connect_configures_one_mode_and_enables(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
    assert robot.state is PiperState.ACTIVE
    assert ("motion_mode", "J") in arm.calls
    assert ("speed_percent", 5) in arm.calls
    assert "enable" in arm.calls


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hardware_version", "H-V9.9-9", "hardware_version"),
        ("node_type", "PIPER_X", "node_type"),
        ("software_version", "S-V1.8-8", "software_version"),
    ],
)
def test_motion_gate_binds_live_firmware_identity_before_enable(
    tmp_path: Path, field, value, message
):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    arm.firmware[field] = value
    with pytest.raises(OutcomePiperStateError, match=message):
        robot.connect()
    assert "enable" not in arm.calls


@pytest.mark.parametrize(("ctrl_mode", "mode_feedback"), [(0, 1), (1, 0)])
def test_motion_mode_feedback_must_confirm_can_move_j_before_enable(
    tmp_path: Path, ctrl_mode, mode_feedback
):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    arm.ctrl_mode = ctrl_mode
    arm.mode_feedback = mode_feedback
    with pytest.raises(OutcomePiperStateError, match="motion-mode feedback mismatch"):
        robot.connect()
    assert "enable" not in arm.calls


@pytest.mark.parametrize("failed_call", ["auto_mode", "sdk_limits", "motion_mode"])
def test_motion_configure_checks_each_sdk_step_immediately(tmp_path: Path, failed_call):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    original_has_comm_error = arm.has_comm_error

    def has_comm_error():
        return any(isinstance(call, tuple) and call[0] == failed_call for call in arm.calls) or (
            original_has_comm_error()
        )

    arm.has_comm_error = has_comm_error
    with pytest.raises(OutcomePiperStateError, match="latched FAULT"):
        robot.connect()
    configured = [call[0] for call in arm.calls if isinstance(call, tuple)]
    expected = {
        "auto_mode": ["get_firmware", "init_effector", "auto_mode"],
        "sdk_limits": ["get_firmware", "init_effector", "auto_mode", "sdk_limits"],
        "motion_mode": [
            "get_firmware",
            "init_effector",
            "auto_mode",
            "sdk_limits",
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
    robot.connect()
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
    robot.connect()
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
    robot.connect()
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
    robot.connect()
    result = robot.send_action(valid_action())
    assert result == valid_action()
    assert ("speed_percent", 5) in arm.calls
    assert arm.gripper.commands == [(0.03, 0.5)]


@pytest.mark.parametrize("timestamp", [NOW - 0.21, NOW + 0.01])
def test_stale_or_future_feedback_latches_fault(tmp_path: Path, timestamp):
    robot, arm, _ = make_robot(tmp_path)
    arm._parser.joint_12.timestamp = timestamp
    with pytest.raises(OutcomePiperStateError):
        robot.connect()
    assert robot.state is PiperState.FAULT


def test_sdk_feedback_failure_latches_and_disconnect_cannot_clear_session(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path)
    robot.connect()
    arm.fail_feedback = True
    with pytest.raises(OutcomePiperStateError, match="latched FAULT"):
        robot.get_observation()
    robot.disconnect()
    assert robot.state is PiperState.FAULT
    with pytest.raises(OutcomePiperStateError):
        robot.connect()


@pytest.mark.parametrize("malformation", ["gripper_status", "frame_frequency"])
def test_feedback_parse_failure_stops_and_terminally_latches_session(tmp_path: Path, malformation):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
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
        robot.connect()


def test_active_arm_disconnect_stops_and_terminally_latches_session(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
    arm.connected = False

    with pytest.raises(OutcomePiperStateError, match="connection lost: arm"):
        robot.get_observation()

    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.FAULT
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected
    with pytest.raises(OutcomePiperStateError, match="terminally latched FAULT"):
        robot.connect()


def test_active_camera_disconnect_stops_and_terminally_latches_session(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
    robot.cameras["wrist"] = FakeCamera(connected=False)

    with pytest.raises(OutcomePiperStateError, match="camera 'wrist'"):
        robot.get_observation()

    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.FAULT
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected
    with pytest.raises(OutcomePiperStateError, match="terminally latched FAULT"):
        robot.connect()


@pytest.mark.parametrize("probe_target", ["arm", "camera"])
def test_active_connection_probe_error_stops_and_latches_session(tmp_path: Path, probe_target):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
    if probe_target == "arm":
        arm.is_connected = lambda: (_ for _ in ()).throw(OSError("arm probe failed"))
    else:
        robot.cameras["wrist"] = FakeCamera(fail_probe=True)

    with pytest.raises(OutcomePiperStateError, match="latched FAULT"):
        robot.get_observation()

    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.FAULT
    robot.disconnect()


def test_disconnect_never_homes_resets_disables_or_stops(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
    robot.disconnect()
    assert not any(
        (isinstance(call, str) and call in {"home", "reset", "disable"})
        or (isinstance(call, tuple) and call[0] in {"home", "reset", "disable"})
        for call in arm.calls
    )


def test_disconnect_is_serialized_after_inflight_action(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
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


def test_watchdog_waits_for_inflight_action_and_observes_fresh_completion(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
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
    assert action_errors == []
    assert robot.state is PiperState.ACTIVE
    robot.disconnect()
    assert "electronic_emergency_stop" not in arm.calls


def test_watchdog_stop_failure_is_exposed_to_the_control_loop(tmp_path: Path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
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
    robot.connect()
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
        robot.connect()
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
        robot.connect()
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
        robot.connect()


def _capture_error(target, operation, *args):
    try:
        operation(*args)
    except Exception as exc:
        target.append(exc)


class FakeJoystick:
    def __init__(self, guid="measured-guid"):
        self.guid = guid
        self.axes = [0.0, 0.05, -0.5, 0.25, -1.0, 1.0]
        self.buttons = [0, 1]
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
        "deadzone": 0.1,
        "control_hz": 20,
        "xyz_step_m": 0.01,
        "yaw_step_rad": 0.02,
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
        "delta_x": 0.0,
        "delta_y": 0.0,
        "delta_z": -0.005,
        "delta_yaw": -0.005,
        "delta_gripper": 0.004,
        "hold": True,
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
        robot.connect()
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
        robot.connect()
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
        robot.connect()
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


def test_xbox_hold_to_run_outputs_zero_motion():
    joystick = FakeJoystick()
    joystick.buttons[1] = 0
    xbox = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    xbox.connect()
    assert xbox.get_action() == {
        "delta_x": 0.0,
        "delta_y": 0.0,
        "delta_z": 0.0,
        "delta_yaw": 0.0,
        "delta_gripper": 0.0,
        "hold": False,
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
        max_yaw_step_rad=0.02,
        max_gripper_step_m=0.004,
        ik_max_nfev=10,
        ik_timeout_s=0.1,
        ik_residual_tolerance=0.001,
        ik_min_singular_value=0.001,
    )
    processor._current_transition = {TransitionKey.OBSERVATION: valid_action()}
    result = processor.action(
        {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "delta_yaw": 0.0,
            "delta_gripper": 0.0,
            "hold": False,
        }
    )
    assert tuple(result) == ACTION_KEYS
    assert isinstance(result, OutcomePiperAction)
    assert result.execute_motion is False


def test_hold_release_through_official_teleop_loop_latches_e_stop_before_motion(
    tmp_path: Path, monkeypatch
):
    from lerobot.processor import make_default_processors
    from lerobot.scripts import lerobot_teleoperate as official

    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
    processor = workflows._processor(robot.config, xbox_config())

    class ReleasedTeleop:
        name = "outcome_piper_xbox"

        def get_action(self):
            return {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "delta_yaw": 0.0,
                "delta_gripper": 0.0,
                "hold": False,
            }

    monkeypatch.setattr(official, "precise_sleep", lambda _: None)
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    with pytest.raises(OutcomePiperStateError, match="hold-to-run was released"):
        official.teleop_loop(
            teleop=ReleasedTeleop(),
            robot=robot,
            fps=20,
            teleop_action_processor=processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            duration=0.0,
        )
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)
    assert arm.gripper.commands == []
    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.E_STOP


def test_hold_release_through_official_record_loop_stops_without_recording_or_motion(
    tmp_path: Path, monkeypatch
):
    from lerobot.processor import make_default_processors
    from lerobot.scripts import lerobot_record as official

    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
    processor = workflows._processor(robot.config, xbox_config())

    frames = []

    class Dataset:
        fps = 20
        features = {"action": {"dtype": "float32", "shape": (7,), "names": list(ACTION_KEYS)}}

        def add_frame(self, frame):
            frames.append(frame)

    clock = iter((0.0, 0.0, 0.06, 0.06))
    monkeypatch.setattr(official.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(official, "precise_sleep", lambda _: None)
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    joystick = FakeJoystick()
    joystick.buttons[1] = 0
    teleop = OutcomePiperXbox(xbox_config(), joystick_factory=lambda _: joystick)
    teleop.connect()
    with pytest.raises(OutcomePiperStateError, match="hold-to-run was released"):
        official.record_loop(
            robot=robot,
            events={"exit_early": False},
            fps=20,
            teleop_action_processor=processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            dataset=Dataset(),
            teleop=teleop,
            control_time_s=0.05,
            single_task="hold release smoke",
        )
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)
    assert arm.gripper.commands == []
    assert frames == []
    assert "electronic_emergency_stop" in arm.calls


@pytest.mark.parametrize(
    "observation, message",
    [
        ({**valid_action(), "joint_1.pos": 1.1}, "joint limits"),
        ({**valid_action(), "gripper.pos": 0.09}, "gripper limits"),
    ],
)
def test_processor_rejects_observation_outside_frozen_limits(observation, message):
    processor = OutcomePiperXboxProcessor(
        safety=safety(),
        max_xyz_step_m=0.01,
        max_yaw_step_rad=0.02,
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
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "delta_yaw": 0.0,
                "delta_gripper": 0.0,
                "hold": False,
            }
        )


def test_processor_ik_failure_does_not_call_robot_sdk(tmp_path: Path, monkeypatch):
    pytest.importorskip("pyAgxArm")
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
    processor = OutcomePiperXboxProcessor(
        safety=safety(),
        max_xyz_step_m=0.01,
        max_yaw_step_rad=0.02,
        max_gripper_step_m=0.004,
        ik_max_nfev=10,
        ik_timeout_s=0.1,
        ik_residual_tolerance=0.001,
        ik_min_singular_value=0.001,
    )
    processor._current_transition = {TransitionKey.OBSERVATION: valid_action()}
    monkeypatch.setattr(
        processor,
        "_solve",
        lambda *_: (_ for _ in ()).throw(OutcomePiperValidationError("IK failed")),
    )
    with pytest.raises(OutcomePiperValidationError, match="IK failed"):
        processor.action(
            {
                "delta_x": 0.001,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "delta_yaw": 0.0,
                "delta_gripper": 0.0,
                "hold": True,
            }
        )
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in arm.calls)


def test_bound_processor_ik_failure_stops_without_motion(tmp_path: Path, monkeypatch):
    pytest.importorskip("pyAgxArm")
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    processor = OutcomePiperXboxProcessor(
        safety=safety(),
        max_xyz_step_m=0.01,
        max_yaw_step_rad=0.02,
        max_gripper_step_m=0.004,
        ik_max_nfev=10,
        ik_timeout_s=0.1,
        ik_residual_tolerance=0.001,
        ik_min_singular_value=0.001,
    )
    processor._current_transition = {TransitionKey.OBSERVATION: valid_action()}
    monkeypatch.setattr(
        processor,
        "_solve",
        lambda *_: (_ for _ in ()).throw(OutcomePiperValidationError("IK failed")),
    )

    with motion_input_safety_scope():
        robot.connect()
        with pytest.raises(OutcomePiperValidationError, match="IK failed"):
            processor.action(
                {
                    "delta_x": 0.001,
                    "delta_y": 0.0,
                    "delta_z": 0.0,
                    "delta_yaw": 0.0,
                    "delta_gripper": 0.0,
                    "hold": True,
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
    robot_config = SimpleNamespace(execution_mode="motion")
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
    import lerobot.scripts.lerobot_record as official

    def fake_record(received_cfg, *, teleop_action_processor):
        captured["cfg"] = received_cfg
        captured["processor"] = teleop_action_processor
        return "dataset"

    monkeypatch.setattr(official, "record", fake_record)
    assert workflows.record(cfg) == "dataset"
    assert captured == {"cfg": cfg, "processor": sentinel}


def test_record_rejects_hub_push_inside_isolated_can_namespace(monkeypatch):
    robot_config = SimpleNamespace(execution_mode="motion")
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


def test_record_active_robot_stops_if_teleop_connect_fails(tmp_path: Path, monkeypatch):
    import lerobot.scripts.lerobot_record as official

    robot, arm, _ = make_robot(tmp_path, mode="motion")

    class Dataset:
        writer = None

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
        robot=robot.config,
        teleop=xbox_config(),
        dataset=DatasetConfig(),
        display_data=False,
        display_compressed_images=False,
        play_sounds=False,
        resume=False,
    )
    monkeypatch.setattr(workflows, "_processor", lambda *_: object())
    monkeypatch.setattr(official, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(official, "make_teleoperator_from_config", lambda _: failing_teleop)
    monkeypatch.setattr(official.LeRobotDataset, "create", lambda *args, **kwargs: Dataset())
    monkeypatch.setattr(official, "aggregate_pipeline_dataset_features", lambda **kwargs: {})
    monkeypatch.setattr(official, "create_initial_features", lambda **kwargs: {})
    monkeypatch.setattr(official, "combine_feature_dicts", lambda *args: {})
    monkeypatch.setattr(official, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(official, "asdict", lambda _: {})

    original_record = official.record
    monkeypatch.setattr(official, "record", original_record.__wrapped__)
    with pytest.raises(OSError, match="Xbox enumeration failed"):
        workflows.record(cfg)

    assert "electronic_emergency_stop" in arm.calls
    assert robot.state is PiperState.E_STOP
    assert arm.calls.index("electronic_emergency_stop") < arm.calls.index("disconnect")
