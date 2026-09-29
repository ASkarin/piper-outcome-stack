import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from lerobot_robot_outcome_piper.continuous_replay import plan_replay, execute_replay
from lerobot_robot_outcome_piper.safety import load_motion_safety, ACTION_KEYS
from lerobot_robot_outcome_piper.teleop_control import HoldSettings
from test_plugin import make_robot


def limits(tmp_path):
    r, _, _ = make_robot(tmp_path, mode="motion")
    return load_motion_safety(r.config.safety_path)


def test_pchip_keeps_endpoints_and_nonnegative_gripper_without_mutating_source(tmp_path):
    safety = limits(tmp_path)
    source = np.array([[0.1] * 6 + [0.004], [0.11] * 6 + [0.008], [0.1] * 6 + [0.0]])
    before = source.copy()
    plan = plan_replay(source, 20, 20, 1.25, safety)
    np.testing.assert_array_equal(source, before)
    values = np.array([list(a.values()) for a in plan["actions"]])
    np.testing.assert_allclose(values[[0, -1]], source[[0, -1]], atol=1e-8)
    assert values[:, 6].min() >= 0 and values[:, 6].max() <= 0.02000001
    assert plan["nominal_duration_s"] >= 0.125 and len(values) > len(source)


def test_derived_workspace_checks_are_not_relaxed(tmp_path, monkeypatch):
    import lerobot_robot_outcome_piper.continuous_replay as module

    monkeypatch.setattr(
        module, "check_execution_target", lambda *a: (_ for _ in ()).throw(ValueError("workspace"))
    )
    with pytest.raises(ValueError, match="workspace"):
        plan_replay([[0.1] * 6 + [0.01], [0.11] * 6 + [0.01]], 20, 20, 1.25, limits(tmp_path))


class Robot:
    def __init__(self, stall=False):
        self.now = 1.0
        self.value = dict(zip(ACTION_KEYS, [0.1] * 6 + [0.01]))
        self.sent = []
        self.holds = []
        self.stall = stall
        self.last_action_telemetry = None

    def observe_control(self):
        self.last_feedback_telemetry = NS(received_monotonic_s=[self.now] * 5)
        return dict(self.value)

    def send_action(self, a):
        self.sent.append(dict(a))
        self.last_action_telemetry = {"result": "sdk_returned", "values": dict(a)}
        if not self.stall:
            self.value = dict(a)

    def send_joint_target(self, q):
        self.holds.append(list(q))

    def sleep(self, dt):
        self.now += dt


def test_replay_final_hold_and_trace_use_derived_commands(tmp_path):
    safety = limits(tmp_path)
    plan = plan_replay(
        [[0.1] * 6 + [0.01], [0.11] * 6 + [0.012], [0.12] * 6 + [0.01]], 20, 20, 1.25, safety
    )
    robot = Robot()
    trace = []
    result = execute_replay(
        robot,
        plan,
        safety,
        HoldSettings(0.001, 0.1, 1.0),
        trace,
        NS(poll=lambda: None),
        lambda: robot.now,
        robot.sleep,
    )
    assert result["status"] == "targets_sent_final_joints_confirmed"
    assert robot.sent == plan["actions"]
    assert all(x["sdk_call"]["values"] == x["target"] for x in trace)


def test_wait_feedback_does_not_resend_or_fabricate_sdk_calls(tmp_path):
    safety = limits(tmp_path)
    robot = Robot(stall=True)
    plan = {
        "actions": [dict(robot.value), dict(zip(ACTION_KEYS, [0.5] * 6 + [0.01]))],
        "control_hz": 20,
        "source_time_s": [0, 0.05],
        "source_rows": [0, 0],
    }
    trace = []
    with pytest.raises(RuntimeError, match="timeout"):
        execute_replay(
            robot,
            plan,
            safety,
            HoldSettings(0.001, 0.1, 0.2),
            trace,
            NS(poll=lambda: None),
            lambda: robot.now,
            robot.sleep,
        )
    assert len(robot.sent) == 1
    assert all(x["sdk_call"] is None for x in trace if x["phase"] == "wait_feedback")


def test_stop_holds_joints_without_new_gripper_command(tmp_path):
    safety = limits(tmp_path)
    robot = Robot()
    plan = {
        "actions": [dict(robot.value)],
        "control_hz": 20,
        "source_time_s": [0],
        "source_rows": [0],
    }
    result = execute_replay(
        robot,
        plan,
        safety,
        HoldSettings(0.001, 0.1, 1.0),
        [],
        NS(poll=lambda: "stop"),
        lambda: robot.now,
        robot.sleep,
    )
    assert result["status"] == "cancelled_held" and len(robot.holds) == 1 and robot.sent == []


@pytest.mark.parametrize("failure", ["feedback", "sdk", "interrupt"])
def test_replay_failure_stops_without_later_targets(tmp_path, failure):
    safety = limits(tmp_path)
    robot = Robot()
    plan = {
        "actions": [dict(robot.value), dict(robot.value)],
        "control_hz": 20,
        "source_time_s": [0, 0.05],
        "source_rows": [0, 0],
    }
    if failure == "feedback":
        robot.observe_control = lambda: (_ for _ in ()).throw(RuntimeError("feedback expired"))
    elif failure == "sdk":
        robot.send_action = lambda a: (_ for _ in ()).throw(RuntimeError("SDK failure"))
    else:
        robot.sleep = lambda dt: (_ for _ in ()).throw(KeyboardInterrupt())
    trace = []
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
        execute_replay(
            robot,
            plan,
            safety,
            HoldSettings(0.001, 0.1, 1.0),
            trace,
            NS(poll=lambda: None),
            lambda: robot.now,
            robot.sleep,
        )
    assert len(robot.sent) < 2
    if failure == "sdk":
        assert trace[-1]["dispatch_result"] == "failed"


@pytest.mark.parametrize("at_start", [True, False])
def test_continuous_cli_workflow_lifecycle_without_hardware(tmp_path, monkeypatch, at_start):
    import time
    from lerobot_robot_outcome_piper import workflows, robot as robot_module, record_control

    config_robot, _, _ = make_robot(tmp_path, mode="motion")
    calls = []

    class Device(Robot):
        stop_outcome = None

        def __init__(self, cfg):
            super().__init__()
            if not at_start:
                self.value["joint_1.pos"] = 0.2

        def connect(self):
            calls.append("connect")

        def enable(self):
            calls.append("enable")

        def disconnect(self):
            calls.append("disconnect")

        def observe_control(self):
            self.now = time.monotonic()
            return super().observe_control()

        def get_observation(self):
            return self.observe_control()

        def _validate_action(self, a):
            calls.append("validate")

        def request_emergency_stop(self, cause):
            calls.append("estop")

    monkeypatch.setattr(robot_module, "OutcomePiper", Device)
    commands = iter(["start"])
    monkeypatch.setattr(
        record_control, "TerminalCommands", lambda: NS(poll=lambda: next(commands, None))
    )
    cfg = NS(
        robot=config_robot.config,
        dataset=NS(root="/synthetic", episode=0),
        trajectory={
            "time_scale": 1.25,
            "control_hz": 20,
            "hold_settings": {
                "joint_tolerance_rad": 0.001,
                "stable_time_s": 0.01,
                "timeout_s": 1.0,
            },
        },
        trajectory_report_path=str(tmp_path / "result.json"),
    )
    cfg.robot.cameras = {}
    ds = NS(num_frames=2, fps=20)
    actions = [{"action": [0.1] * 6 + [0.01]}, {"action": [0.11] * 6 + [0.01]}]
    if at_start:
        workflows._continuous_replay(cfg, ds, actions, ACTION_KEYS)
    else:
        with pytest.raises(ValueError, match="start"):
            workflows._continuous_replay(cfg, ds, actions, ACTION_KEYS)
    result = json.loads((tmp_path / "result.json").read_text())
    assert calls[0] == "connect" and calls[-1] == "disconnect" and "estop" not in calls
    assert ("enable" in calls) == at_start
    assert result["status"] == ("targets_sent_final_joints_confirmed" if at_start else "failed")
