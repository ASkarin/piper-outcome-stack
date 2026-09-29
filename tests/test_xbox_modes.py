import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from lerobot.types import TransitionKey
from lerobot_robot_outcome_piper.teleop_control import (
    TeleopControl,
    TeleopMode,
    TeleopState,
    TranslationStrategy,
)
from lerobot_robot_outcome_piper.processor import OutcomePiperAction, input_deltas
from test_processor import processor, raw_action, observation, require_sdk_kinematics
from test_plugin import valid_action
from test_xbox_pause import session as base_session, settle  # noqa: F401


@pytest.fixture
def session(base_session):  # noqa: F811
    return base_session


def test_rb_edges_require_confirmed_neutral_pause_and_never_queue():
    c = TeleopControl()
    c.observe(False, True, True)  # held at startup
    c.confirm_hold()
    c.observe(False, True, True)
    assert c.mode is TeleopMode.TRANSLATION
    c.observe(False, True, False)
    c.observe(False, True, True)
    assert c.mode is TeleopMode.ORIENTATION and c.mode_event["accepted"]
    c.observe(False, True, True)
    assert c.mode_event is None
    c.observe(True, True, False)
    assert c.state is TeleopState.RUNNING
    c.observe(True, True, True)
    assert not c.mode_event["accepted"] and c.mode is TeleopMode.ORIENTATION
    c.observe(False, True, True)
    c.confirm_hold()
    c.observe(False, True, True)
    assert c.mode is TeleopMode.ORIENTATION
    c.observe(False, False, False)
    c.observe(False, False, True)
    assert not c.mode_event["accepted"]
    c.observe(False, True, False)
    c.observe(False, True, True)
    assert c.mode is TeleopMode.TRANSLATION


def test_mapping_combined_rotation_and_unassigned_axis():
    a = dict(
        stick_x=1.0, stick_y=1.0, stick_z=1.0, stick_yaw=1.0, left_trigger=0.0, right_trigger=1.0
    )
    xyz, rv, g = input_deltas(
        a, TeleopMode.TRANSLATION, 0.001, 0.01, 0.002, TranslationStrategy.FIXED_ORIENTATION
    )
    assert xyz == [0.001] * 3 and rv == [0, 0, 0.01] and g == 0.002
    xyz, rv, g = input_deltas(a, TeleopMode.ORIENTATION, 0.001, 0.01, 0.002)
    assert xyz == [0] * 3 and np.linalg.norm(rv) == pytest.approx(0.01)
    assert rv == pytest.approx([0.01 / math.sqrt(3)] * 3)


@pytest.mark.parametrize(
    "rpy",
    [
        [0.2, 0.3, math.pi - 0.0001],
        [0.1, math.pi / 2 - 0.0001, 0.4],
        [0.2, math.pi / 2 + 0.0001, 0.3],
    ],
)
def test_base_axis_composition_crosses_euler_boundaries_without_committing(monkeypatch, rpy):
    k = require_sdk_kinematics()
    p = processor()
    p.control.mode = TeleopMode.ORIENTATION
    monkeypatch.setattr(k, "fk_from_mdh", lambda *_: [0.1, 0.1, 0.3, *rpy])
    captured = []
    monkeypatch.setattr(p, "_solve", lambda q, target: captured.append(target) or q)
    p._current_transition = {TransitionKey.OBSERVATION: observation([0.1] * 6)}
    result = p.action({**raw_action(), "stick_x": 1.0, "neutral": False})
    old = Rotation.from_euler("xyz", rpy)
    expected = (Rotation.from_rotvec([0.02, 0, 0]) * old).as_matrix()
    assert np.asarray(result.orientation_target) == pytest.approx(
        old.as_matrix()
    )  # Fake FK returns unchanged actual waypoint.
    assert Rotation.from_euler("xyz", captured[-1][3:]).as_matrix() == pytest.approx(expected)
    assert np.asarray(p.control.orientation_target) == pytest.approx(old.as_matrix())


@pytest.mark.parametrize(
    "outcome", ["success", "release", "failure", "gripper_failure", "stale", "B"]
)
def test_orientation_commit_only_after_valid_success(session, outcome):
    robot, arm, c, clock = session
    settle(session)
    old = np.array(c.orientation_target)
    _, epoch = c.observe(True, True)
    action = OutcomePiperAction(valid_action(0.01, 0.035), intent="run", epoch=epoch)
    action.generated_monotonic_s = clock.now
    goal = (Rotation.from_rotvec([0.01, 0, 0]) * Rotation.from_matrix(old)).as_matrix()
    action.orientation_target = goal.tolist()
    if outcome == "release":
        original = arm.move_j

        def released(q):
            original(q)
            c.request_hold()

        arm.move_j = released
    elif outcome == "failure":
        arm.move_j = lambda _: (_ for _ in ()).throw(OSError("failed"))
    elif outcome == "gripper_failure":
        arm.gripper.move_gripper_m = lambda *a, **kw: (_ for _ in ()).throw(OSError("failed"))
    elif outcome == "stale":
        c.request_hold()
    elif outcome == "B":
        robot.request_emergency_stop("B")
    if outcome in ("failure", "gripper_failure", "B"):
        with pytest.raises(RuntimeError):
            robot.send_action(action)
    else:
        robot.send_action(action)
    assert np.asarray(c.orientation_target) == pytest.approx(goal if outcome == "success" else old)
    if outcome == "success":
        assert robot.last_action_telemetry["orientation_committed"]
        assert robot.last_action_telemetry["teleop_mode"] == "TRANSLATION"


def test_switch_preserves_target_b_preempts_mapping_and_enable_resets():
    p = processor()
    p.control.initialize_orientation(np.eye(3))
    before = p.control.orientation_target
    with pytest.raises(RuntimeError):
        p.action(
            {**raw_action(), "emergency_stop": True, "mode_switch": True, "stick_x": "invalid"}
        )
    assert p.control.state is TeleopState.E_STOP and p.control.orientation_target == before
    c = TeleopControl()
    c.initialize_orientation(np.eye(3))
    c.confirm_hold()
    c.observe(False, True, False)
    c.observe(False, True, True)
    assert c.orientation_target == before
    c.prepare_enable()
    assert c.orientation_target is None and c.mode is TeleopMode.TRANSLATION


def test_input_preview_uses_measured_rb_without_robot_or_ik(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from piper_outcome_stack.ops import xbox_preview
    from test_plugin import xbox_config
    from lerobot_robot_outcome_piper import teleoperator, robot

    cfg = asdict(xbox_config())
    cfg.pop("mode_switch_button")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"teleop": cfg}))
    report = tmp_path / "buttons.json"
    report.write_text(
        json.dumps(
            {
                "status": "input_measured",
                "device": {"guid": "measured-guid"},
                "buttons": {"hold_button": 1, "emergency_stop_button": 0, "mode_switch_button": 2},
            }
        )
    )
    assert xbox_preview.preview_config(path, report).mode_switch_button == 2
    zero = {
        k: 0.0
        for k in ("stick_x", "stick_y", "stick_z", "stick_yaw", "left_trigger", "right_trigger")
    }
    neutral = {
        **zero,
        "hold": False,
        "neutral": True,
        "emergency_stop": False,
        "mode_switch": False,
        "translation_switch": False,
        "home": False,
        "work": False,
    }
    sequence = iter(
        [
            neutral,
            {**neutral, "mode_switch": True},
            neutral,
            {**neutral, "hold": True},
            {**neutral, "hold": True, "neutral": False, "stick_x": 0.5},
            {**neutral, "emergency_stop": True},
        ]
    )

    class Input:
        def __init__(self, cfg):
            pass

        def connect(self):
            pass

        def get_action(self):
            return next(sequence)

        def disconnect(self):
            pass

    monkeypatch.setattr(teleoperator, "OutcomePiperXbox", Input)
    monkeypatch.setattr(
        robot.OutcomePiper, "__init__", lambda *a, **k: pytest.fail("Robot constructed")
    )
    monkeypatch.setattr(xbox_preview.time, "sleep", lambda _: None)
    output = tmp_path / "preview"
    assert (
        xbox_preview.main(
            ["--config", str(path), "--buttons-report", str(report), "--output", str(output)]
        )
        == 0
    )
    events = [json.loads(s) for s in (output / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "B"
    assert any(e.get("mode_event", {}) and e["mode_event"]["accepted"] for e in events)
    assert events[-2]["mode"] == "ORIENTATION"
    assert events[-2]["rotation_vector_rad"] == pytest.approx([0.01, 0, 0])
    assert all(e["simulated_hold"] for e in events)


@pytest.mark.parametrize(
    "mode,key",
    [(TeleopMode.TRANSLATION, k) for k in ("stick_x", "stick_y", "stick_z")]
    + [(TeleopMode.ORIENTATION, k) for k in ("stick_x", "stick_y", "stick_yaw")],
)
@pytest.mark.parametrize("sign", [-1, 1])
def test_six_directions_use_official_kinematics(mode, key, sign):
    k = require_sdk_kinematics()
    joints = list(map(math.radians, [25, 45, -30, 60, -10, -34]))
    p = processor(max_rotation_step_rad=math.radians(0.1))
    p.control.mode = mode
    p._current_transition = {TransitionKey.OBSERVATION: observation(joints)}
    raw = {**raw_action(), key: sign * 0.025, "neutral": False}
    xyz, rv, _ = input_deltas(raw, mode, 0.01, math.radians(0.1), 0.004)
    before = k.fk_from_mdh(list(k.get_mdh("piper")), joints)
    result = p.action(raw)
    assert result.intent == "run"
    after = k.fk_from_mdh(list(k.get_mdh("piper")), [result[f"joint_{i}.pos"] for i in range(1, 7)])
    from lerobot_robot_outcome_piper.position_ik import grasp_position

    assert grasp_position(after) == pytest.approx(grasp_position(before) + xyz, abs=1e-6)
    expected = (Rotation.from_rotvec(rv) * Rotation.from_euler("xyz", before[3:])).as_matrix()
    assert Rotation.from_euler("xyz", after[3:]).as_matrix() == pytest.approx(expected, abs=1e-6)


def test_mode_switch_changes_no_hold_target_and_sends_no_command(session):
    robot, arm, c, clock = session
    settle(session)
    before = list(arm.calls)
    target = c.orientation_target
    c.observe(False, True, False)
    intent, epoch = c.observe(False, True, True)
    action = OutcomePiperAction(robot.get_observation(), intent=intent, epoch=epoch)
    action.generated_monotonic_s = clock.now
    robot.send_action(action)
    assert arm.calls == before and c.orientation_target == target
    assert robot.last_action_telemetry["teleop_mode"] == "ORIENTATION"
    assert robot.last_action_telemetry["mode_event"]["accepted"]
