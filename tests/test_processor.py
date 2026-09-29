from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lerobot")

PLUGIN_SRC = Path(__file__).parents[1] / "packages" / "lerobot_robot_outcome_piper" / "src"
sys.path.insert(0, str(PLUGIN_SRC))

from lerobot.processor import RobotProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.types import TransitionKey  # noqa: E402
from lerobot_robot_outcome_piper.errors import OutcomePiperValidationError  # noqa: E402
from lerobot_robot_outcome_piper.processor import (  # noqa: E402
    OutcomePiperXboxProcessor,
    make_xbox_processor,
)
from lerobot_robot_outcome_piper.safety import ACTION_KEYS, JOINT_KEYS, MotionSafety  # noqa: E402


def safety(*, joint_step: float = 1.0) -> MotionSafety:
    return MotionSafety(
        joint_lower=(-2.0,) * 6,
        joint_upper=(2.0,) * 6,
        max_joint_step=(joint_step,) * 6,
        gripper_lower=0.0,
        gripper_upper=0.08,
        max_gripper_step=0.01,
        workspace_lower=(-1.0, -1.0, -1.0),
        workspace_upper=(1.0, 1.0, 1.0),
        feedback_timeout_s=0.2,
        watchdog_timeout_s=1.0,
        motion_speed_percent=5,
        gripper_force_n=0.5,
        stop_strategy="electronic_emergency_stop",
    )


def processor(**overrides: float | int | MotionSafety) -> OutcomePiperXboxProcessor:
    values = {
        "safety": safety(),
        "max_xyz_step_m": 0.01,
        "max_rotation_step_rad": 0.02,
        "max_gripper_step_m": 0.004,
        "ik_max_nfev": 200,
        "ik_timeout_s": 1.0,
        "ik_residual_tolerance": 1e-7,
        "ik_min_singular_value": 1e-8,
    }
    values.update(overrides)
    step = OutcomePiperXboxProcessor(**values)
    from lerobot_robot_outcome_piper.teleop_control import TranslationStrategy

    step.control.translation_strategy = TranslationStrategy.FIXED_ORIENTATION
    step.control.confirm_hold()
    step.control.observe(False, True)
    step.control.observe(True, True)
    return step


def observation(joints: list[float], gripper: float = 0.03) -> dict[str, float]:
    return {
        **{key: value for key, value in zip(JOINT_KEYS, joints, strict=True)},
        "gripper.pos": gripper,
    }


def raw_action(*, dx: float = 0.0, yaw: float = 0.0) -> dict[str, float | bool]:
    return {
        "stick_x": dx / 0.01,
        "stick_y": 0.0,
        "stick_z": 0.0,
        "stick_yaw": yaw / 0.02,
        "left_trigger": 0.0,
        "right_trigger": 0.0,
        "hold": True,
        "neutral": dx == 0 and yaw == 0,
        "emergency_stop": False,
        "mode_switch": False,
        "translation_switch": False,
        "home": False,
        "work": False,
    }


def require_sdk_kinematics():
    return pytest.importorskip("pyAgxArm.utiles.mdh_kinematics")


def test_processor_save_load_round_trip_preserves_frozen_configuration(tmp_path: Path):
    original = make_xbox_processor(
        safety(),
        max_xyz_step_m=0.01,
        max_rotation_step_rad=0.02,
        max_gripper_step_m=0.004,
        ik_max_nfev=20,
        ik_timeout_s=0.1,
        ik_residual_tolerance=0.001,
        ik_min_singular_value=0.001,
    )
    config_filename = "outcome_piper_xbox_processor.json"
    original.save_pretrained(tmp_path, config_filename=config_filename)

    saved = json.loads((tmp_path / config_filename).read_text(encoding="utf-8"))
    assert saved["steps"][0]["config"]["safety"]["joint_lower"] == [-2.0] * 6
    loaded = RobotProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=config_filename,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    assert isinstance(loaded.steps[0], OutcomePiperXboxProcessor)
    assert loaded.steps[0].safety == safety()
    assert loaded.steps[0].get_config() == original.steps[0].get_config()


def test_standard_piper_mdh_and_fk_golden():
    kinematics = require_sdk_kinematics()
    mdh = list(kinematics.get_mdh("piper"))

    assert np.asarray(mdh) == pytest.approx(
        np.asarray(
            [
                (0.123, 0.0, 0.0, 0.0),
                (0.0, 0.0, -math.pi / 2, -3.0058060377846343),
                (0.0, 0.28503, 0.0, -1.793849405199772),
                (0.25075, -0.02198, math.pi / 2, 0.0),
                (0.0, 0.0, -math.pi / 2, 0.0),
                (0.091, 0.0, math.pi / 2, 0.0),
            ]
        ),
        abs=1e-12,
    )
    assert kinematics.fk_from_mdh(mdh, [0.0] * 6) == pytest.approx(
        [0.0561275121646699, 0.0, 0.213266268101552, 0.0, 1.48352986419518, 0.0],
        abs=1e-12,
    )


def test_fk_ik_fk_round_trip_and_continuity():
    kinematics = require_sdk_kinematics()
    mdh = list(kinematics.get_mdh("piper"))
    solve = processor(ik_min_singular_value=1e-10)
    current = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
    target = [0.105, -0.195, 0.295, -0.095, 0.195, -0.295]
    target_pose = kinematics.fk_from_mdh(mdh, target)

    solution = solve._solve(current, target_pose)
    solved_pose = kinematics.fk_from_mdh(mdh, solution)

    assert solved_pose == pytest.approx(target_pose, abs=1e-7)
    assert max(abs(value - initial) for value, initial in zip(solution, current, strict=True)) < 0.1


def test_processor_locks_roll_pitch_while_applying_yaw(monkeypatch):
    kinematics = require_sdk_kinematics()
    joints = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
    initial_pose = kinematics.fk_from_mdh(list(kinematics.get_mdh("piper")), joints)
    step = processor()
    step._current_transition = {TransitionKey.OBSERVATION: observation(joints)}
    captured: list[list[float]] = []

    def fake_solve(current: list[float], target_pose: list[float]) -> list[float]:
        captured.append(target_pose)
        return current

    monkeypatch.setattr(step, "_solve", fake_solve)
    step.action(raw_action(yaw=0.01))
    step._current_transition = {TransitionKey.OBSERVATION: observation(joints)}
    step.action(raw_action(yaw=-0.01))

    assert captured[0][3:5] == pytest.approx(initial_pose[3:5])
    assert captured[1][3:5] == pytest.approx(initial_pose[3:5])
    assert captured[0][5] == pytest.approx(initial_pose[5] + 0.01)
    assert captured[1][5] == pytest.approx(initial_pose[5] - 0.01)


@pytest.mark.parametrize(
    "result, message",
    [
        (SimpleNamespace(success=False, x=np.zeros(6), jac=np.eye(6)), "did not converge"),
        (SimpleNamespace(success=True, x=np.zeros(6), jac=np.diag([1, 1, 1, 1, 1, 0])), "singular"),
    ],
)
def test_ik_rejects_failed_or_singular_solution(monkeypatch, result, message):
    require_sdk_kinematics()
    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "least_squares", lambda *args, **kwargs: result)
    step = processor(safety=safety(joint_step=1.0), ik_residual_tolerance=100.0)

    with pytest.raises(OutcomePiperValidationError, match=message):
        step._solve([0.0] * 6, [0.0] * 6)


def test_ik_unreachable_residual_and_time_budget_fail(monkeypatch):
    require_sdk_kinematics()
    import scipy.optimize
    import lerobot_robot_outcome_piper.processor as processor_module

    unreachable = SimpleNamespace(success=True, x=np.zeros(6), jac=np.eye(6))
    monkeypatch.setattr(scipy.optimize, "least_squares", lambda *args, **kwargs: unreachable)
    with pytest.raises(OutcomePiperValidationError, match="residual"):
        processor(ik_residual_tolerance=1e-12)._solve([0.0] * 6, [10.0] * 6)

    clock = iter((0.0, 0.0, 0.2, 0.2))
    monkeypatch.setattr(processor_module.time, "monotonic", lambda: next(clock))
    with pytest.raises(OutcomePiperValidationError, match="time budget"):
        processor(ik_timeout_s=0.1)._solve([0.0] * 6, [0.0] * 6)


def test_processor_failure_path_never_calls_sdk(monkeypatch):
    require_sdk_kinematics()
    step = processor()
    step._current_transition = {TransitionKey.OBSERVATION: observation([0.0] * 6)}
    sdk_calls: list[dict[str, float]] = []
    monkeypatch.setattr(
        step,
        "_solve",
        lambda *_: (_ for _ in ()).throw(OutcomePiperValidationError("IK failed")),
    )

    with pytest.raises(OutcomePiperValidationError, match="IK failed"):
        action = step.action(raw_action(dx=0.001))
        sdk_calls.append({key: float(action[key]) for key in ACTION_KEYS})
    assert sdk_calls == []


def test_resume_uses_latest_feedback_preserves_roll_pitch_and_grasp(monkeypatch):
    step = processor()
    current = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
    step._current_transition = {TransitionKey.OBSERVATION: observation(current)}
    calls = []

    def solve(q, pose):
        calls.append((q[:], pose[:]))
        return q[:]

    monkeypatch.setattr(step, "_solve", solve)
    step.action(raw_action(dx=0.001))
    locked = step.control.orientation_target
    step.control.gripper_target = 0.05
    step.action({**raw_action(), "hold": False})
    from lerobot_robot_outcome_piper.teleop_control import TranslationStrategy

    step.control.translation_strategy = TranslationStrategy.FIXED_ORIENTATION
    step.control.confirm_hold()
    latest = [v + 0.001 for v in current]
    step._current_transition = {TransitionKey.OBSERVATION: observation(latest, gripper=0.03)}
    assert step.action(raw_action()).intent == "center"
    result = step.action(raw_action(dx=0.001))
    assert result["gripper.pos"] == 0.05
    assert calls[-1][0] == latest and np.allclose(
        __import__("scipy")
        .spatial.transform.Rotation.from_euler("xyz", calls[-1][1][3:])
        .as_matrix(),
        locked,
    )
    assert set(result) == set(ACTION_KEYS)


def test_processor_b_preempts_invalid_numeric_input():
    step = processor()
    with pytest.raises(OutcomePiperValidationError, match="emergency stop requested"):
        step.action(
            {
                **raw_action(),
                "emergency_stop": True,
                "mode_switch": False,
                "translation_switch": False,
                "stick_x": "invalid",
            }
        )


def test_expected_ik_rejection_pauses_without_emergency_stop(monkeypatch):
    from lerobot_robot_outcome_piper.errors import OutcomePiperIntentRejected
    from lerobot_robot_outcome_piper.teleop_control import TeleopState

    step = processor()
    step._current_transition = {TransitionKey.OBSERVATION: observation([0.1] * 6)}
    monkeypatch.setattr(
        step,
        "_solve",
        lambda *_: (_ for _ in ()).throw(OutcomePiperIntentRejected("unreachable target")),
    )
    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.processor.request_input_emergency_stop",
        lambda *_: pytest.fail("unexpected electronic stop"),
    )
    action = step.action(raw_action(dx=0.001))
    assert action.intent == "hold" and action.rejection_reason == "unreachable target"
    assert step.control.state is TeleopState.HOLD_REQUESTED
    assert set(action) == set(ACTION_KEYS)


def test_gripper_feedback_above_command_bound_can_be_commanded_back_inside(monkeypatch):
    step = processor()
    step._current_transition = {TransitionKey.OBSERVATION: observation([0.1] * 6, gripper=0.081)}
    monkeypatch.setattr(step, "_solve", lambda q, target: q)
    result = step.action({**raw_action(), "left_trigger": 1.0, "right_trigger": 0.0})
    assert result["gripper.pos"] == pytest.approx(0.077)


def test_ik_timeout_requests_hold_fault_not_electronic_stop(monkeypatch):
    from lerobot_robot_outcome_piper.errors import OutcomePiperControlTimeout

    step = processor()
    step._current_transition = {TransitionKey.OBSERVATION: observation([0.1] * 6)}
    calls = []
    monkeypatch.setattr(
        step, "_solve", lambda *_: (_ for _ in ()).throw(OutcomePiperControlTimeout("IK timeout"))
    )
    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.processor.request_input_fault_hold",
        lambda e: calls.append(str(e)),
    )
    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.processor.request_input_emergency_stop",
        lambda *_: pytest.fail("unexpected electronic stop"),
    )
    with pytest.raises(OutcomePiperControlTimeout):
        step.action(raw_action(dx=0.001))
    assert calls == ["IK timeout"]


@pytest.mark.parametrize("step_deg, first_segments", [(1, 1), (5, 1)])
def test_real_pause_pose_correction_uses_bounded_waypoints(step_deg, first_segments):
    from dataclasses import replace

    k = require_sdk_kinematics()
    initial = list(map(math.radians, [24.962, 44.976, -29.958, 59.998, -9.990, -33.963]))
    current = list(map(math.radians, [25.120, 44.976, -29.958, 60.491, -9.984, -34.409]))
    step = processor(
        safety=replace(
            safety(joint_step=math.radians(step_deg)),
            joint_lower=tuple(map(math.radians, [-150, 0, -170, -100, -70, -180])),
            joint_upper=tuple(map(math.radians, [150, 180, 0, 100, 70, 180])),
        )
    )
    pose = k.fk_from_mdh(list(k.get_mdh("piper")), initial)
    step.control.initialize_orientation(
        __import__("scipy").spatial.transform.Rotation.from_euler("xyz", pose[3:]).as_matrix()
    )
    for tick in range(4):
        step._current_transition = {TransitionKey.OBSERVATION: observation(current)}
        result = step.action(raw_action(dx=1e-6))
        assert result.intent == "run" and set(result) == set(ACTION_KEYS)
        nxt = [result[key] for key in JOINT_KEYS]
        assert max(abs(a - b) for a, b in zip(nxt, current)) <= math.radians(step_deg)
        if tick == 0:
            assert result.joint_plan["segments"] == first_segments
        current = nxt
    final_pose = k.fk_from_mdh(list(k.get_mdh("piper")), current)
    assert final_pose[3:5] == pytest.approx(pose[3:5], abs=1e-6)
    # Pausing must discard any correction plan, while retaining the session lock.
    result = step.action({**raw_action(), "hold": False})
    assert result.intent != "run" and result.joint_plan is None
    assert np.asarray(step.control.orientation_target) == pytest.approx(
        __import__("scipy").spatial.transform.Rotation.from_euler("xyz", pose[3:]).as_matrix()
    )


@pytest.mark.parametrize(
    "current,target,allowed",
    [
        ((0, -0.1, 0), (0, -0.05, 0), True),
        ((0, -0.1, 0), (0, -0.11, 0), False),
        ((0, -0.1, 0), (0.1, -0.1, 0), False),
        ((-0.1, -0.1, 0), (-0.11, -0.05, 0), False),
        ((0, -0.1, 0), (0, 1.01, 0), False),
        ((0, 1.1, 0), (0, 1.05, 0), True),
        ((0, 0.5, 0), (0, -0.01, 0), False),
    ],
)
def test_workspace_reentry_is_inward_on_every_violated_axis(current, target, allowed):
    from lerobot_robot_outcome_piper.safety import workspace_step_allowed

    assert (
        workspace_step_allowed(current, target, (0, 0, 0), (1, 1, 1), allow_reentry=True) is allowed
    )
    assert not workspace_step_allowed(current, target, (0, 0, 0), (1, 1, 1))


def test_outside_workspace_neutral_rearm_then_inward_motion():
    from dataclasses import replace

    k = require_sdk_kinematics()
    current = list(map(math.radians, [19.527, 44.873, -29.851, 54.984, -6.323, -28.906]))
    pose = k.fk_from_mdh(list(k.get_mdh("piper")), current)
    step = processor(
        safety=replace(
            safety(joint_step=math.radians(5)), workspace_lower=(-1, pose[1] + 0.0006, -1)
        )
    )
    step._current_transition = {TransitionKey.OBSERVATION: observation(current)}
    neutral = step.action(raw_action())
    assert neutral.intent == "center" and neutral.joint_plan is None
    assert not step.control.permits(neutral.epoch)
    for _ in range(3):
        step._current_transition = {TransitionKey.OBSERVATION: observation(current)}
        action = step.action({**raw_action(), "stick_y": 0.025, "neutral": False})
        assert action.intent == "run"
        nxt = [action[key] for key in JOINT_KEYS]
        nxt_pose = k.fk_from_mdh(list(k.get_mdh("piper")), nxt)
        assert nxt_pose[1] > pose[1]
        current, pose = nxt, nxt_pose
    assert pose[1] >= step.safety.workspace_lower[1]
