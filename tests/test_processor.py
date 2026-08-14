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
        "max_yaw_step_rad": 0.02,
        "max_gripper_step_m": 0.004,
        "ik_max_nfev": 200,
        "ik_timeout_s": 1.0,
        "ik_residual_tolerance": 1e-7,
        "ik_min_singular_value": 1e-8,
    }
    values.update(overrides)
    return OutcomePiperXboxProcessor(**values)


def observation(joints: list[float], gripper: float = 0.03) -> dict[str, float]:
    return {
        **{key: value for key, value in zip(JOINT_KEYS, joints, strict=True)},
        "gripper.pos": gripper,
    }


def raw_action(*, dx: float = 0.0, yaw: float = 0.0) -> dict[str, float | bool]:
    return {
        "delta_x": dx,
        "delta_y": 0.0,
        "delta_z": 0.0,
        "delta_yaw": yaw,
        "delta_gripper": 0.0,
        "hold": True,
    }


def require_sdk_kinematics():
    return pytest.importorskip("pyAgxArm.utiles.mdh_kinematics")


def test_processor_save_load_round_trip_preserves_frozen_configuration(tmp_path: Path):
    original = make_xbox_processor(
        safety(),
        max_xyz_step_m=0.01,
        max_yaw_step_rad=0.02,
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
        (SimpleNamespace(success=True, x=np.full(6, 1.1), jac=np.eye(6)), "step limit"),
    ],
)
def test_ik_rejects_failed_singular_or_discontinuous_solution(monkeypatch, result, message):
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

    clock = iter((0.0, 0.0, 0.2))
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
