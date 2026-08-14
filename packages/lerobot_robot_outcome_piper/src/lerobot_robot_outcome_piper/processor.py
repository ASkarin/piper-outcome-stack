"""Xbox Cartesian increments to the one canonical seven-value PiPER action."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotActionProcessorStep,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.types import RobotAction, TransitionKey

from .errors import OutcomePiperValidationError
from .input_safety import request_input_emergency_stop
from .safety import ACTION_KEYS, JOINT_KEYS, MotionSafety
from .teleoperator import RAW_ACTION_KEYS


class OutcomePiperAction(dict[str, float]):
    """Canonical action values plus process-local execution intent.

    The inherited mapping is deliberately limited to the seven public dataset
    fields.  ``execute_motion`` is an attribute, not a mapping item, so the
    official recorder continues to serialize exactly the canonical schema.
    """

    __slots__ = ("execute_motion",)

    def __init__(self, values: dict[str, float], *, execute_motion: bool) -> None:
        super().__init__(values)
        self.execute_motion = execute_motion


def _angle_delta(actual: float, target: float) -> float:
    return math.atan2(math.sin(actual - target), math.cos(actual - target))


@ProcessorStepRegistry.register("outcome_piper_xbox_to_joint_action")
@dataclass
class OutcomePiperXboxProcessor(RobotActionProcessorStep):
    safety: MotionSafety
    max_xyz_step_m: float
    max_yaw_step_rad: float
    max_gripper_step_m: float
    ik_max_nfev: int
    ik_timeout_s: float
    ik_residual_tolerance: float
    ik_min_singular_value: float
    _locked_roll_pitch: tuple[float, float] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.safety, dict):
            values = dict(self.safety)
            for key in (
                "joint_lower",
                "joint_upper",
                "max_joint_step",
                "workspace_lower",
                "workspace_upper",
            ):
                values[key] = tuple(values[key])
            self.safety = MotionSafety(**values)

    def get_config(self) -> dict[str, Any]:
        """Return the complete JSON-safe constructor configuration for LeRobot reloads."""

        return {
            "safety": {
                "joint_lower": list(self.safety.joint_lower),
                "joint_upper": list(self.safety.joint_upper),
                "max_joint_step": list(self.safety.max_joint_step),
                "gripper_lower": self.safety.gripper_lower,
                "gripper_upper": self.safety.gripper_upper,
                "max_gripper_step": self.safety.max_gripper_step,
                "workspace_lower": list(self.safety.workspace_lower),
                "workspace_upper": list(self.safety.workspace_upper),
                "feedback_timeout_s": self.safety.feedback_timeout_s,
                "watchdog_timeout_s": self.safety.watchdog_timeout_s,
                "motion_speed_percent": self.safety.motion_speed_percent,
                "gripper_force_n": self.safety.gripper_force_n,
                "stop_strategy": self.safety.stop_strategy,
            },
            "max_xyz_step_m": self.max_xyz_step_m,
            "max_yaw_step_rad": self.max_yaw_step_rad,
            "max_gripper_step_m": self.max_gripper_step_m,
            "ik_max_nfev": self.ik_max_nfev,
            "ik_timeout_s": self.ik_timeout_s,
            "ik_residual_tolerance": self.ik_residual_tolerance,
            "ik_min_singular_value": self.ik_min_singular_value,
        }

    def _solve(self, current: list[float], target_pose: list[float]) -> list[float]:
        import numpy as np
        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
        from scipy.optimize import least_squares

        mdh = list(get_mdh("piper"))

        def residual(q: Any) -> Any:
            pose = fk_from_mdh(mdh, q.tolist())
            return np.array(
                [
                    pose[0] - target_pose[0],
                    pose[1] - target_pose[1],
                    pose[2] - target_pose[2],
                    _angle_delta(pose[3], target_pose[3]),
                    _angle_delta(pose[4], target_pose[4]),
                    _angle_delta(pose[5], target_pose[5]),
                ],
                dtype=float,
            )

        started = time.monotonic()
        timed_out = False

        def bounded_residual(q: Any) -> Any:
            nonlocal timed_out
            if time.monotonic() - started > self.ik_timeout_s:
                timed_out = True
                raise TimeoutError("IK exceeded its frozen time budget")
            return residual(q)

        try:
            result = least_squares(
                bounded_residual,
                current,
                bounds=(self.safety.joint_lower, self.safety.joint_upper),
                method="trf",
                max_nfev=self.ik_max_nfev,
            )
        except TimeoutError as exc:
            raise OutcomePiperValidationError("IK exceeded its frozen time budget") from exc
        norm = float(np.linalg.norm(residual(result.x)))
        solution = [float(value) for value in result.x]
        if timed_out or time.monotonic() - started > self.ik_timeout_s:
            raise OutcomePiperValidationError("IK exceeded its frozen time budget")
        if not result.success or not all(math.isfinite(value) for value in solution):
            raise OutcomePiperValidationError("IK did not converge to a finite solution")
        singular_values = np.linalg.svd(result.jac, compute_uv=False)
        if time.monotonic() - started > self.ik_timeout_s:
            raise OutcomePiperValidationError("IK exceeded its frozen time budget")
        if (
            len(singular_values) != 6
            or not np.all(np.isfinite(singular_values))
            or float(singular_values[-1]) < self.ik_min_singular_value
        ):
            raise OutcomePiperValidationError("IK solution is singular")
        if norm > self.ik_residual_tolerance:
            raise OutcomePiperValidationError(
                f"IK residual {norm:.6g} exceeds {self.ik_residual_tolerance:.6g}"
            )
        for index, (target, initial, step) in enumerate(
            zip(solution, current, self.safety.max_joint_step, strict=True), start=1
        ):
            if abs(target - initial) > step:
                raise OutcomePiperValidationError(f"IK joint_{index} exceeds frozen step limit")
        return solution

    def action(self, action: RobotAction) -> RobotAction:
        try:
            return self._action(action)
        except Exception as exc:
            request_input_emergency_stop(exc)
            raise

    def _action(self, action: RobotAction) -> RobotAction:
        if set(action) != set(RAW_ACTION_KEYS):
            raise OutcomePiperValidationError(
                "Xbox action does not match the frozen six-field schema"
            )
        try:
            deltas = [float(action[key]) for key in RAW_ACTION_KEYS[:-1]]
        except (TypeError, ValueError) as exc:
            raise OutcomePiperValidationError("Xbox deltas must be numeric") from exc
        if not all(math.isfinite(value) for value in deltas):
            raise OutcomePiperValidationError("Xbox deltas must be finite")
        if type(action["hold"]) is not bool:
            raise OutcomePiperValidationError("Xbox hold must be boolean")
        if any(abs(value) > self.max_xyz_step_m for value in deltas[:3]):
            raise OutcomePiperValidationError("Xbox XYZ delta exceeds the measured step limit")
        if abs(deltas[3]) > self.max_yaw_step_rad:
            raise OutcomePiperValidationError("Xbox yaw delta exceeds the measured step limit")
        if abs(deltas[4]) > self.max_gripper_step_m:
            raise OutcomePiperValidationError("Xbox gripper delta exceeds the measured step limit")
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict) or any(key not in observation for key in ACTION_KEYS):
            raise OutcomePiperValidationError("current seven-value PiPER observation is required")
        current = [float(observation[key]) for key in JOINT_KEYS]
        current_gripper = float(observation["gripper.pos"])
        if not all(math.isfinite(value) for value in (*current, current_gripper)):
            raise OutcomePiperValidationError("PiPER observation must be finite")
        if any(
            not lower <= value <= upper
            for value, lower, upper in zip(
                current,
                self.safety.joint_lower,
                self.safety.joint_upper,
                strict=True,
            )
        ):
            raise OutcomePiperValidationError("PiPER observation is outside frozen joint limits")
        if not self.safety.gripper_lower <= current_gripper <= self.safety.gripper_upper:
            raise OutcomePiperValidationError("PiPER observation is outside frozen gripper limits")
        if not action["hold"]:
            return OutcomePiperAction(
                {
                    **{key: value for key, value in zip(JOINT_KEYS, current, strict=True)},
                    "gripper.pos": current_gripper,
                },
                execute_motion=False,
            )

        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

        pose = fk_from_mdh(list(get_mdh("piper")), current)
        if self._locked_roll_pitch is None:
            self._locked_roll_pitch = (pose[3], pose[4])
        target_pose = [
            pose[0] + deltas[0],
            pose[1] + deltas[1],
            pose[2] + deltas[2],
            self._locked_roll_pitch[0],
            self._locked_roll_pitch[1],
            pose[5] + deltas[3],
        ]
        if any(
            not lower <= value <= upper
            for value, lower, upper in zip(
                target_pose[:3],
                self.safety.workspace_lower,
                self.safety.workspace_upper,
                strict=True,
            )
        ):
            raise OutcomePiperValidationError("Xbox target is outside the frozen workspace")
        target_joints = self._solve(current, target_pose)
        target_gripper = current_gripper + deltas[4]
        if not self.safety.gripper_lower <= target_gripper <= self.safety.gripper_upper:
            raise OutcomePiperValidationError("Xbox gripper target is outside frozen limits")
        if abs(target_gripper - current_gripper) > self.safety.max_gripper_step:
            raise OutcomePiperValidationError("Xbox gripper target exceeds frozen step limit")
        return OutcomePiperAction(
            {
                **{key: value for key, value in zip(JOINT_KEYS, target_joints, strict=True)},
                "gripper.pos": target_gripper,
            },
            execute_motion=True,
        )

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for key in RAW_ACTION_KEYS:
            action_features.pop(key, None)
        for key in ACTION_KEYS:
            action_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features

    def reset(self) -> None:
        self._locked_roll_pitch = None


def make_xbox_processor(
    safety: MotionSafety,
    *,
    max_xyz_step_m: float,
    max_yaw_step_rad: float,
    max_gripper_step_m: float,
    ik_max_nfev: int,
    ik_timeout_s: float,
    ik_residual_tolerance: float,
    ik_min_singular_value: float,
) -> RobotProcessorPipeline[tuple[RobotAction, dict[str, Any]], RobotAction]:
    return RobotProcessorPipeline(
        steps=[
            OutcomePiperXboxProcessor(
                safety=safety,
                max_xyz_step_m=max_xyz_step_m,
                max_yaw_step_rad=max_yaw_step_rad,
                max_gripper_step_m=max_gripper_step_m,
                ik_max_nfev=ik_max_nfev,
                ik_timeout_s=ik_timeout_s,
                ik_residual_tolerance=ik_residual_tolerance,
                ik_min_singular_value=ik_min_singular_value,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
