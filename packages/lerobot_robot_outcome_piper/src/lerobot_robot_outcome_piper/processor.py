"""Xbox Cartesian increments to the one canonical seven-value PiPER action."""

from __future__ import annotations

import math
import logging
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

from .stage_timing import measured
from .errors import (
    OutcomePiperValidationError,
    OutcomePiperIntentRejected,
    OutcomePiperControlTimeout,
    OutcomePiperLogError,
)
from .input_safety import request_input_emergency_stop, request_input_fault_hold
from .workspace import workspace_pose_allowed
from dataclasses import asdict
from .safety import ACTION_KEYS, JOINT_KEYS, MotionSafety, step_within_limit, check_joint_feedback
from .teleoperator import RAW_ACTION_KEYS, AXIS_KEYS, CONTROL_KEYS
from .teleop_control import TeleopControl, TeleopMode, TranslationStrategy, TeleopState


class OutcomePiperAction(dict[str, float]):
    """Canonical action values plus process-local execution intent.

    The inherited mapping is deliberately limited to the seven public dataset
    fields.  Control intent and epoch are attributes, not a mapping item, so the
    official recorder continues to serialize exactly the canonical schema.
    """

    __slots__ = (
        "intent",
        "epoch",
        "generated_monotonic_s",
        "rejection_reason",
        "joint_plan",
        "orientation_target",
        "gripper_input",
        "gripper_plan",
        "pose_plan",
        "reference_plan",
        "feedback_limit_events",
    )

    def __init__(self, values: dict[str, float], *, intent: str, epoch: int) -> None:
        super().__init__(values)
        self.intent = intent
        self.epoch = epoch
        self.generated_monotonic_s = time.monotonic()
        self.rejection_reason = None
        self.joint_plan = None
        self.orientation_target = None
        self.gripper_input = False
        self.gripper_plan = None
        self.pose_plan = None
        self.reference_plan = None
        self.feedback_limit_events = []


def input_deltas(
    action, mode, xyz_step, rotation_step, gripper_step, strategy=TranslationStrategy.WRIST_PRIORITY
):
    """Map normalized axes once; combined base-axis rotation has a bounded norm."""
    x, y, z, yaw = (action[k] for k in AXIS_KEYS[:4])
    translation = (
        [x * xyz_step, y * xyz_step, z * xyz_step]
        if mode is TeleopMode.TRANSLATION
        else [0.0, 0.0, 0.0]
    )
    rotation = (
        [0.0, 0.0, yaw if strategy is TranslationStrategy.FIXED_ORIENTATION else 0.0]
        if mode is TeleopMode.TRANSLATION
        else [x, y, yaw]
    )
    norm = math.sqrt(sum(v * v for v in rotation))
    rotation = [v * rotation_step / max(1.0, norm) for v in rotation]
    return translation, rotation, (action["right_trigger"] - action["left_trigger"]) * gripper_step


@ProcessorStepRegistry.register("outcome_piper_xbox_to_joint_action")
@dataclass
class OutcomePiperXboxProcessor(RobotActionProcessorStep):
    safety: MotionSafety
    max_xyz_step_m: float
    max_rotation_step_rad: float
    max_gripper_step_m: float
    ik_max_nfev: int
    ik_timeout_s: float
    ik_residual_tolerance: float
    ik_min_singular_value: float
    work_joint_rad: tuple[float, ...] | None = None
    work_gripper_m: float | None = None
    pose_timing: dict | None = None
    gripper_reference: dict | None = None
    streaming_reference: dict | None = None
    control: TeleopControl = field(default_factory=TeleopControl, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.streaming_reference is not None:
            from .continuous_reference import ReferenceSettings

            ReferenceSettings(**self.streaming_reference)
        if self.pose_timing is not None:
            from .joint_pose import PoseTiming

            PoseTiming(**self.pose_timing)
        if self.work_joint_rad is not None:
            self.work_joint_rad = tuple(self.work_joint_rad)
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
        if self.gripper_reference is not None:
            from .gripper_reference import GripperReferenceSettings

            settings = GripperReferenceSettings(**self.gripper_reference)
            if settings.max_lead_m > self.safety.max_gripper_step:
                raise ValueError("gripper reference lead exceeds execution step limit")

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
                "workspace_geometry": None
                if self.safety.workspace_geometry is None
                else asdict(self.safety.workspace_geometry),
            },
            "work_joint_rad": self.work_joint_rad,
            "work_gripper_m": self.work_gripper_m,
            "pose_timing": self.pose_timing,
            "streaming_reference": self.streaming_reference,
            "gripper_reference": self.gripper_reference,
            "max_xyz_step_m": self.max_xyz_step_m,
            "max_rotation_step_rad": self.max_rotation_step_rad,
            "max_gripper_step_m": self.max_gripper_step_m,
            "ik_max_nfev": self.ik_max_nfev,
            "ik_timeout_s": self.ik_timeout_s,
            "ik_residual_tolerance": self.ik_residual_tolerance,
            "ik_min_singular_value": self.ik_min_singular_value,
        }

    def _plan_gripper(self, current: float, delta: float):
        """Plan a trigger increment within the available travel, not an absolute action."""
        if self.gripper_reference is not None:
            from .gripper_reference import GripperReferenceSettings, gripper_candidate

            previous, revision = self.control.gripper_reference_snapshot(self.control.epoch)
            target = self.control.gripper_target
            target = current if target is None else target
            reference = gripper_candidate(
                previous,
                target,
                current,
                delta / self.max_gripper_step_m,
                GripperReferenceSettings(**self.gripper_reference),
                time.monotonic(),
                self.safety.gripper_lower,
                self.safety.gripper_upper,
            )
            planned = reference["target_m"]
            if not self.safety.gripper_lower <= planned <= self.safety.gripper_upper:
                raise OutcomePiperIntentRejected("Xbox gripper target is outside frozen limits")
            if planned != target and not step_within_limit(
                planned, current, self.safety.max_gripper_step
            ):
                raise OutcomePiperIntentRejected("Xbox gripper target exceeds frozen step limit")
            return planned, {
                "feedback_m": current,
                "target_m": planned,
                "base_revision": revision,
                "reference": reference,
            }
        if delta == 0:
            target = self.control.gripper_target
            return (current if target is None else target), None
        lower, upper = self.safety.gripper_lower, self.safety.gripper_upper
        requested = current + delta
        # Only shorten travel in the commanded direction. Out-of-range feedback
        # must return inside the interval within one legal step, never jump to it.
        target = max(lower, requested) if delta < 0 else min(upper, requested)
        if not lower <= target <= upper or (target - current) * delta < 0:
            raise OutcomePiperIntentRejected(
                "Xbox gripper feedback cannot reach the command interval in this direction"
            )
        if not step_within_limit(target, current, self.safety.max_gripper_step):
            raise OutcomePiperIntentRejected("Xbox gripper target exceeds frozen step limit")
        return target, {
            "feedback_m": current,
            "requested_delta_m": delta,
            "target_m": target,
            "planned_delta_m": target - current,
            "boundary": "lower" if target == lower else "upper" if target == upper else None,
            "travel_shortened": target != requested,
        }

    def _solve(self, current: list[float], target_pose: list[float]) -> list[float]:
        import numpy as np
        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        mdh = list(get_mdh("piper"))
        target_rotation = Rotation.from_euler("xyz", target_pose[3:])

        def residual(q: Any) -> Any:
            pose = fk_from_mdh(mdh, q.tolist())
            return np.array(
                [
                    pose[0] - target_pose[0],
                    pose[1] - target_pose[1],
                    pose[2] - target_pose[2],
                    *(target_rotation.inv() * Rotation.from_euler("xyz", pose[3:])).as_rotvec(),
                ],
                dtype=float,
            )

        started = time.monotonic()
        timed_out = False
        evaluations = 0

        def deadline_error():
            return OutcomePiperControlTimeout(
                f"IK exceeded its time budget: elapsed={time.monotonic() - started:.6f}s, "
                f"budget={self.ik_timeout_s:g}s, evaluations={evaluations}, "
                f"current_joint_rad={current}, target_base_xyz_rpy={list(target_pose)}"
            )

        def bounded_residual(q: Any) -> Any:
            nonlocal timed_out, evaluations
            evaluations += 1
            if time.monotonic() - started > self.ik_timeout_s:
                timed_out = True
                raise TimeoutError("IK exceeded its frozen time budget")
            return residual(q)

        try:
            result = least_squares(
                bounded_residual,
                # Only the optimizer seed is bounded. FK, step validation and
                # observations retain the original measured joint angles.
                np.clip(current, self.safety.joint_lower, self.safety.joint_upper),
                bounds=(self.safety.joint_lower, self.safety.joint_upper),
                method="trf",
                # A tiny joystick rotation can have a small gradient before
                # meeting the configured pose residual. Keep residual/step termination.
                gtol=None,
                max_nfev=self.ik_max_nfev,
            )
        except TimeoutError as exc:
            raise deadline_error() from exc
        norm = float(np.linalg.norm(residual(result.x)))
        solution = [float(value) for value in result.x]
        if timed_out or time.monotonic() - started > self.ik_timeout_s:
            raise deadline_error()
        if not all(math.isfinite(value) for value in solution):
            raise OutcomePiperValidationError("IK returned a nonfinite solution")
        if not result.success:
            raise OutcomePiperIntentRejected("IK did not converge to a finite solution")
        singular_values = np.linalg.svd(result.jac, compute_uv=False)
        if time.monotonic() - started > self.ik_timeout_s:
            raise deadline_error()
        if (
            len(singular_values) != 6
            or not np.all(np.isfinite(singular_values))
            or float(singular_values[-1]) < self.ik_min_singular_value
        ):
            raise OutcomePiperIntentRejected("IK solution is singular")
        if norm > self.ik_residual_tolerance:
            raise OutcomePiperIntentRejected(
                f"IK residual {norm:.6g} exceeds {self.ik_residual_tolerance:.6g}"
            )
        return solution

    @measured("processor")
    def action(self, action: RobotAction) -> RobotAction:
        self._feedback_limit_events = []
        try:
            if not action.get("emergency_stop", False):
                from .console import check_operator_logging

                check_operator_logging()
            result = self._action(action)
            result.feedback_limit_events = self._feedback_limit_events
            return result
        except OutcomePiperIntentRejected as exc:
            self.control.request_hold(str(exc))
            observation = self.transition[TransitionKey.OBSERVATION]
            result = OutcomePiperAction(
                {k: float(observation[k]) for k in ACTION_KEYS},
                intent="hold",
                epoch=self.control.epoch,
            )
            result.feedback_limit_events = self._feedback_limit_events
            result.rejection_reason = str(exc)
            logging.warning("Xbox input rejected; movement paused: %s", exc)
            return result
        except (OutcomePiperControlTimeout, OutcomePiperLogError) as exc:
            request_input_fault_hold(exc)
            raise
        except Exception as exc:
            request_input_emergency_stop(exc)
            raise

    def _action(self, action: RobotAction) -> RobotAction:
        from .stage_timing import mark_preparation_pose

        mark_preparation_pose(False)
        if set(action) != set(RAW_ACTION_KEYS):
            raise OutcomePiperValidationError("Xbox action does not match the input/control schema")
        if any(type(action[key]) is not bool for key in CONTROL_KEYS):
            raise OutcomePiperValidationError("Xbox control flags must be boolean")
        if action["emergency_stop"]:
            self.control.stop(True)
            request_input_emergency_stop("Xbox emergency-stop button pressed")
            raise OutcomePiperValidationError("Xbox emergency stop requested")
        try:
            axes = [float(action[key]) for key in AXIS_KEYS]
        except (TypeError, ValueError) as exc:
            raise OutcomePiperValidationError("Xbox axes must be numeric") from exc
        if not all(math.isfinite(value) for value in axes):
            raise OutcomePiperValidationError("Xbox axes must be finite")
        if any(abs(v) > 1 for v in axes[:4]) or any(not 0 <= v <= 1 for v in axes[4:]):
            raise OutcomePiperValidationError("Xbox normalized axis exceeds its range")
        intent, epoch = self.control.observe(
            action["hold"],
            action["neutral"],
            action["mode_switch"],
            action["home"],
            action["work"],
            action["translation_switch"],
        )
        xyz_delta, rotation_delta, gripper_delta = input_deltas(
            dict(zip(AXIS_KEYS, axes)),
            self.control.mode,
            self.max_xyz_step_m,
            self.max_rotation_step_rad,
            self.max_gripper_step_m,
            self.control.translation_strategy,
        )
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict) or any(key not in observation for key in ACTION_KEYS):
            raise OutcomePiperValidationError("current seven-value PiPER observation is required")
        current = [float(observation[key]) for key in JOINT_KEYS]
        current_gripper = float(observation["gripper.pos"])
        if not all(math.isfinite(value) for value in (*current, current_gripper)):
            raise OutcomePiperValidationError("PiPER observation must be finite")
        self._feedback_limit_events = check_joint_feedback(
            current,
            self.safety,
            target=self.control.joint_target,
            tolerance=0.0
            if self.control.hold_settings is None
            else self.control.hold_settings.joint_tolerance_rad,
        )
        if current_gripper < 0:
            raise OutcomePiperValidationError("PiPER gripper feedback is negative")
        if self.control.state is TeleopState.POSE_READY and self.control.hold_confirmed:
            from .joint_pose import JointPoseSequence

            mark_preparation_pose(self.control.recording_phase in (None, "preparing"))
            if self.control.pose_sequence is None:
                start = list(current)
                for event in self._feedback_limit_events:
                    start[event["joint"] - 1] = event["boundary_rad"]
                if self.control.pose_kind == "work":
                    if self.work_joint_rad is None or self.work_gripper_m is None:
                        raise OutcomePiperIntentRejected("work pose is not configured")
                    goal = [*self.work_joint_rad, self.work_gripper_m]
                else:
                    goal = [0.0] * 7
                sequence = JointPoseSequence(
                    start,
                    current_gripper,
                    self.safety,
                    self.control.hold_settings,
                    goal,
                    self.control.pose_kind,
                    timing=self.pose_timing,
                    incremental=True,
                )
                with self.control._lock:
                    if self.control.epoch == epoch and self.control.state is TeleopState.POSE_READY:
                        sequence.control_epoch = epoch
                        self.control.pose_sequence = sequence
                        logging.info("[姿态规划] 正在检查完整路径；机械臂保持当前位置。")
            sequence = self.control.pose_sequence
            if sequence is not None and sequence.control_epoch == epoch:
                if not sequence.planning_complete:
                    sequence.plan_chunk()
                    if sequence.planning_complete:
                        logging.info(
                            "[姿态规划完成] %s个航点；名义时长%s；按住LB直至到位。",
                            sequence.waypoint_count,
                            "按反馈推进"
                            if sequence.nominal_duration_s is None
                            else f"{sequence.nominal_duration_s:.2f}秒",
                        )
                result = OutcomePiperAction(
                    {**dict(zip(JOINT_KEYS, current)), "gripper.pos": current_gripper},
                    intent="hold",
                    epoch=epoch,
                )
                result.pose_plan = sequence.telemetry()
                return result
        if intent == "pose":
            mark_preparation_pose(self.control.recording_phase in (None, "preparing"))
            sequence = self.control.pose_sequence
            if (
                sequence is None
                or sequence.control_epoch != epoch
                or not sequence.planning_complete
            ):
                raise OutcomePiperIntentRejected("pose plan is no longer valid; select A/Y again")
            if sequence.window is None and sequence.index == 0:
                sequence.validate_start(current, current_gripper)
            if sequence.complete:
                self.control.pose_event = "completed"
                logging.info(
                    "Xbox 姿态到位详情：%s；夹爪目标=%.3fmm，反馈=%.3fmm",
                    "关节回零" if self.control.pose_kind == "home" else "工作姿态",
                    sequence.targets[-1][6] * 1000,
                    current_gripper * 1000,
                )
                logging.info(
                    "[关节到位] %s；正在保持。",
                    "零位" if self.control.pose_kind == "home" else "工作姿态",
                )
                self.control.request_hold()
                result = OutcomePiperAction(
                    {**dict(zip(JOINT_KEYS, current)), "gripper.pos": sequence.targets[-1][6]},
                    intent="hold",
                    epoch=self.control.epoch,
                )
            else:
                sequence.next_waypoint()
                result = OutcomePiperAction(sequence.values, intent="pose", epoch=epoch)
            result.pose_plan = sequence.telemetry()
            return result
        if intent in ("run", "center"):
            intent, epoch = self.control.arm_input_intent(
                epoch, any((*xyz_delta, *rotation_delta)), detail=dict(zip(AXIS_KEYS, axes))
            )
        if intent != "run":
            target_gripper, gripper_plan = (
                self._plan_gripper(current_gripper, gripper_delta)
                if intent == "center"
                else (current_gripper, None)
            )
            result = OutcomePiperAction(
                {**dict(zip(JOINT_KEYS, current)), "gripper.pos": target_gripper},
                intent=intent,
                epoch=epoch,
            )
            result.gripper_plan = gripper_plan
            result.gripper_input = intent == "center" and gripper_delta != 0
            return result

        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

        from scipy.spatial.transform import Rotation
        import warnings
        import numpy as np

        pose = fk_from_mdh(list(get_mdh("piper")), current)
        from .position_ik import solve_position
        from .workspace import grasp_position, grasp_offset

        wrist_priority = (
            self.control.mode is TeleopMode.TRANSLATION
            and self.control.translation_strategy is TranslationStrategy.WRIST_PRIORITY
        )
        position = grasp_position(pose, self.safety.workspace_geometry)
        observed_rotation = Rotation.from_euler("xyz", pose[3:])
        self.control.initialize_orientation(observed_rotation.as_matrix())
        reference = None
        reference_revision = None
        previous_reference = None
        candidate_rotation = (
            Rotation.from_rotvec(rotation_delta) * observed_rotation
            if any(rotation_delta)
            else Rotation.from_matrix(self.control.orientation_target)
        )
        target_position = [v + d for v, d in zip(position, xyz_delta, strict=True)]
        if self.streaming_reference is not None:
            from .continuous_reference import ReferenceSettings, reference_candidate

            previous_reference, reference_revision = self.control.reference_snapshot(epoch)
            reference = reference_candidate(
                previous_reference,
                position,
                observed_rotation.as_matrix(),
                self.control.orientation_target,
                xyz_delta,
                rotation_delta,
                self.max_xyz_step_m,
                self.max_rotation_step_rad,
                ReferenceSettings(**self.streaming_reference),
                time.monotonic(),
                lock_orientation=not wrist_priority,
            )
            target_position = reference["position"]
            candidate_rotation = Rotation.from_matrix(reference["rotation"])
        # Euler angles are only the fixed SDK IK boundary representation.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Gimbal lock detected.*", category=UserWarning
            )
            rpy = candidate_rotation.as_euler("xyz")
        target_pose = [
            *(
                np.asarray(target_position)
                - candidate_rotation.apply(grasp_offset(self.safety.workspace_geometry))
            ),
            *rpy,
        ]
        target_gripper, gripper_plan = self._plan_gripper(current_gripper, gripper_delta)
        if not wrist_priority and not workspace_pose_allowed(
            pose,
            target_pose,
            current_gripper,
            target_gripper,
            self.safety,
            allow_reentry=True,
        ):
            raise OutcomePiperIntentRejected(
                "Xbox target is outside the workspace and does not move inward"
            )
        seed = current if previous_reference is None else previous_reference["joint_target"]
        if wrist_priority:
            if previous_reference is None and self.control.joint_target is not None:
                seed = self.control.joint_target
            goal_joints, ik_detail = solve_position(
                seed,
                current,
                target_position,
                self.safety,
                current_gripper,
                target_gripper,
                timeout=self.ik_timeout_s,
                max_nfev=self.ik_max_nfev,
            )
        else:
            goal_joints = self._solve(seed, target_pose)
            ik_detail = {"level": "full_pose"}
        # Plan one waypoint of a joint-space segment, rather than rejecting a
        # valid pose correction as if it were a single-cycle command. Recompute
        # from feedback next tick; never queue targets across pause/resume.
        segments = max(
            1,
            math.ceil(
                max(
                    abs(goal - q) / step
                    for goal, q, step in zip(
                        goal_joints, current, self.safety.max_joint_step, strict=True
                    )
                )
            ),
        )
        target_joints = [
            q + (goal - q) / segments for q, goal in zip(current, goal_joints, strict=True)
        ]
        waypoint_pose = fk_from_mdh(list(get_mdh("piper")), target_joints)
        if not workspace_pose_allowed(
            pose,
            waypoint_pose,
            current_gripper,
            target_gripper,
            self.safety,
            allow_reentry=True,
        ):
            raise OutcomePiperIntentRejected(
                "Xbox joint waypoint is outside the workspace and does not move inward"
            )
        if not self.safety.gripper_lower <= target_gripper <= self.safety.gripper_upper:
            raise OutcomePiperIntentRejected("Xbox gripper target is outside frozen limits")
        result = OutcomePiperAction(
            {
                **{key: value for key, value in zip(JOINT_KEYS, target_joints, strict=True)},
                "gripper.pos": target_gripper,
            },
            intent="run",
            epoch=epoch,
        )
        result.gripper_plan = gripper_plan
        if reference is not None:
            # Commit the pose of the actually dispatched joint waypoint, not a
            # farther IK goal when the ordinary joint budget subdivides it.
            reference.update(
                position=grasp_position(waypoint_pose, self.safety.workspace_geometry).tolist(),
                rotation=Rotation.from_euler("xyz", waypoint_pose[3:]).as_matrix().tolist(),
                joint_target=list(target_joints),
            )
            reference["velocity"] = [v / segments for v in reference["velocity"]]
            result.reference_plan = {"base_revision": reference_revision, "reference": reference}
            candidate_rotation = Rotation.from_matrix(reference["rotation"])
        result.orientation_target = (
            Rotation.from_euler("xyz", waypoint_pose[3:]).as_matrix().tolist()
        )
        result.joint_plan = {
            "segments": segments,
            "goal_joint_rad": goal_joints,
            "translation_strategy": self.control.translation_strategy.value,
            "control_point": "grasp_center",
            "ik": ik_detail,
        }
        return result

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
        if self.control.pending_mode is not None or self.control.state in (
            TeleopState.POSE_READY,
            TeleopState.POSE_MOVING,
        ):
            self.control.request_hold("processor_reset")
        self.control.reset_reference()
        self.control.orientation_target = None


def make_xbox_processor(
    safety: MotionSafety,
    *,
    max_xyz_step_m: float,
    max_rotation_step_rad: float,
    max_gripper_step_m: float,
    ik_max_nfev: int,
    ik_timeout_s: float,
    ik_residual_tolerance: float,
    ik_min_singular_value: float,
    work_joint_rad: tuple[float, ...] | None = None,
    work_gripper_m: float | None = None,
    pose_timing: dict | None = None,
    gripper_reference: dict | None = None,
    streaming_reference: dict | None = None,
) -> RobotProcessorPipeline[tuple[RobotAction, dict[str, Any]], RobotAction]:
    return RobotProcessorPipeline(
        steps=[
            OutcomePiperXboxProcessor(
                safety=safety,
                work_joint_rad=work_joint_rad,
                work_gripper_m=work_gripper_m,
                pose_timing=pose_timing,
                streaming_reference=streaming_reference,
                gripper_reference=gripper_reference,
                max_xyz_step_m=max_xyz_step_m,
                max_rotation_step_rad=max_rotation_step_rad,
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
