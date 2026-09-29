"""Offline execution bounds for actual seven-dimensional targets, without label mapping."""

import numpy as np
from .safety import step_within_limit
from .workspace import workspace_pose_allowed


def check_execution_target(observation, target, safety):
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

    encoded = np.asarray(target, dtype=np.float32).astype(float).tolist()
    observation = list(observation)
    if (
        len(encoded) != 7
        or len(observation) != 7
        or not np.isfinite([*encoded, *observation]).all()
    ):
        raise ValueError("execution requires finite seven-dimensional state and action")
    q, g = encoded[:6], encoded[6]
    for i, (value, lo, hi, step, old) in enumerate(
        zip(
            q,
            safety.joint_lower,
            safety.joint_upper,
            safety.max_joint_step,
            observation[:6],
            strict=True,
        )
    ):
        if not lo <= value <= hi:
            raise ValueError(f"candidate J{i + 1} outside joint bounds: {value}")
        if not step_within_limit(value, old, step):
            raise ValueError(f"candidate J{i + 1} exceeds execution step")
    if not safety.gripper_lower <= g <= safety.gripper_upper:
        raise ValueError(f"candidate gripper outside bounds: {g}")
    if not step_within_limit(g, observation[6], safety.max_gripper_step):
        raise ValueError("candidate gripper exceeds execution step")
    mdh = list(get_mdh("piper"))
    current_pose = fk_from_mdh(mdh, observation[:6])
    target_pose = fk_from_mdh(mdh, q)
    if not workspace_pose_allowed(current_pose, target_pose, observation[6], g, safety):
        raise ValueError("candidate target outside configured workspace/table clearance")
