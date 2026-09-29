"""Position-only grasp-center IK. No SDK connection or command dispatch."""

import time
import numpy as np
from scipy.optimize import least_squares, minimize
from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
from .errors import OutcomePiperIntentRejected, OutcomePiperControlTimeout
from .workspace import workspace_pose_allowed, grasp_position


def solve_position(seed, feedback, target, safety, gripper, target_gripper, *, timeout, max_nfev):
    started = time.monotonic()
    mdh = list(get_mdh("piper"))
    seed, feedback, target = map(lambda v: np.asarray(v, float), (seed, feedback, target))
    lower = np.maximum(safety.joint_lower, feedback - np.asarray(safety.max_joint_step))
    upper = np.minimum(safety.joint_upper, feedback + np.asarray(safety.max_joint_step))
    current_pose = fk_from_mdh(mdh, feedback.tolist())

    def pose(q):
        if time.monotonic() - started >= timeout:
            raise OutcomePiperControlTimeout("position IK exceeded shared time budget")
        return fk_from_mdh(mdh, q.tolist())

    def residual(q):
        return grasp_position(pose(q), safety.workspace_geometry) - target

    def valid(q):
        return (
            np.isfinite(q).all()
            and np.all(q >= lower)
            and np.all(q <= upper)
            and np.linalg.norm(residual(q)) <= 1e-5
            and workspace_pose_allowed(
                current_pose, pose(q), gripper, target_gripper, safety, allow_reentry=True
            )
        )

    # Only the seed may be projected into the search bounds; no output is clipped.
    initial = np.clip(seed, lower, upper)
    fixed = None
    if np.all(seed[3:] >= lower[3:]) and np.all(seed[3:] <= upper[3:]):

        def expand(arm):
            return np.r_[arm, seed[3:]]

        fixed = least_squares(
            lambda arm: residual(expand(arm)),
            initial[:3],
            bounds=(lower[:3], upper[:3]),
            gtol=None,
            max_nfev=max(2, max_nfev // 3),
        )
        q = expand(fixed.x)
        if fixed.success and valid(q):
            level = "fixed_wrist"
        else:
            fixed = None
    if fixed is None:
        scale = np.asarray(safety.max_joint_step)
        weights = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])

        def objective(q):
            residual(q)  # Shared deadline also applies to optimizer callbacks.
            return float(np.sum(weights * ((q - seed) / scale) ** 2))

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints={"type": "eq", "fun": residual},
            options={"maxiter": max_nfev, "ftol": 1e-10},
        )
        q = result.x
        if not result.success or not valid(q):
            raise OutcomePiperIntentRejected(
                "grasp-center position unreachable within joint/step/workspace constraints"
            )
        level = "weighted_wrist"
    # Position Jacobian diagnostics are not the six-dimensional orientation gate.
    h = 1e-6
    jac = np.column_stack([(residual(q + np.eye(6)[i] * h) - residual(q)) / h for i in range(6)])
    detail = dict(
        level=level,
        position_error_m=float(np.linalg.norm(residual(q))),
        position_sigma_min=float(np.linalg.svd(jac, compute_uv=False)[-1]),
        wrist_delta_rad=(q[3:] - seed[3:]).tolist(),
        solve_duration_s=time.monotonic() - started,
    )
    return q.tolist(), detail
