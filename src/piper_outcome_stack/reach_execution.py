"""State-only S2 candidates; reuse the existing freshness and dispatch boundaries."""

import json
from pathlib import Path
import time

import numpy as np

from piper_policy.reach_bc import FEEDBACK_GAIN
from piper_policy.reach_grid import ReachEndpointGrid
from lerobot_robot_outcome_piper.safety import ACTION_KEYS

from .policy_execution import check_policy_observation


class ReachPredictor:
    def __init__(self, bundle, case_id, direction, method):
        bundle = Path(bundle)
        self.manifest = json.loads((bundle / "manifest.json").read_text())
        record = json.loads((bundle / "grid.json").read_text())
        self.targets = json.loads((bundle / "targets.json").read_text())
        if record["source_checkpoint"] != self.manifest["source_checkpoint"]:
            raise ValueError("grid and manifest source checkpoints differ")
        if self.targets["tip_offset_m"] != [0.0, 0.0, 0.135]:
            raise ValueError("S2 bundle requires the base-frame 135mm fingertip midpoint")
        self.grid = ReachEndpointGrid(
            record["axes_m"], record["endpoints_rad"], record["fixed_gripper_m"]
        )
        if self.grid.gripper != 0.001:
            raise ValueError("S2 fixed gripper must be 1mm")
        matches = [row for row in self.targets["cases"] if row["id"] == case_id]
        if (
            len(matches) != 1
            or direction not in ("out", "back")
            or method not in ("grid", "ik_reference")
        ):
            raise ValueError(
                "explicit frozen case, direction and grid/ik_reference method required"
            )
        self.case, self.direction, self.method = matches[0], direction, method
        if direction == "out":
            self.goal = np.asarray(self.case["tip_goal_base_m"], dtype=float)
            self.ik_endpoint = np.asarray(self.case["endpoint_rad_m"], dtype=float)[:6]
        else:
            self.goal = np.asarray(self.targets["anchor_tip_base_m"], dtype=float)
            self.ik_endpoint = np.asarray(self.targets["anchor_state_rad_m"], dtype=float)[:6]
        if (
            self.goal.shape != (3,)
            or self.ik_endpoint.shape != (6,)
            or not np.isfinite(self.ik_endpoint).all()
        ):
            raise ValueError("invalid frozen reach target")
        self.grid.endpoint(self.goal)  # Both methods are confined to the same frozen goal domain.

    def reset_execution(self):
        pass  # No chunk, temporal filter or changing reference exists in this fixed-goal policy.

    def predict(self, state):
        state = np.asarray(state, dtype=float)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("reach requires finite physical state7")
        if self.method == "grid":
            return self.grid.action(state, self.goal)
        return np.r_[state[:6] + FEEDBACK_GAIN * (self.ik_endpoint - state[:6]), self.grid.gripper]


def tip_fk_base_m(state):
    """SDK MDH + local-Z135mm, not signed table height or independent TCP measurement."""
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    from scipy.spatial.transform import Rotation

    pose = fk_from_mdh(list(get_mdh("piper")), list(state)[:6])
    return np.asarray(pose[:3]) + Rotation.from_euler("xyz", pose[3:]).apply([0, 0, 0.135])


def check_reach_geometry(safety):
    geometry = safety.workspace_geometry
    if geometry is None or tuple(geometry.tip_offset_m) != (0.0, 0.0, 0.135):
        raise ValueError("current workspace must explicitly use the same 135mm fingertip point")


def reach_candidate(
    predictor,
    observation,
    telemetry,
    timing,
    safety,
    clock=time.monotonic,
    record_action_chunk=False,
    *,
    control=None,
    epoch=None,
    gripper_reference_settings=None,
):
    """Predict/check one full action; never connects, enables or sends to the SDK."""
    from lerobot_robot_outcome_piper.execution_constraints import check_execution_target

    if gripper_reference_settings is not None:
        raise ValueError("fixed-gripper S2 does not use the visual-policy gripper reference")
    started = clock()
    check_policy_observation(telemetry, timing, started)
    check_reach_geometry(safety)
    state = [float(observation[key]) for key in ACTION_KEYS]
    target = predictor.predict(state)
    predicted = clock()
    details = dict(
        method=predictor.method,
        case=predictor.case["id"],
        direction=predictor.direction,
        goal_base_m=predictor.goal.tolist(),
        candidate_rad_m=target.tolist(),
        source_checkpoint=predictor.manifest["source_checkpoint"],
        observation_sequence=telemetry["sequence"],
        inference_s=predicted - started,
        tcp_metric="FK tracking only; no independent measurement",
    )
    try:
        check_execution_target(state, target, safety)
        details["observation_age_at_candidate_s"] = check_policy_observation(
            telemetry, timing, clock()
        )
    except ValueError as exc:
        exc.policy_prediction = details
        raise
    return dict(zip(ACTION_KEYS, map(float, target), strict=True)), details
