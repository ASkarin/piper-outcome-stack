"""Recorded low-pose geometry; no hardware I/O or inferred failed command."""

from dataclasses import replace
import numpy as np
import pytest
from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
from lerobot.types import TransitionKey
from lerobot_robot_outcome_piper import position_ik
from lerobot_robot_outcome_piper.errors import (
    OutcomePiperIntentRejected,
    OutcomePiperControlTimeout,
)
from lerobot_robot_outcome_piper.workspace import WorkspaceGeometry, grasp_position
from lerobot_robot_outcome_piper.teleop_control import TranslationStrategy, TeleopState
from test_processor import safety, processor, observation, raw_action

# Last dispatched target in B trial 20260930T011233Z-u7jr5lqm, episode 2.
Q = np.array(
    [
        0.7498807131193631,
        2.2029472250288986,
        -1.5334366777385546,
        -0.2290395577392159,
        0.6299067803372735,
        0.4122991291986205,
    ]
)
GAP = 0.09192591166309699


def limits(floor):
    return replace(
        safety(),
        joint_lower=tuple(np.deg2rad([-154, 0, -175, -102, -75, -170])),
        joint_upper=tuple(np.deg2rad([154, 195, 0, 102, 75, 170])),
        max_joint_step=(np.deg2rad(10),) * 6,
        gripper_upper=0.12,
        workspace_lower=(0.05, -0.46, floor),
        workspace_upper=(0.49, 0.47, 0.35),
        workspace_geometry=WorkspaceGeometry(
            [0, 0, 0.135],
            [0, 0, 0.105],
            [0, 1, 0],
            [0.0038339069163626075, -0.0009827576495255808, 0.9999921676419066],
            -0.010026744786570889,
        ),
    )


@pytest.mark.parametrize(
    "floor,down,allowed",
    [(0.008, 0.00115, False), (0.005, 0.00115, True), (0.005, 0.005, False), (0.005, -0.002, True)],
)
def test_converged_boundary_does_not_enter_weighted_optimizer(monkeypatch, floor, down, allowed):
    monkeypatch.setattr(
        position_ik,
        "minimize",
        lambda *a, **k: pytest.fail("boundary must hold, not search another wrist pose"),
    )
    s = limits(floor)
    target = grasp_position(
        fk_from_mdh(list(get_mdh("piper")), Q.tolist()), s.workspace_geometry
    ) + [0, 0, -down]

    def solve():
        return position_ik.solve_position(Q, Q, target, s, GAP, GAP, timeout=1, max_nfev=100)

    if allowed:
        goal, detail = solve()
        assert detail["level"] == "fixed_wrist"
        assert goal[3:] == pytest.approx(Q[3:])
    else:
        with pytest.raises(OutcomePiperIntentRejected, match="桌面边界"):
            solve()


def test_boundary_becomes_current_feedback_hold_without_fault(monkeypatch):
    step = processor(safety=limits(0.005))
    step.control.translation_strategy = TranslationStrategy.WRIST_PRIORITY
    obs = observation(Q.tolist(), GAP)
    step._current_transition = {TransitionKey.OBSERVATION: obs}
    for name in ("request_input_fault_hold", "request_input_emergency_stop"):
        monkeypatch.setattr(
            "lerobot_robot_outcome_piper.processor." + name,
            lambda *a: pytest.fail("ordinary boundary must not fault the session"),
        )
    raw = {**raw_action(dx=0.001), "stick_x": 0.0, "stick_z": -0.5}
    result = step.action(raw)
    assert result.intent == "hold"
    assert dict(result) == obs
    assert "桌面边界" in result.rejection_reason
    assert step.control.state is TeleopState.HOLD_REQUESTED


def test_position_diagnostics_cannot_return_after_shared_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(position_ik.time, "monotonic", lambda: now[0])
    original_svd = position_ik.np.linalg.svd

    def delayed_svd(*args, **kwargs):
        result = original_svd(*args, **kwargs)
        now[0] = 1.01
        return result

    monkeypatch.setattr(position_ik.np.linalg, "svd", delayed_svd)
    s = limits(0.005)
    target = grasp_position(fk_from_mdh(list(get_mdh("piper")), Q.tolist()), s.workspace_geometry)
    with pytest.raises(OutcomePiperControlTimeout, match="shared time budget"):
        position_ik.solve_position(Q, Q, target, s, GAP, GAP, timeout=1, max_nfev=100)
