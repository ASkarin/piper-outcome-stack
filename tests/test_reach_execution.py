import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from lerobot_robot_outcome_piper.safety import ACTION_KEYS
from piper_outcome_stack.reach_execution import ReachPredictor, reach_candidate
from piper_outcome_stack import reach_trial


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    axes = np.array([[-0.02, 0, 0.02], [-0.02, 0, 0.02], [0.18, 0.2, 0.22]])
    values = np.empty((3, 3, 3, 6))
    for x in range(3):
        for y in range(3):
            for z in range(3):
                values[x, y, z] = (
                    0.01 + 0.2 * axes[0, x] - 0.3 * axes[1, y] + 0.4 * (axes[2, z] - 0.2)
                )
    grid = dict(
        source_checkpoint="frozen/source/010000",
        axes_m=axes.tolist(),
        endpoints_rad=values.tolist(),
        fixed_gripper_m=0.001,
    )
    target = dict(
        tip_offset_m=[0.0, 0.0, 0.135],
        anchor_state_rad_m=[0.0] * 6 + [0.001],
        anchor_tip_base_m=[0.0, 0.0, 0.2],
        cases=[
            dict(
                id="evaluation-00",
                tip_goal_base_m=[0.01, -0.01, 0.21],
                endpoint_rad_m=[0.04] * 6 + [0.001],
            )
        ],
    )
    for name, data in [
        ("grid", grid),
        ("targets", target),
        ("manifest", dict(source_checkpoint=grid["source_checkpoint"])),
    ]:
        (root / (name + ".json")).write_text(json.dumps(data))
    return root


def test_grid_and_frozen_ik_reference_are_explicit_and_grid_does_not_use_ik_labels(bundle):
    state = np.zeros(7)
    grid = ReachPredictor(bundle, "evaluation-00", "out", "grid")
    ik = ReachPredictor(bundle, "evaluation-00", "out", "ik_reference")
    np.testing.assert_allclose(grid.predict(state)[:6], 0.2 * 0.019, atol=1e-16)
    np.testing.assert_allclose(ik.predict(state)[:6], 0.2 * 0.04)
    before = grid.predict(state)
    p = bundle / "targets.json"
    data = json.loads(p.read_text())
    data["cases"][0]["endpoint_rad_m"] = [100.0] * 6 + [0.001]
    p.write_text(json.dumps(data))
    np.testing.assert_array_equal(
        ReachPredictor(bundle, "evaluation-00", "out", "grid").predict(state), before
    )
    np.testing.assert_allclose(
        ReachPredictor(bundle, "evaluation-00", "back", "ik_reference").predict(state)[:6], 0
    )
    p = bundle / "manifest.json"
    p.write_text(json.dumps(dict(source_checkpoint="other")))
    with pytest.raises(ValueError, match="source checkpoints differ"):
        ReachPredictor(bundle, "evaluation-00", "out", "grid")


def telemetry():
    return dict(quality="checked", oldest_received_monotonic_s=99.99, sequence=3)


def test_state_only_candidate_checks_freshness_before_and_after_and_preserves_rejection(
    bundle, monkeypatch
):
    predictor = ReachPredictor(bundle, "evaluation-00", "out", "grid")
    observation = dict(zip(ACTION_KEYS, [0.0] * 6 + [0.001], strict=True))
    safety = NS(workspace_geometry=NS(tip_offset_m=(0.0, 0.0, 0.135)))
    checks = []
    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.execution_constraints.check_execution_target",
        lambda state, target, safety: checks.append((state, target)),
    )
    clock = iter([100.0, 100.01, 100.02]).__next__
    target, details = reach_candidate(
        predictor, observation, telemetry(), NS(observation_max_age_s=0.1), safety, clock
    )
    assert set(target) == set(ACTION_KEYS) and len(checks) == 1
    assert details["observation_sequence"] == 3 and target["gripper.pos"] == 0.001
    clock = iter([100.0, 100.01, 101.0]).__next__
    with pytest.raises(ValueError, match="expired") as exc:
        reach_candidate(
            predictor, observation, telemetry(), NS(observation_max_age_s=0.1), safety, clock
        )
    assert exc.value.policy_prediction["candidate_rad_m"] == list(target.values())
    checks.clear()
    with pytest.raises(ValueError, match="expired"):
        reach_candidate(
            predictor,
            observation,
            {**telemetry(), "oldest_received_monotonic_s": 99.0},
            NS(observation_max_age_s=0.1),
            safety,
            lambda: 100.0,
        )
    assert checks == []


def test_configuration_derivation_is_explicit_and_preserves_original_numbers(tmp_path):
    p = tmp_path / "operator.json"
    before = dict(
        robot=dict(cameras={"d435": {"fps": 60}}, safety_path="/current/safety.json"),
        teleop=dict(control_hz=50),
    )
    p.write_text(json.dumps(before))
    raw = p.read_bytes()
    output = tmp_path / "state.json"
    result = reach_trial.state_config(p, output)
    after = json.loads(output.read_text())
    assert after == {**before, "robot": {**before["robot"], "cameras": {}}}
    assert p.read_bytes() == raw and result["hardware_io"] is False


def test_check_mode_does_not_construct_or_run_a_robot(tmp_path, monkeypatch):
    from lerobot_robot_outcome_piper import OutcomePiper, OutcomePiperXbox

    def forbidden(*args, **kwargs):
        raise AssertionError("device construction/run forbidden in check")

    monkeypatch.setattr(OutcomePiper, "__init__", forbidden)
    monkeypatch.setattr(OutcomePiperXbox, "__init__", forbidden)
    monkeypatch.setattr(reach_trial, "run_leg", forbidden)
    monkeypatch.setattr(reach_trial, "load_configuration", lambda path: ({}, None, None, None))
    monkeypatch.setattr(
        reach_trial,
        "check_recorded_inputs",
        lambda *a: dict(
            status="offline_check_completed",
            recorded_rows=1,
            rejected_candidates_preserved=0,
            hardware_io=False,
        ),
    )
    assert (
        reach_trial.main(
            [
                "check",
                "--bundle",
                str(tmp_path),
                "--config",
                "current.json",
                "--reference",
                "old.json",
                "--output",
                str(tmp_path / "check.json"),
            ]
        )
        == 0
    )
