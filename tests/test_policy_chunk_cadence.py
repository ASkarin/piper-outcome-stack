from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_policy.policy_execution import ACTChunkPredictor
from piper_outcome_stack.policy_execution import predict_candidate
from lerobot_robot_outcome_piper.safety import ACTION_KEYS


def make(steps=5, te=None):
    p = ACTChunkPredictor(NS(config=NS(chunk_size=50)), None, None)
    p.configure_temporal_ensemble(te)
    p.configure_action_steps(steps)
    return p


def test_cached_targets_are_checked_against_current_feedback(monkeypatch):
    p = make()
    calls = []
    checked = []

    def predict(*args):
        calls.append(True)
        return np.full((50, 7), 0.001, dtype=np.float32)

    p.predict = predict

    def bounds(state, target, safety):
        checked.append(list(state))
        if state[0] == 2:
            raise ValueError("cached target outside current bounds")

    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.execution_constraints.check_execution_target", bounds
    )
    for i in range(3):
        obs = {**dict(zip(ACTION_KEYS, [float(i)] + [0.0] * 6)), "d435": None}
        args = (
            p,
            obs,
            dict(
                quality="checked",
                oldest_received_monotonic_s=1.0,
                observed_monotonic_s=1.0,
                sequence=i,
            ),
            NS(observation_max_age_s=0.1),
            None,
            lambda: 1.0,
        )
        if i == 2:
            with pytest.raises(ValueError, match="cached target"):
                predict_candidate(*args)
        else:
            _, details = predict_candidate(*args)
            assert details["inference_performed"] == (i == 0)
            assert details["anchor"] == [0.0] * 7
            assert details["generation_observation_sequence"] == 0
    assert len(calls) == 1 and len(checked) == 3
    with pytest.raises(ValueError, match="expired"):
        predict_candidate(
            p,
            obs,
            dict(quality="checked", oldest_received_monotonic_s=0.0, sequence=4),
            NS(observation_max_age_s=0.1),
            None,
            lambda: 1.0,
        )
