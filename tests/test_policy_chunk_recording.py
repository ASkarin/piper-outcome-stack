from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_outcome_stack.policy_execution import predict_candidate
from lerobot_robot_outcome_piper.safety import ACTION_KEYS


def fixture(index=0, inferred=True):
    chunk = np.arange(350, dtype=np.float32).reshape(50, 7) / 100000
    p = NS(
        execution_chunk=lambda *args, **kwargs: (chunk, index, inferred),
        select_target=lambda c, **kwargs: c[kwargs["chunk_index"]].copy(),
        generation_state=[0.0] * 7,
        generation_sequence=12,
        n_action_steps=50,
        temporal_ensemble_coeff=None,
        temporal_ensembler=None,
        last_contributors=1,
    )
    obs = {**dict(zip(ACTION_KEYS, [0.001] * 7)), "d435": None}
    args = (
        p,
        obs,
        dict(
            quality="checked",
            oldest_received_monotonic_s=1.0,
            observed_monotonic_s=1.0,
            sequence=13,
        ),
        NS(observation_max_age_s=0.1),
        None,
        lambda: 1.0,
    )
    return chunk, args


def test_recording_copies_new_chunk_and_links_cached_targets(monkeypatch):
    checked = []
    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.execution_constraints.check_execution_target",
        lambda state, target, safety: checked.append((list(state), target.copy())),
    )
    chunk, args = fixture()
    target, details = predict_candidate(*args, record_action_chunk=True)
    saved = details["absolute_action_chunk_rad_m"]
    chunk[:] = 99
    assert saved[0][0] == 0 and details["generation_observation_sequence"] == 12
    assert details["anchor"] == [0.0] * 7 and details["candidate_chunk_index"] == 0
    chunk, args = fixture(index=49, inferred=False)
    target, details = predict_candidate(*args, record_action_chunk=True)
    assert "absolute_action_chunk_rad_m" not in details
    assert details["candidate_chunk_index"] == 49
    np.testing.assert_array_equal(list(target.values()), chunk[49])
    assert len(checked) == 2 and checked[-1][0] == [0.001] * 7
    _, details = predict_candidate(*args)
    assert "absolute_action_chunk_rad_m" not in details and "selected_target_rad_m" not in details


def test_rejected_chunk_is_preserved_without_changing_rejection(monkeypatch):
    def reject(*args):
        raise ValueError("outside workspace")

    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.execution_constraints.check_execution_target", reject
    )
    chunk, args = fixture()
    with pytest.raises(ValueError, match="outside workspace") as caught:
        predict_candidate(*args, record_action_chunk=True)
    np.testing.assert_array_equal(
        caught.value.policy_prediction["absolute_action_chunk_rad_m"], chunk
    )


def test_expired_new_chunk_is_preserved_but_not_returned(monkeypatch):
    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.execution_constraints.check_execution_target",
        lambda *args: None,
    )
    chunk, args = fixture()
    times = iter([1.0, 1.11, 1.12])
    with pytest.raises(ValueError, match="expired") as caught:
        predict_candidate(*args[:-1], lambda: next(times), record_action_chunk=True)
    assert caught.value.policy_timing["inference_performed"]
    np.testing.assert_array_equal(
        caught.value.policy_prediction["absolute_action_chunk_rad_m"], chunk
    )
