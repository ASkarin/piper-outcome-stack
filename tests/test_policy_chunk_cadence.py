from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_outcome_stack.policy_execution import ACTChunkPredictor, predict_candidate
from lerobot_robot_outcome_piper.safety import ACTION_KEYS


def make(steps=5, te=None):
    p = ACTChunkPredictor(NS(config=NS(chunk_size=50)), None, None)
    p.configure_temporal_ensemble(te)
    p.configure_action_steps(steps)
    return p


@pytest.mark.parametrize("steps", [3, 5, 10])
def test_steps_keep_fixed_anchor_and_refresh_at_boundary(steps):
    p = make(steps)
    calls = []

    def predict(image, state):
        calls.append(list(state))
        return np.repeat(np.arange(50, dtype=np.float32)[:, None], 7, axis=1) + state[0]

    p.predict = predict
    for tick in range(12):
        chunk, index, new = p.execution_chunk(None, [tick * 100.0] * 7, tick)
        target = p.select_target(chunk, chunk_index=index, new_chunk=new)
        assert index == tick % steps and new == (tick % steps == 0)
        np.testing.assert_array_equal(target, (tick // steps) * steps * 100 + tick % steps)
        assert p.generation_sequence == tick - tick % steps
    assert len(calls) == (12 + steps - 1) // steps
    p.reset_execution()
    assert p.execution_chunk(None, [999.0] * 7, 99)[2]


@pytest.mark.parametrize("steps", [3, 5, 10])
def test_sparse_te_matches_control_time_alignment_and_weight_units(steps):
    p = make(steps, te=0.01)
    rng = np.random.default_rng(5)
    chunks = rng.normal(size=((85 + steps - 1) // steps, 50, 7)).astype(np.float32)
    calls = []

    def predict(*args):
        chunk = chunks[len(calls)]
        calls.append(True)
        return chunk.copy()

    p.predict = predict
    for tick in range(85):
        c, index, new = p.execution_chunk(None, [tick] * 7, tick)
        actual = p.select_target(c, chunk_index=index, new_chunk=new)
        origins = [o for o in range(0, tick + 1, steps) if tick - o < 50]
        values = np.stack([chunks[o // steps, tick - o] for o in origins])
        weights = np.exp(-0.01 * (np.array(origins) - origins[0]))
        np.testing.assert_allclose(actual, np.average(values, axis=0, weights=weights), atol=2e-7)
        assert p.last_contributors == len(origins)
    assert len(calls) == (85 + steps - 1) // steps


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
            dict(quality="checked", oldest_received_monotonic_s=1.0, sequence=i),
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


@pytest.mark.parametrize("steps", [0, -1, 51, 1.5, True])
def test_invalid_cadence_rejected(steps):
    with pytest.raises(ValueError, match="action steps"):
        make(steps)
