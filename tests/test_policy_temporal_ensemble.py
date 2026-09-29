from types import SimpleNamespace

import numpy as np
import pytest

from piper_outcome_stack.policy_execution import ACTChunkPredictor


def predictor(horizon=50, coefficient=0.01):
    p = ACTChunkPredictor(SimpleNamespace(config=SimpleNamespace(chunk_size=horizon)), None, None)
    p.configure_temporal_ensemble(coefficient)
    return p


def test_matches_explicit_time_aligned_absolute_history_after_full_window():
    p = predictor()
    rng = np.random.default_rng(42)
    chunks = rng.normal(size=(80, 50, 7)).astype(np.float32)
    before = chunks.copy()
    for tick, chunk in enumerate(chunks):
        first = max(0, tick - 49)
        candidates = np.stack([chunks[origin, tick - origin] for origin in range(first, tick + 1)])
        weights = np.exp(-0.01 * np.arange(len(candidates)))
        expected = (candidates * weights[:, None]).sum(0) / weights.sum()
        np.testing.assert_allclose(p.select_target(chunk), expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(chunks, before)


def test_different_anchors_are_not_rebased_to_current_observation():
    p = predictor(3)
    old_absolute = np.repeat(np.array([[0.0], [1.0], [2.0]], np.float32), 7, axis=1)
    new_absolute = old_absolute + 10
    np.testing.assert_allclose(p.select_target(old_absolute), 0)
    expected = (1 + 10 * np.exp(-0.01)) / (1 + np.exp(-0.01))
    np.testing.assert_allclose(p.select_target(new_absolute), expected, rtol=1e-6)


def test_reset_clears_prior_episode_and_no_te_preserves_first_step():
    p = predictor(3)
    p.select_target(np.zeros((3, 7), np.float32))
    p.reset_execution()
    np.testing.assert_array_equal(p.select_target(np.ones((3, 7), np.float32)), np.ones(7))
    assert p.ensemble_updates == 1
    p.configure_temporal_ensemble(None)
    np.testing.assert_array_equal(p.select_target(np.full((3, 7), 2, np.float32)), np.full(7, 2))
    assert p.ensemble_updates == 0


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf")])
def test_invalid_coefficient_fails(coefficient):
    with pytest.raises(ValueError, match="finite"):
        predictor(coefficient=coefficient)
