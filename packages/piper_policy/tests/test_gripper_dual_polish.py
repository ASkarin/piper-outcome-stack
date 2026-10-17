"""The observed correlated dual must converge without relaxing the width bounds."""

import json
from pathlib import Path

import pytest
import torch

from piper_policy.gripper_refit import _polish_nonnegative_dual


def observed_problem():
    data = json.loads((Path(__file__).parent / "fixtures/close_gripper_dual.json").read_text())
    return tuple(
        torch.tensor(data[key], dtype=torch.float64) for key in ("gram", "rhs", "multipliers")
    )


def test_observed_correlated_face_reaches_original_precision_and_keeps_input():
    gram, rhs, warm = observed_problem()
    before = warm.clone()
    result, steps = _polish_nonnegative_dual(gram, rhs, warm, tolerance=1e-8, max_steps=137)
    assert steps == 2 and (result >= 0).all()
    residual = rhs - gram @ result
    kkt = torch.where(result > 0, residual.abs(), residual.clamp_min(0))
    assert kkt.max() <= 1e-8
    assert result[29] == 0 and warm[29] > 0  # remove the wrongly positive warm face

    def objective(x):
        return 0.5 * x @ gram @ x - rhs @ x

    assert objective(result) <= objective(warm)
    torch.testing.assert_close(warm, before, rtol=0, atol=0)


@pytest.mark.parametrize("rhs, expected", [([1.0, -1.0], [0.5, 0.0]), ([-1.0, -2.0], [0.0, 0.0])])
def test_dual_adds_required_face_and_leaves_inactive_constraints(rhs, expected):
    gram = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    result, _ = _polish_nonnegative_dual(
        gram,
        torch.tensor(rhs, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        tolerance=1e-12,
        max_steps=5,
    )
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))


def test_exhausted_budget_does_not_return_or_mutate_an_unverified_solution():
    gram, rhs, warm = observed_problem()
    before = warm.clone()
    with pytest.raises(RuntimeError, match="budget exhausted"):
        _polish_nonnegative_dual(gram, rhs, warm, tolerance=1e-8, max_steps=1)
    torch.testing.assert_close(warm, before, rtol=0, atol=0)
