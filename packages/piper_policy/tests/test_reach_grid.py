import itertools

import numpy as np
import pytest

from piper_policy.reach_grid import ReachEndpointGrid


def polynomial(point):
    x, y, z = point
    return np.array(
        [1 + x + y * z, 2 + y + x * z, 3 + z + x * y, x * y * z, x - y + z, 10 + x + y + z]
    )


def grid():
    axes = np.array([[-1.0, 0.0, 2.0], [-2.0, 0.0, 3.0], [-3.0, 0.0, 4.0]])
    values = np.empty((3, 3, 3, 6))
    for index in itertools.product(range(3), repeat=3):
        values[index] = polynomial([axes[d, index[d]] for d in range(3)])
    return ReachEndpointGrid(axes, values, 0.001)


def test_multiaffine_exact_across_cells_nodes_faces_and_upper_boundary():
    policy = grid()
    for point in itertools.product(
        *[[-1.0, -0.7, 0.0, 0.6, 2.0], [-2.0, -0.2, 0.0, 1.2, 3.0], [-3.0, -1.1, 0.0, 0.8, 4.0]]
    ):
        np.testing.assert_allclose(policy.endpoint(point), polynomial(point), atol=1e-13, rtol=0)


def test_fixed_feedback_uses_current_state_and_does_not_clip():
    policy = grid()
    state = np.arange(7, dtype=float) * 0.1
    goal = np.array([0.6, 1.2, 0.8])
    first = policy.action(state, goal)
    np.testing.assert_allclose(first[:6], 0.8 * state[:6] + 0.2 * polynomial(goal), atol=1e-14)
    np.testing.assert_allclose(policy.action(state + 0.3, goal)[:6] - first[:6], 0.24, atol=1e-14)
    assert first[6] == 0.001 and first[5] > 2


def test_grid_rejects_extrapolation_nonfinite_and_missing_nodes():
    policy = grid()
    with pytest.raises(ValueError, match="outside training grid"):
        policy.endpoint([2.000001, 0, 0])
    with pytest.raises(ValueError, match="finite goal"):
        policy.endpoint([np.nan, 0, 0])
    with pytest.raises(ValueError, match="state7"):
        policy.action([0] * 6, [0] * 3)
    with pytest.raises(ValueError, match="27 joint6"):
        ReachEndpointGrid(policy.axes, policy.values[:, :2], 0.001)
