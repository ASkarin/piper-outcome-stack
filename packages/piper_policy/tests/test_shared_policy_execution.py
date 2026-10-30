from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from piper_policy.policy_execution import ACTChunkPredictor


def predictor():
    cached = {}

    def pre(batch):
        cached["anchor"] = batch["observation.state"].clone()
        return batch

    def post(chunk):
        return chunk + cached["anchor"]

    model = NS(
        config=NS(
            chunk_size=3,
            input_features={
                "observation.images.d435": NS(shape=(3, 2, 2)),
            },
        ),
        predict_action_chunk=lambda batch: torch.full((1, 3, 7), 0.001),
    )
    return ACTChunkPredictor(model, pre, post)


def test_entire_chunk_keeps_original_anchor_after_next_observation():
    p = predictor()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    first = p.predict(image, np.zeros(7))
    second = p.predict(image, np.ones(7))
    np.testing.assert_allclose(first, 0.001)
    np.testing.assert_allclose(second, 1.001)
    second[0] = 10
    np.testing.assert_allclose(first, 0.001)


@pytest.mark.parametrize(
    "image,state",
    [
        (np.zeros((2, 2, 3), dtype=float), np.zeros(7)),
        (np.zeros((3, 2, 2), dtype=np.uint8), np.zeros(7)),
        (np.zeros((2, 2, 3), dtype=np.uint8), [float("nan")] * 7),
    ],
)
def test_invalid_inputs_fail(image, state):
    with pytest.raises(ValueError):
        predictor().predict(image, state)


def _reference(tmp_path, n=2, offset=0.0):
    p = predictor()
    images = np.zeros((n, 2, 2, 3), np.uint8)
    states = np.zeros((n, 7), np.float32)
    expected = np.stack([p.predict(images[i], states[i]) for i in range(n)]) + offset
    path = tmp_path / "reference.npz"
    np.savez(path, images=images, states=states, expected_absolute_chunks=expected)
    return p, path
