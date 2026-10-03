from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from piper_policy.policy_execution import ACTChunkPredictor


def test_batch_decoding_preserves_per_row_anchor_and_excludes_truth():
    captured = {}

    def pre(batch):
        assert set(batch) == {"observation.state", "observation.images.d435"}
        captured["state"] = batch["observation.state"].clone()
        return batch

    def post(chunk):
        return chunk + captured["state"][:, None, :]

    model = NS(
        config=NS(chunk_size=3, input_features={"observation.images.d435": NS(shape=(3, 2, 2))}),
        predict_action_chunk=lambda batch: torch.ones(len(batch["observation.state"]), 3, 7) * 0.01,
    )
    p = ACTChunkPredictor(model, pre, post)
    actual = p.predict_batch(torch.zeros(2, 3, 2, 2), torch.stack([torch.zeros(7), torch.ones(7)]))
    np.testing.assert_allclose(actual[0], 0.01)
    np.testing.assert_allclose(actual[1], 1.01)
    with pytest.raises(ValueError, match="RGB"):
        p.predict_batch(torch.full((2, 3, 2, 2), 2.0), torch.zeros(2, 7))
