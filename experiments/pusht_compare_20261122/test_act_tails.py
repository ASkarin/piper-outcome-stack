"""A short segment gets real training starts without changing the normalizer population."""

from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

from piper_outcome_stack.retained_dataset import RetainedDataset
from train_act import training_population


def test_short_segments_and_tails_use_only_real_labels_and_balance_by_original_demo():
    class Source:
        def __getitem__(self, index):
            return dict(
                index=torch.tensor(index),
                frame_index=torch.tensor(index),
                episode_index=torch.tensor(0),
                timestamp=torch.tensor(0.0),
            )

    raw = RetainedDataset.__new__(RetainedDataset)
    raw.chunk_size = 50
    raw.starts, raw.ends = [0, 52, 55], [52, 55, 105]
    raw.num_frames, raw.num_episodes = 105, 3
    raw.repo_id, raw.fps = "local/test", 50
    raw.episodes = [0, 1, 2]
    raw.absolute_to_relative_idx = None
    raw.parts, raw.part_ends = [(Source(), np.arange(len(raw)))], [len(raw)]
    raw.actions = np.repeat(np.arange(len(raw), dtype=np.float32)[:, None], 7, axis=1)
    raw.lineage = [
        dict(source_session="c", source_episode=ep, task="Push-T", layout="start-03")
        for ep in (0, 0, 1)
    ]
    raw.meta = NS(info=NS(total_frames=len(raw), total_episodes=3))
    counts = dict(train=2, task_layout_cells=1)
    full, _, reference, _ = training_population(raw, counts, "complete")
    tail, weights, normalization, summary = training_population(raw, counts, "masked_segment_tails")
    np.testing.assert_array_equal(normalization, reference)
    assert reference.tolist() == [0, 1, 2, 55] and len(full) == 4 and len(tail) == 105
    assert summary["masked_tail_starts"] == 101 and summary["eligible_segments"] == 3
    assert weights[:55].sum().item() == pytest.approx(0.5)
    assert weights[55:].sum().item() == pytest.approx(0.5)
    for index, valid in [(2, 50), (50, 2), (52, 3), (54, 1), (55, 50), (104, 1)]:
        item = tail[index]
        assert item["action_is_pad"].tolist() == [False] * valid + [True] * (50 - valid)
        assert item["action"][:valid, 0].tolist() == list(range(index, index + valid))
        assert item["action"][valid:, 0].tolist() == [index + valid - 1] * (50 - valid)
