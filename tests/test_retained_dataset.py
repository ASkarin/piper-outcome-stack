from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from piper_outcome_stack.retained_dataset import (
    RetainedDataset,
    retained_indices,
    require_disjoint_sessions,
)


def test_retained_index_uses_source_offsets_and_rejects_reordering():
    rows = [
        {"provenance": {"merged_episode": 0}, "source_frames": 5, "original_frame_indices": [1, 4]},
        {"provenance": {"merged_episode": 1}, "source_frames": 3, "original_frame_indices": [0, 2]},
    ]
    ids, starts, ends, _ = retained_indices(rows, [0, 5], [5, 8])
    assert ids.tolist() == [1, 4, 5, 7] and starts == [0, 2] and ends == [2, 4]
    rows[0]["original_frame_indices"] = [4, 1]
    with pytest.raises(ValueError, match="increasing"):
        retained_indices(rows, [0, 5], [5, 8])


def test_future_chunk_never_crosses_episode_or_source_boundary():
    class Source:
        def __init__(self, offset):
            self.offset = offset

        def __getitem__(self, index):
            return {
                "source_frame": index + self.offset,
                **{k: torch.tensor(0) for k in ["episode_index", "frame_index", "index"]},
                "timestamp": torch.tensor(0.0),
            }

    view = RetainedDataset.__new__(RetainedDataset)
    view.num_frames, view.chunk_size = 6, 4
    view.parts = [(Source(0), np.array([1, 4, 5, 7])), (Source(100), np.array([2, 3]))]
    view.part_ends, view.starts, view.ends = [4, 6], [0, 2, 4], [2, 4, 6]
    view.actions = np.repeat(np.arange(6, dtype=np.float32)[:, None], 7, axis=1)
    end = view[3]
    assert end["source_frame"] == 7 and end["episode_index"] == 1
    assert end["action"][:, 0].tolist() == [3, 3, 3, 3]
    assert end["action_is_pad"].tolist() == [False, True, True, True]
    next_source = view[4]
    assert next_source["source_frame"] == 102 and next_source["episode_index"] == 2
    assert next_source["frame_index"] == 0
    assert next_source["action"][:, 0].tolist() == [4, 5, 5, 5]


def test_source_session_overlap_is_rejected():
    a = NS(lineage=[{"source_session": "a"}])
    b = NS(lineage=[{"source_session": "b"}])
    require_disjoint_sessions(a, b)
    with pytest.raises(ValueError, match="overlap"):
        require_disjoint_sessions(a, a)
