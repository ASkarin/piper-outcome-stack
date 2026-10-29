"""Check source-demo balancing and continuous-window boundaries."""

from types import SimpleNamespace as NS
import torch
from train import CompleteWindows, sampler_class


class ExampleData:
    def __init__(self):
        self.starts, self.ends, self.lineage = [], [], []
        # Eight task/layout cells, one extra demo in the first cell, a second
        # segment of demo0, and a too-short segment from that same demo.
        for demo, length in [(i, 60) for i in range(9)] + [(0, 70), (0, 49)]:
            start = self.ends[-1] if self.ends else 0
            self.starts.append(start)
            self.ends.append(start + length)
            cell = demo if demo < 8 else 0
            self.lineage.append(
                dict(
                    source_session=f"session{demo}",
                    source_episode=0,
                    task=f"task{cell // 2}",
                    layout=f"layout{cell % 2}",
                )
            )
        self.num_episodes = len(self.starts)
        self.repo_id, self.fps = "local/example", 50
        self.meta = NS(info=NS(total_frames=self.ends[-1], total_episodes=self.num_episodes))

    def __getitem__(self, index):
        end = next(end for start, end in zip(self.starts, self.ends) if start <= index < end)
        return {"action_is_pad": torch.arange(index, index + 50) >= end}


def test_sampling_mass_and_no_cross_gap_windows():
    base = ExampleData()
    view = CompleteWindows(base, 9)
    for index in range(len(view)):
        assert not view[index]["action_is_pad"].any()
    masses = {}
    for index, source_index in enumerate(view.indices):
        episode = next(
            i for i, (a, b) in enumerate(zip(base.starts, base.ends)) if a <= source_index < b
        )
        demo = base.lineage[episode]["source_session"]
        masses[demo] = masses.get(demo, 0) + float(view.weights[index])
    assert abs(masses["session0"] - 1 / 16) < 1e-12
    assert abs(masses["session8"] - 1 / 16) < 1e-12
    assert all(abs(masses[f"session{i}"] - 1 / 8) < 1e-12 for i in range(1, 8))
    assert view.summary["eligible_segments"] == 10


def test_paired_sampler_order_and_compressed_index_range():
    view = CompleteWindows(ExampleData(), 9)
    cls = sampler_class(view)
    args = (view.meta.episodes["dataset_from_index"], view.meta.episodes["dataset_to_index"])
    kwargs = dict(
        episode_indices_to_use=view.episodes,
        drop_n_last_frames=0,
        shuffle=True,
        seed=42,
        absolute_to_relative_idx=None,
    )
    first, second = cls(*args, **kwargs), cls(*args, **kwargs)
    for epoch in range(2):
        a, b = list(first), list(second)
        assert a == b and len(a) == len(view) and min(a) >= 0 and max(a) < len(view)
