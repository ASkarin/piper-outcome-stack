"""Source-demonstration-balanced sampling for retained robot action windows."""

from collections import defaultdict
from copy import copy

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def demonstration_weights(starts, ends, lineage, expected_demos, expected_cells):
    """Assign equal probability to each cell, then each original demo, then its starts."""
    groups, cells = defaultdict(list), defaultdict(set)
    for start, end, source in zip(starts, ends, lineage, strict=True):
        demo = (source["source_session"], source["source_episode"])
        cell = (source["task"], source["layout"])
        groups[demo].extend(range(start, end))
        cells[cell].add(demo)
    require(
        len(groups) == expected_demos and len(cells) == expected_cells,
        f"window demo/cell coverage differs: {len(groups)}/{len(cells)}",
    )
    weights = torch.zeros(ends[-1], dtype=torch.float64)
    for demos in cells.values():
        for demo in demos:
            weights[groups[demo]] = 1.0 / (len(cells) * len(demos) * len(groups[demo]))
    require(
        bool((weights > 0).all()) and abs(float(weights.sum()) - 1) < 1e-10,
        "sampling mass does not sum to one",
    )
    cells_summary = [
        dict(task=task, layout=layout, demos=len(demos), probability=1 / len(cells))
        for (task, layout), demos in sorted(cells.items())
    ]
    return weights, cells_summary


class CompleteWindows(Dataset):
    def __init__(self, base, expected_demos, expected_cells=8):
        self.base = base
        self.indices, starts, ends, self.lineage = [], [], [], []
        for start, end, source in zip(base.starts, base.ends, base.lineage, strict=True):
            if end - start < 50:
                continue
            begin = len(self.indices)
            self.indices.extend(range(start, end - 49))
            starts.append(begin)
            ends.append(len(self.indices))
            self.lineage.append(source)
        self.weights, cells_summary = demonstration_weights(
            starts, ends, self.lineage, expected_demos, expected_cells
        )
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.num_frames, self.num_episodes = len(self.indices), len(starts)
        self.repo_id, self.fps = base.repo_id, base.fps
        self.episodes, self.absolute_to_relative_idx = list(range(len(starts))), None
        self.meta = copy(base.meta)
        self.meta.info = copy(base.meta.info)
        self.meta.info.total_frames = len(self)
        self.meta.info.total_episodes = self.num_episodes
        self.meta.episodes = dict(
            dataset_from_index=starts,
            dataset_to_index=ends,
            episode_index=self.episodes,
            length=[b - a for a, b in zip(starts, ends)],
        )
        self.summary = dict(
            original_demos=expected_demos,
            complete_window_starts=len(self),
            retained_segments=base.num_episodes,
            eligible_segments=self.num_episodes,
            cells=cells_summary,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.base[int(self.indices[index])]
        require(not bool(item["action_is_pad"].any()), "window crossed a retained segment boundary")
        return item


def sampler_class(training, *, weights=None):
    sampling_weights = training.weights if weights is None else weights

    class BalancedSampler(Sampler):
        def __init__(
            self,
            starts,
            ends,
            *,
            episode_indices_to_use,
            drop_n_last_frames,
            shuffle,
            seed,
            absolute_to_relative_idx,
        ):
            require(
                starts == training.meta.episodes["dataset_from_index"]
                and ends == training.meta.episodes["dataset_to_index"]
                and drop_n_last_frames == 0
                and shuffle
                and absolute_to_relative_idx is None,
                "upstream sampler invocation differs from the retained-window contract",
            )
            require(
                episode_indices_to_use == training.episodes, "sampler episode selection differs"
            )
            self.seed, self.epoch = seed, 0

        def __len__(self):
            return len(training)

        def __iter__(self):
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            self.epoch += 1
            return iter(
                torch.multinomial(
                    sampling_weights, len(self), replacement=True, generator=generator
                ).tolist()
            )

    return BalancedSampler
