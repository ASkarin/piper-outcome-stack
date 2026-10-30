"""Read an explicitly retained-frame view without rewriting source LeRobot data."""

from piper_policy import statistics as policy_statistics

from copy import copy
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset


def retained_indices(episodes, source_starts, source_ends):
    indices, starts, ends, lineage = [], [], [], []
    rows = sorted(episodes, key=lambda row: row["provenance"]["merged_episode"])
    if len(rows) != len(source_starts) or len(source_starts) != len(source_ends):
        raise ValueError("frame map and source episode counts differ")
    for episode, row in enumerate(rows):
        if row["provenance"]["merged_episode"] != episode:
            raise ValueError("frame map episode identifiers must be contiguous")
        ids = np.asarray(row["original_frame_indices"], dtype=np.int64)
        length = int(source_ends[episode] - source_starts[episode])
        if row["source_frames"] != length:
            raise ValueError("frame map source length differs from dataset")
        if not len(ids) or ids[0] < 0 or ids[-1] >= length or np.any(np.diff(ids) <= 0):
            raise ValueError("retained frames must be nonempty, increasing and in the episode")
        starts.append(len(indices))
        indices.extend((int(source_starts[episode]) + ids).tolist())
        ends.append(len(indices))
        lineage.append(dict(row["provenance"]))
    return np.asarray(indices), starts, ends, lineage


def segmented_indices(segments, source_starts, source_ends):
    """Keep explicit continuous intervals as separate future-action horizons."""
    indices, starts, ends, lineage = [], [], [], []
    used = set()
    for row in segments:
        episode = row["dataset_episode"]
        if not 0 <= episode < len(source_starts):
            raise ValueError("segment episode is outside source dataset")
        begin, end = row["first_frame"], row["end_frame_exclusive"]
        length = source_ends[episode] - source_starts[episode]
        if not 0 <= begin < end <= length:
            raise ValueError("segment interval is outside source episode")
        ids = range(source_starts[episode] + begin, source_starts[episode] + end)
        if used.intersection(ids):
            raise ValueError("source segments overlap")
        used.update(ids)
        starts.append(len(indices))
        indices.extend(ids)
        ends.append(len(indices))
        lineage.append(dict(row["provenance"]))
    if not indices:
        raise ValueError("source segments are empty")
    return np.asarray(indices), starts, ends, lineage


class RetainedDataset(Dataset):
    """Concatenate frozen source views; future actions stop at each episode boundary."""

    def __init__(self, sources, *, chunk_size=50, repo_id="local/piper-retained"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import pandas as pd

        if chunk_size <= 0:
            raise ValueError("chunk size must be positive")
        self.chunk_size = chunk_size
        self.parts, self.starts, self.ends, self.lineage = [], [], [], []
        self.part_ends = []
        self.actions, self.states = [], []
        offset = 0
        for source in sources:
            root = Path(source["root"])
            base = LeRobotDataset(
                source["repo_id"], root=root, video_backend="pyav", return_uint8=True
            )
            if "segments" in source:
                ids, starts, ends, lineage = segmented_indices(
                    source["segments"],
                    list(base.meta.episodes["dataset_from_index"]),
                    list(base.meta.episodes["dataset_to_index"]),
                )
                for row in source["segments"]:
                    if base.meta.episodes["tasks"][row["dataset_episode"]] != [
                        row["provenance"]["task"]
                    ]:
                        raise ValueError("segment task differs from source dataset")
            else:
                mapping = json.loads(Path(source["frame_map"]).read_text())
                rows = [
                    row
                    for row in mapping["episodes"]
                    if row["provenance"]["split"] == source["map_split"]
                ]
                ids, starts, ends, lineage = retained_indices(
                    rows,
                    list(base.meta.episodes["dataset_from_index"]),
                    list(base.meta.episodes["dataset_to_index"]),
                )
            table = pd.concat(
                [
                    pd.read_parquet(path, columns=["index", "observation.state", "action"])
                    for path in sorted((root / "data").rglob("*.parquet"))
                ],
                ignore_index=True,
            ).sort_values("index")
            if not np.array_equal(table["index"], np.arange(base.num_frames)):
                raise ValueError("source frame indices are not contiguous")
            if base.fps != 50:
                raise ValueError("this view requires the recorded 50Hz dataset contract")
            if self.parts and base.meta.features != self.parts[0][0].meta.features:
                raise ValueError("source dataset feature contracts differ")
            self.parts.append((base, ids))
            self.starts.extend([offset + x for x in starts])
            self.ends.extend([offset + x for x in ends])
            self.lineage.extend(lineage)
            self.actions.append(np.stack(table["action"])[ids].astype(np.float32))
            self.states.append(np.stack(table["observation.state"])[ids].astype(np.float32))
            offset += len(ids)
            self.part_ends.append(offset)
        if not self.parts:
            raise ValueError("at least one source view is required")
        self.actions = np.concatenate(self.actions)
        self.states = np.concatenate(self.states)
        self.num_frames = len(self.actions)
        self.num_episodes = len(self.starts)
        self.episodes = list(range(self.num_episodes))
        self.absolute_to_relative_idx = None
        self.repo_id, self.fps = repo_id, 50
        self.meta = copy(self.parts[0][0].meta)
        self.meta.info = copy(self.meta.info)
        self.meta.info.total_frames = self.num_frames
        self.meta.info.total_episodes = self.num_episodes
        self.meta.repo_id = repo_id
        tasks = sorted({task for base, _ in self.parts for task in base.meta.tasks.index})
        self.task_to_index = {task: index for index, task in enumerate(tasks)}
        self.meta.info.total_tasks = len(tasks)
        self.meta.tasks = pd.DataFrame(
            {"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task")
        )
        self.meta.episodes = {
            "dataset_from_index": self.starts,
            "dataset_to_index": self.ends,
            "episode_index": self.episodes,
            "length": [b - a for a, b in zip(self.starts, self.ends)],
        }
        self.meta.stats = dict(self.meta.stats)
        self.meta.stats["observation.state"] = policy_statistics.numeric_stats(self.states)
        self.meta.stats["action"] = policy_statistics.numeric_stats(self.actions)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        if not 0 <= index < self.num_frames:
            raise IndexError(index)
        part = int(np.searchsorted(self.part_ends, index, side="right"))
        local = index - (self.part_ends[part - 1] if part else 0)
        base, ids = self.parts[part]
        item = dict(base[int(ids[local])])
        if "task" in item:
            item["task_index"] = torch.full_like(
                item["task_index"], self.task_to_index[item["task"]]
            )
        episode = int(np.searchsorted(self.ends, index, side="right"))
        future = np.arange(index, index + self.chunk_size)
        end = self.ends[episode]
        item["action"] = torch.from_numpy(self.actions[np.minimum(future, end - 1)].copy())
        item["action_is_pad"] = torch.from_numpy(future >= end)
        item["episode_index"] = torch.full_like(item["episode_index"], episode)
        item["frame_index"] = torch.full_like(item["frame_index"], index - self.starts[episode])
        item["index"] = torch.full_like(item["index"], index)
        item["timestamp"] = torch.full_like(item["timestamp"], (index - self.starts[episode]) / 50)
        return item

    def evaluation_indices(self, samples_per_episode):
        if samples_per_episode <= 0:
            raise ValueError("evaluation samples per episode must be positive")
        return np.concatenate(
            [
                np.linspace(start, end - 1, min(samples_per_episode, end - start), dtype=np.int64)
                for start, end in zip(self.starts, self.ends)
            ]
        ).tolist()


def require_disjoint_sessions(train, development):
    overlap = {x["source_session"] for x in train.lineage} & {
        x["source_session"] for x in development.lineage
    }
    if overlap:
        raise ValueError(f"train/development source sessions overlap: {sorted(overlap)}")
