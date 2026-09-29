"""Revalidate and extract sealed Xbox episodes; never modify a failed original."""

import argparse
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .action_audit import verify_telemetry
from .raw_io import write_json


def read(path):
    return json.loads(Path(path).read_text())


class RawDatasetView:
    """Minimal audit reader over raw frames, without Dataset encoding or hardware."""

    def __init__(self, root):
        self.root = Path(root)
        self.features = read(self.root / "raw.json")["features"]
        self.meta = SimpleNamespace(
            depth_keys=[
                k for k, v in self.features.items() if (v.get("info") or {}).get("is_depth_map")
            ]
        )
        self.rows = []
        attempts = sorted(
            (read(p / "result.json")["episode_index"], p, read(p / "result.json"))
            for p in (self.root / "attempts").iterdir()
            if read(p / "result.json")["status"] == "saved"
        )
        if not attempts or [i for i, _, _ in attempts] != list(range(len(attempts))):
            raise ValueError("saved episode indices invalid")
        for index, path, result in attempts:
            count = 0
            for line in (path / "frames.jsonl").open():
                row = json.loads(line)
                if row["frame_index"] != count:
                    raise ValueError("noncontiguous raw frames")
                values = row["values"]
                for key in ("action", "observation.state"):
                    if np.asarray(values[key]).shape != (7,) or not np.isfinite(values[key]).all():
                        raise ValueError("invalid raw state/action")
                expected_images = {
                    k for k, v in self.features.items() if v["dtype"] in ("image", "video")
                }
                if set(row["files"]) != expected_images:
                    raise ValueError("missing raw image streams")
                for key, name in row["files"].items():
                    pixels = np.load(path / name, allow_pickle=False)
                    if (
                        list(pixels.shape) != list(self.features[key]["shape"])
                        or not np.isfinite(pixels).all()
                    ):
                        raise ValueError("invalid raw image shape/pixels")
                self.rows.append((index, path, row))
                count += 1
            if count == 0 or count != result["frames"]:
                raise ValueError("raw frame count differs")
        self.num_frames, self.num_episodes = len(self.rows), len(attempts)

    def select_columns(self, columns):
        return self

    def __getitem__(self, index):
        episode, path, row = self.rows[index]
        frame = dict(row["values"], episode_index=episode, frame_index=row["frame_index"])
        frame.update(
            {k: np.load(path / row["files"][k], allow_pickle=False) for k in self.meta.depth_keys}
        )
        return frame


def audit_raw(root):
    return verify_telemetry(root, RawDatasetView(root))


def recover_sealed(source, output, episodes):
    source, output = Path(source).resolve(), Path(output).resolve()
    original_capture = read(source / "capture.json")
    if original_capture["status"] != "failed":
        raise ValueError(
            "recovery requires an ended failed capture; active capture is not eligible"
        )
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("select distinct sealed episode indices explicitly")
    if output.exists():
        raise FileExistsError(output)
    selected = {}
    for path in (source / "attempts").iterdir():
        result = read(path / "result.json")
        if result["status"] == "saved" and result["episode_index"] in episodes:
            if result["episode_index"] in selected:
                raise ValueError("ambiguous sealed episode")
            selected[result["episode_index"]] = (path, result)
    if set(selected) != set(episodes):
        raise ValueError("requested episode is not sealed")
    remap = {old: new for new, old in enumerate(sorted(episodes))}
    output.mkdir(parents=True)
    write_json(output / "capture.json", {"status": "recovering"})
    report = dict(
        status="validating",
        source=str(source),
        original_capture=original_capture,
        episodes=[],
        excluded_unsealed=True,
    )
    try:
        shutil.copy2(source / "raw.json", output / "raw.json")
        for old, (path, result) in sorted(selected.items()):
            dest = output / "attempts" / path.name
            shutil.copytree(path, dest)
            write_json(dest / "result.json", dict(result, episode_index=remap[old]))
            report["episodes"].append(
                dict(
                    episode_index=remap[old],
                    source_episode_index=old,
                    source_attempt=str(path),
                    frames=result["frames"],
                )
            )
        seen = set()
        for log in sorted((source / "telemetry").glob("*/events.jsonl")):
            events = [json.loads(line) for line in log.open()]
            saved = {
                (e["episode_index"], e["attempt"]): e
                for e in events
                if e["event"] == "episode_saved" and e["episode_index"] in selected
            }
            # A post-seal fault may precede episode_saved, but the persisted save
            # request + sealed result + full revalidation can recover that episode.
            for e in events:
                if e["event"] == "episode_save_requested" and e["episode_index"] in selected:
                    saved.setdefault(
                        (e["episode_index"], e["attempt"]),
                        dict(
                            e,
                            event="episode_saved",
                            frames=selected[e["episode_index"]][1]["frames"],
                        ),
                    )
            if not saved:
                continue
            if seen.intersection(k[0] for k in saved):
                raise ValueError("ambiguous telemetry episode ownership")
            seen.update(k[0] for k in saved)
            directory = output / "telemetry" / log.parent.name
            directory.mkdir(parents=True)
            retained = []
            for e in events:
                key = (e.get("episode_index"), e.get("attempt"))
                if e["event"] in ("session", "hold_reference", "teleoperation_config") or (
                    key in saved
                    and e["event"]
                    in (
                        "frame_pending",
                        "episode_started",
                        "episode_save_requested",
                        "episode_outcome",
                    )
                ):
                    entry = copy.deepcopy(e)
                    if "episode_index" in entry:
                        entry["source_episode_index"] = entry["episode_index"]
                        entry["episode_index"] = remap[entry["episode_index"]]
                    retained.append(entry)
                    for item in entry.get("raw_depth", {}).values():
                        target = output / item["path"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source / item["path"], target)
            for (old, attempt), e in saved.items():
                count = sum(
                    v["event"] == "frame_pending"
                    and v.get("episode_index") == remap[old]
                    and v.get("attempt") == attempt
                    for v in retained
                )
                if count != selected[old][1]["frames"] or e["frames"] != count:
                    raise ValueError("seal and telemetry frame counts differ")
                retained.append(
                    dict(
                        e, event="episode_saved", episode_index=remap[old], source_episode_index=old
                    )
                )
                if not any(
                    v["event"] == "episode_outcome" and v.get("episode_index") == remap[old]
                    for v in retained
                ):
                    request = next(
                        (
                            v
                            for v in retained
                            if v["event"] == "episode_save_requested"
                            and v["episode_index"] == remap[old]
                        ),
                        None,
                    )
                    retained.append(
                        dict(
                            event="episode_outcome",
                            episode_index=remap[old],
                            attempt=attempt,
                            task_outcome="unknown" if request is None else request["task_outcome"],
                            position_id=None if request is None else request["position_id"],
                            data_valid=True,
                            outcome_source="not_recorded"
                            if request is None
                            else "persisted_save_request",
                        )
                    )
            retained.append(
                dict(
                    event="recovered_sealed",
                    source_events=str(log),
                    original_failures=[e for e in events if e["event"] == "failed"],
                )
            )
            (directory / "events.jsonl").write_text(
                "".join(json.dumps(e, allow_nan=False) + "\n" for e in retained)
            )
            write_json(
                directory / "complete.json",
                dict(
                    status="complete",
                    frames=sum(e["frames"] for e in saved.values()),
                    episodes=[remap[k[0]] for k in saved],
                    recovered_from=str(log),
                ),
            )
        if seen != set(selected):
            raise ValueError("sealed episode lacks persisted save evidence")
        report.update(audit=audit_raw(output), status="complete")
        write_json(output / "recovery.json", report)
        write_json(output / "capture.json", {"status": "closed", "recovered_from": str(source)})
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        write_json(output / "recovery.json", report)
        write_json(output / "capture.json", {"status": "failed"})
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode", type=int, action="append", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(recover_sealed(args.raw_root, args.output, args.episode), indent=2))
    return 0
