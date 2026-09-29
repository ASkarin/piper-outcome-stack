"""Offline pause candidates for human review. Never removes or changes samples."""

from pathlib import Path
import argparse
import json
import math
import numpy as np


def candidates(root, *, min_duration_s=1.0):
    root = Path(root)
    if not math.isfinite(min_duration_s) or min_duration_s <= 0:
        raise ValueError("duration must be finite and positive")
    states = {}
    if (root / "raw.json").exists():
        if json.loads((root / "capture.json").read_text())["status"] != "closed":
            raise ValueError("pause review requires closed capture")
        for path in root.glob("attempts/*"):
            result = json.loads((path / "result.json").read_text())
            if result["status"] != "saved":
                continue
            for line in (path / "frames.jsonl").open():
                row = json.loads(line)
                states[result["episode_index"], row["frame_index"]] = np.array(
                    row["values"]["observation.state"]
                )
    else:
        import pyarrow.parquet as pq

        for path in (root / "data").rglob("*.parquet"):
            table = pq.read_table(
                path, columns=["episode_index", "frame_index", "observation.state"]
            ).to_pydict()
            for ep, frame, state in zip(
                table["episode_index"], table["frame_index"], table["observation.state"]
            ):
                states[int(ep), int(frame)] = np.asarray(state)
    groups = []
    segment = []
    key = None
    tolerance = np.array([math.radians(0.1)] * 6 + [0.0005])

    def flush():
        if len(segment) < 2:
            return
        first, last = segment[0], segment[-1]
        duration = last["time"] - first["time"]
        if duration >= min_duration_s:
            groups.append(
                dict(
                    candidate_id=len(groups),
                    episode_index=first["episode"],
                    first_frame=first["frame"],
                    last_frame=last["frame"],
                    frames=len(segment),
                    start_monotonic_s=first["time"],
                    end_monotonic_s=last["time"],
                    duration_s=duration,
                    selected_for_removal=False,
                )
            )

    for path in sorted((root / "telemetry").glob("*/events.jsonl")):
        events = [json.loads(line) for line in path.open()]
        saved = {
            (e["episode_index"], e["attempt"]) for e in events if e["event"] == "episode_saved"
        }
        for e in events:
            if e["event"] != "frame_pending" or (e["episode_index"], e["attempt"]) not in saved:
                continue
            a = e["action"]
            ep = e["episode_index"]
            frame = e["frame_index"]
            state = states.get((ep, frame))
            valid = (
                a.get("result") == "holding"
                and a.get("hold_confirmed") is True
                and state is not None
                and state.shape == (7,)
                and np.isfinite(state).all()
            )
            current_key = (
                str(path),
                ep,
                a.get("hold_id"),
                json.dumps(a.get("retained_gripper_command"), sort_keys=True),
            )
            t = e["observation"]["observed_monotonic_s"]
            contiguous = bool(
                segment and frame == segment[-1]["frame"] + 1 and t > segment[-1]["time"]
            )
            stable = (
                bool(
                    segment
                    and np.all(
                        np.ptp(np.array([v["state"] for v in segment] + [state]), axis=0)
                        <= tolerance
                    )
                )
                if valid
                else False
            )
            if not valid or current_key != key or not contiguous or not stable:
                flush()
                segment = []
            if valid:
                segment.append(dict(episode=ep, frame=frame, time=t, state=state))
            key = current_key
        flush()
        segment = []
        key = None
    return dict(
        source=str(root.resolve()),
        status="review_required",
        removed_frames=0,
        criteria=dict(min_duration_s=min_duration_s, joint_range_deg=0.1, gripper_range_mm=0.5),
        note="Candidate only: object settling/contact and image changes require human review. No smoothing or deletion performed.",
        candidates=groups,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-duration-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    report = candidates(args.root, min_duration_s=args.min_duration_s)
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return 0
