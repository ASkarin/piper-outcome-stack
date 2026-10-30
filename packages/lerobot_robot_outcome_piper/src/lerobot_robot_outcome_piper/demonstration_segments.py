"""Identify protection/operator waits and preserve continuous demonstration segments.

Raw frames are never removed. The same rule annotates live telemetry and builds
the offline conversion map, including existing captures with transient reasons.
"""

import json
from pathlib import Path


class DemonstrationTracker:
    def __init__(self):
        self.protection_reason = None
        self.previous_eligible = True
        self.segment_id = 0

    def observe(self, action):
        reason = action.get("rejection_reason") or action.get("reason")
        state = action.get("demonstration_pause")
        if state is not None:
            self.protection_reason = state["protection_reason"]
            manual = state["manual"]
            if reason and self.protection_reason != reason:
                raise ValueError("protection metadata differs from the rejected input")
        else:
            manual = False
            if reason:
                self.protection_reason = reason
            elif action.get("result") in ("sdk_returned", "holding") and any(
                c.get("name") in ("move_j", "move_gripper_m") and c.get("result") == "sdk_returned"
                for c in action.get("commands", [])
            ):
                # A new accepted gripper command also resumes useful data while
                # the arm remains held. A retained command is not a new action.
                self.protection_reason = None
        eligible = not manual and self.protection_reason is None
        if eligible and not self.previous_eligible:
            self.segment_id += 1
        self.previous_eligible = eligible
        return {
            "eligible": eligible,
            "segment_id": self.segment_id,
            "protection_reason": self.protection_reason,
            "operator_paused": manual,
        }


def segments_from_telemetry(root):
    """Map saved attempts to contiguous eligible ranges; exclude discarded attempts."""
    groups, saved = {}, {}
    for path in sorted((Path(root) / "telemetry").glob("*/events.jsonl")):
        for line in path.open():
            event = json.loads(line)
            if event["event"] not in ("frame_pending", "episode_saved"):
                continue
            key = (path.parent.name, event["episode_index"], event["attempt"])
            if event["event"] == "episode_saved":
                if event["episode_index"] in saved:
                    raise ValueError("duplicate saved source episode")
                saved[event["episode_index"]] = (key, event["frames"])
                continue
            group = groups.setdefault(
                key, {"tracker": DemonstrationTracker(), "frames": 0, "runs": []}
            )
            frame = event["frame_index"]
            if frame != group["frames"]:
                raise ValueError("noncontiguous source telemetry frames")
            annotation = group["tracker"].observe(event["action"])
            if "demonstration" in event and event["demonstration"] != annotation:
                raise ValueError("recorded demonstration annotation differs from action evidence")
            kind = (
                annotation["eligible"],
                annotation["protection_reason"],
                annotation["operator_paused"],
            )
            runs = group["runs"]
            if not runs or runs[-1]["kind"] != kind:
                runs.append({"kind": kind, "first_frame": frame, "end_frame_exclusive": frame})
            runs[-1]["end_frame_exclusive"] = frame + 1
            group["frames"] += 1
    segments, excluded, source_frames = [], [], 0
    if list(sorted(saved)) != list(range(len(saved))):
        raise ValueError("saved source episodes must be contiguous")
    for episode, (key, count) in sorted(saved.items()):
        group = groups[key]
        if count != group["frames"]:
            raise ValueError("saved frame count differs from telemetry")
        source_frames += count
        for run in group["runs"]:
            eligible, reason, manual = run["kind"]
            entry = {
                "source_session": key[0],
                "source_episode_index": episode,
                "source_attempt": key[2],
                "first_frame": run["first_frame"],
                "end_frame_exclusive": run["end_frame_exclusive"],
                "frames": run["end_frame_exclusive"] - run["first_frame"],
            }
            if eligible:
                # Adjacent eligible runs belong to the same physical-time segment.
                if (
                    segments
                    and all(
                        segments[-1][k] == entry[k]
                        for k in ("source_session", "source_episode_index", "source_attempt")
                    )
                    and segments[-1]["end_frame_exclusive"] == entry["first_frame"]
                ):
                    segments[-1]["end_frame_exclusive"] = entry["end_frame_exclusive"]
                    segments[-1]["frames"] += entry["frames"]
                else:
                    segments.append({"episode_index": len(segments), **entry})
            else:
                excluded.append({**entry, "protection_reason": reason, "operator_paused": manual})
    return {
        "schema": "piper-demonstration-segments-v1",
        "source_frames": source_frames,
        "retained_frames": sum(s["frames"] for s in segments),
        "excluded_frames": sum(s["frames"] for s in excluded),
        "segments": segments,
        "excluded": excluded,
    }
