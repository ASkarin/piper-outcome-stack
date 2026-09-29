"""Offline next-observed-state labels. No robot or camera instances are created."""

from .raw_io import write_json
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
from .safety import ACTION_KEYS, load_motion_safety
from .teach_data import (
    SCHEMA,
    RULE,
    read_json,
    load_attempt,
    read_pixels,
    timing_summary,
)


ACTION_RULE = "next_observed_state_gripper_nonnegative_v2"


def state(row):
    return np.asarray([*row["joint_rad"], row["gripper_m"]], dtype=np.float32)


def action_state(row, rule=ACTION_RULE):
    value = state(row)
    if not np.isfinite(value).all():
        raise ValueError("non-finite candidate action")
    if rule == ACTION_RULE:
        value[6] = max(0.0, value[6])
    elif rule != RULE:
        raise ValueError("unknown teach action rule")
    return value


def gripper_mapping(row):
    return dict(
        raw_next_gripper_m=row["gripper_m"],
        action_gripper_m=float(action_state(row)[6]),
        gripper_zero_mapped=row["gripper_m"] < 0,
    )


def mapping_summary(rows):
    values = [row["gripper_m"] for row in rows]
    return dict(
        mapped_frames=sum(v < 0 for v in values),
        raw_target_min_m=min(values),
        rule="action.gripper=max(0,next_raw_gripper); observation unchanged",
    )


def check_candidate(observation, target, safety, *, rule=ACTION_RULE):
    from .execution_constraints import check_execution_target

    check_execution_target(
        [*observation["joint_rad"], observation["gripper_m"]], action_state(target, rule), safety
    )


def features_for(config, first_pixels, video):
    from lerobot.utils.feature_utils import hw_to_dataset_features

    numeric = dict.fromkeys(ACTION_KEYS, float)
    shapes = {k: (*v.shape, 1) if v.ndim == 2 else v.shape for k, v in first_pixels.items()}
    features = {
        **hw_to_dataset_features(numeric, "action", use_video=video),
        **hw_to_dataset_features({**numeric, **shapes}, "observation", use_video=video),
    }
    for f in features.values():
        if f.get("info", {}).get("is_depth_map"):
            f["dtype"] = "image"
            f["info"]["depth_unit"] = "m"
    return features


def convert(
    source, output, repo_id, safety_path, *, video=True, include_depth=False, positions=None
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from .episode_save import compress_episode_depth

    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("converted Dataset must be independent of the raw session directory")
    if output.exists():
        raise FileExistsError(output)
    manifest = read_json(source / "session.json")
    if manifest.get("schema") != SCHEMA or manifest.get("source") != "manual_teach":
        raise ValueError("not a raw teach session; Xbox data cannot be converted here")
    config = manifest["config"]
    safety = load_motion_safety(Path(safety_path))
    requested = None if positions is None else set(positions)
    if requested is not None and (
        not requested or any(not isinstance(p, str) or not p for p in requested)
    ):
        raise ValueError("positions must contain explicit nonempty identifiers")
    selected = []
    for path in sorted(source.glob("attempt-*")):
        result = read_json(path / "result.json")
        if result.get("status") == "saved":
            if requested is not None and result["position_id"] not in requested:
                continue
            result, rows = load_attempt(path, config)
            connection = result.get("source_connection")
            if connection is not None:
                closed = read_json(source / connection / "result.json")
                if not closed.get("finished_at_utc"):
                    raise ValueError("end the hardware session before conversion")
            selected.append((path, result, rows))
    if requested is not None and requested != {result["position_id"] for _, result, _ in selected}:
        raise ValueError("requested positions have no saved attempts")
    if not selected:
        raise ValueError("no saved valid teach attempts")
    # Validate labels before constructing the Dataset. Never clip a demonstrated path.
    for path, _, rows in selected:
        for a, b in zip(rows, rows[1:]):
            try:
                check_candidate(a, b, safety)
            except ValueError as exc:
                raise ValueError(f"{path.name} frame {a['frame_index']}: {exc}") from exc
    first_pixels = read_pixels(selected[0][0], selected[0][2][0])
    exported_streams = [
        key
        for key in first_pixels
        if include_depth or selected[0][2][0]["camera"][key]["stream"] == "color"
    ]
    if include_depth and not any(
        m["stream"] == "depth" for m in selected[0][2][0]["camera"].values()
    ):
        raise ValueError("source has no captured depth to export")
    features = features_for(config, {k: first_pixels[k] for k in exported_streams}, video)
    dataset = LeRobotDataset.create(
        repo_id,
        config["fps"],
        features=features,
        root=output,
        robot_type="outcome_piper",
        use_videos=video,
        image_writer_threads=2,
        streaming_encoding=False,
    )
    info = dict(
        source="manual_teach",
        rule=ACTION_RULE,
        status="converting",
        original_root=str(source),
        selected_positions=None if requested is None else sorted(requested),
        gripper_mapping=mapping_summary([b for _, _, rows in selected for b in rows[1:]]),
        raw_depth_storage="source_archive",
        exported_streams=exported_streams,
        raw_schema=SCHEMA,
        safety=asdict(safety),
        hardware_replay_verified=False,
        sdk_commands_sent=False,
        scene=config["robot"]["scene"],
    )
    write_json(output / "teach-conversion.json", info)
    evidence = output / "teach"
    evidence.mkdir()
    shutil.copyfile(source / "session.json", evidence / "session.json")
    index = []
    try:
        for episode, (path, result, rows) in enumerate(selected):
            raw = evidence / path.name
            # Numeric/RGB evidence remains portable; Z16 stays in the one source archive.
            raw.mkdir()
            for name in ("result.json", "samples.jsonl"):
                shutil.copyfile(path / name, raw / name)
            for row in rows:
                for key, file in row["files"].items():
                    if row["camera"][key]["stream"] == "color":
                        shutil.copyfile(path / file, raw / file)
            connection = result.get("source_connection")
            if connection is not None and not (evidence / connection).exists():
                shutil.copytree(source / connection, evidence / connection)
            for a, b in zip(rows, rows[1:]):
                pixels = read_pixels(path, a, exported_streams)
                frame = {
                    "observation.state": state(a),
                    "action": action_state(b),
                    "task": config["task"],
                }
                for key, array in pixels.items():
                    if a["camera"][key]["stream"] == "depth":
                        array = (array.astype(np.float32) * a["camera"][key]["depth_scale_m"])[
                            ..., None
                        ]
                    frame[f"observation.images.{key}"] = array
                dataset.add_frame(frame)
                index.append(
                    dict(
                        episode_index=episode,
                        frame_index=a["frame_index"],
                        raw_attempt=path.name,
                        observation_frame=a["frame_index"],
                        target_frame=b["frame_index"],
                        rule=ACTION_RULE,
                        delta_time_s=b["sampled_monotonic_s"] - a["sampled_monotonic_s"],
                        **gripper_mapping(b),
                        sdk_call_start=None,
                        sdk_call_end=None,
                        sdk_result=None,
                    )
                )
            dataset.writer._wait_image_writer()
            compress_episode_depth(
                dataset.writer.episode_buffer,
                dataset.features,
                (),
                evidence / f"compression-{episode}.json",
            )
            dataset.save_episode()
            write_json(
                evidence / f"episode-{episode}.json",
                dict(
                    raw_attempt=path.name,
                    task_outcome=result["task_outcome"],
                    note=result.get("note", ""),
                    data_valid=True,
                    timing=timing_summary(rows),
                ),
            )
        dataset.finalize()
        write_json(evidence / "index.json", index)
        info["status"] = "verifying"
        write_json(output / "teach-conversion.json", info)
        loaded = LeRobotDataset(repo_id, root=output, depth_output_unit="m")
        counts = audit_teach(output, loaded, allow_verifying=True)
        info.update(status="complete", **counts)
        write_json(output / "teach-conversion.json", info)
        return counts
    except BaseException as exc:
        info.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "teach-conversion.json", info)
        try:
            dataset.finalize()
        except Exception as cleanup:
            info["cleanup_error"] = str(cleanup)
            write_json(output / "teach-conversion.json", info)
        raise


def audit_teach(root, dataset, *, allow_verifying=False, raw_source=None):
    root = Path(root)
    info = read_json(root / "teach-conversion.json")
    valid = ("complete", "verifying") if allow_verifying else ("complete",)
    if (
        info.get("source") != "manual_teach"
        or info.get("rule") not in (RULE, ACTION_RULE)
        or info.get("status") not in valid
    ):
        raise RuntimeError("incomplete or invalid teach conversion")
    if (root / "telemetry").exists():
        raise RuntimeError("mixed Xbox/teach evidence is unsupported")
    evidence = root / "teach"
    manifest = read_json(evidence / "session.json")
    if manifest.get("schema") != SCHEMA or manifest.get("source") != "manual_teach":
        raise RuntimeError("missing original teach provenance")
    config = manifest["config"]
    source_archive = info.get("raw_depth_storage") == "source_archive"
    archive = Path(raw_source or info["original_root"])
    rgb_streams = {name for name, cfg in config["robot"]["cameras"].items() if cfg["use_rgb"]}
    if source_archive and dataset.meta.depth_keys:
        if read_json(archive / "session.json") != manifest:
            raise RuntimeError("raw depth archive does not match this Dataset")
    if source_archive:
        columns = {
            k.removeprefix("observation.images.")
            for k, v in dataset.features.items()
            if v["dtype"] in ("image", "video")
        }
        if columns != set(info["exported_streams"]):
            raise RuntimeError("exported stream manifest and Dataset differ")
    if dataset.fps != config["fps"] or dataset.meta.robot_type != "outcome_piper":
        raise RuntimeError("Dataset identity or fps mismatch")
    if dataset.features["action"]["names"] != list(ACTION_KEYS) or dataset.features[
        "observation.state"
    ]["names"] != list(ACTION_KEYS):
        raise RuntimeError("seven-dimensional state/action order differs")
    from .safety import MotionSafety

    safety = MotionSafety(**info["safety"])
    index = read_json(evidence / "index.json")
    if len(index) != dataset.num_frames:
        raise RuntimeError("teach provenance/Dataset count mismatch")
    rule = info["rule"]
    mapped_targets = []
    cache = {}
    seen = set()
    image_keys = [k for k, v in dataset.features.items() if v["dtype"] == "image"]
    table = dataset.select_columns(
        ["episode_index", "frame_index", "observation.state", "action", *image_keys]
    )
    for i, entry in enumerate(index):
        key = (entry["episode_index"], entry["frame_index"])
        if key in seen:
            raise RuntimeError("duplicate teach Dataset row")
        seen.add(key)
        path = entry["raw_attempt"]
        if path not in cache:
            cache[path] = load_attempt(
                evidence / path, config, pixel_streams=rgb_streams if source_archive else None
            )
            if source_archive and dataset.meta.depth_keys:
                if (archive / path / "samples.jsonl").read_bytes() != (
                    evidence / path / "samples.jsonl"
                ).read_bytes():
                    raise RuntimeError("raw depth archive samples differ")
        result, rows = cache[path]
        if (
            info.get("selected_positions") is not None
            and result["position_id"] not in info["selected_positions"]
        ):
            raise RuntimeError("teach position selection differs from provenance")
        n = entry["observation_frame"]
        if (
            entry["rule"] != rule
            or entry["target_frame"] != n + 1
            or n != entry["frame_index"]
            or n + 1 >= len(rows)
        ):
            raise RuntimeError("invalid next-frame label provenance")
        if any(entry[k] is not None for k in ("sdk_call_start", "sdk_call_end", "sdk_result")):
            raise RuntimeError("teach row invents SDK command evidence")
        a, b = rows[n : n + 2]
        if entry["delta_time_s"] != b["sampled_monotonic_s"] - a["sampled_monotonic_s"]:
            raise RuntimeError("teach label time interval mismatch")
        check_candidate(a, b, safety, rule=rule)
        if rule == ACTION_RULE:
            if any(entry.get(k) != v for k, v in gripper_mapping(b).items()):
                raise RuntimeError("teach gripper mapping provenance mismatch")
            mapped_targets.append(b)
        actual = table[i]
        if key != (int(actual["episode_index"]), int(actual["frame_index"])):
            raise RuntimeError("teach Dataset row order mismatch")
        if not np.array_equal(
            np.asarray(actual["action"]), action_state(b, rule)
        ) or not np.array_equal(np.asarray(actual["observation.state"]), state(a)):
            raise RuntimeError("teach state/action differs from original feedback")
        pixels = read_pixels(evidence / path, a, rgb_streams if source_archive else None)
        if source_archive and dataset.meta.depth_keys:
            depth_streams = {k.removeprefix("observation.images.") for k in dataset.meta.depth_keys}
            pixels.update(read_pixels(archive / path, a, depth_streams))
        for stream in image_keys:
            if stream in dataset.meta.depth_keys:
                continue
            name = stream.removeprefix("observation.images.")
            actual_rgb = np.asarray(actual[stream])
            expected_rgb = pixels[name].astype(np.float32) / 255.0
            if actual_rgb.shape == (3, *expected_rgb.shape[:2]):
                actual_rgb = actual_rgb.transpose(1, 2, 0)
            if not np.array_equal(actual_rgb, expected_rgb):
                raise RuntimeError("converted RGB differs from raw image")
        if dataset.meta.video_keys:
            decoded = dataset[i]
            for stream in dataset.meta.video_keys:
                name = stream.removeprefix("observation.images.")
                rgb = np.asarray(decoded[stream])
                if rgb.shape != (3, *pixels[name].shape[:2]) or not np.isfinite(rgb).all():
                    raise RuntimeError("converted video frame cannot be decoded correctly")
        for stream in dataset.meta.depth_keys:
            name = stream.removeprefix("observation.images.")
            raw = pixels[name]
            expected = raw.astype(np.float32) * a["camera"][name]["depth_scale_m"]
            if raw.dtype != np.uint16 or not np.array_equal(
                np.asarray(actual[stream]).squeeze(0), expected
            ):
                raise RuntimeError("converted depth differs from raw Z16")
    # Verify exactly N-1 rows per saved attempt and no attempt split across episodes.
    episodes = {}
    for entry in index:
        episodes.setdefault(entry["episode_index"], set()).add(entry["raw_attempt"])
    if set(episodes) != set(range(dataset.num_episodes)) or any(
        len(v) != 1 for v in episodes.values()
    ):
        raise RuntimeError("invalid teach episode mapping")
    if len(cache) != dataset.num_episodes or sum(len(r) - 1 for _, r in cache.values()) != len(
        index
    ):
        raise RuntimeError("teach terminal frame or episode count mismatch")
    if info.get("selected_positions") is not None and set(info["selected_positions"]) != {
        result["position_id"] for result, _ in cache.values()
    }:
        raise RuntimeError("teach position selection is incomplete")
    if rule == ACTION_RULE and info.get("gripper_mapping") != mapping_summary(mapped_targets):
        raise RuntimeError("teach gripper mapping summary mismatch")
    return {"episodes": dataset.num_episodes, "frames": dataset.num_frames}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--safety", type=Path, required=True)
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="materialize metric TIFF from the retained Z16 archive",
    )
    parser.add_argument(
        "--position",
        action="append",
        dest="positions",
        help="include only saved attempts at these positions; repeat for multiple positions",
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            convert(
                args.source,
                args.output,
                args.repo_id,
                args.safety,
                video=args.video,
                include_depth=args.include_depth,
                positions=args.positions,
            )
        )
    )
    return 0
