"""Operator-started read-only RGB/state capture, independent of inference dependencies."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

ACTION_KEYS = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]


def read_config(path):
    import draccus
    from lerobot_robot_outcome_piper.config import OutcomePiperConfig

    values = json.loads(Path(path).read_text())
    robot_values = dict(values.get("robot", values))
    robot_values.pop("type", None)
    config = draccus.decode(OutcomePiperConfig, robot_values)
    if config.execution_mode != "read_only" or config.capture_timing is None:
        raise ValueError("shadow capture requires explicit read_only and capture_timing")
    if config.scene is None or "d435" not in config.cameras:
        raise ValueError("shadow capture requires the existing scene and d435 RGB")
    camera = config.cameras["d435"]
    if not camera.use_rgb or (camera.width, camera.height) != (640, 480):
        raise ValueError("shadow capture requires640x480 RGB")
    return config


def acquire_pairs(
    robot,
    output,
    *,
    pairs=8,
    fps=50,
    spacing=25,
    stride=100,
    clock=time.monotonic,
    sleep=time.sleep,
):
    if robot.config.execution_mode != "read_only":
        raise ValueError("shadow capture cannot use a motion-configured robot")
    if not (type(pairs) is int and pairs > 0 and fps == 50 and 0 < spacing < stride):
        raise ValueError("invalid50Hz shadow pair schedule")
    timing = robot.config.capture_timing
    selected = {i * stride: (i, "first") for i in range(pairs)}
    selected.update({i * stride + spacing: (i, "second") for i in range(pairs)})
    frames = {}
    telemetry = []
    last_sequence = None
    end_tick = (pairs - 1) * stride + spacing
    for tick in range(end_tick + 1):
        start = clock()
        observation = robot.get_observation()
        meta = dict(robot.last_observation_telemetry)
        now = clock()
        age = now - meta["oldest_received_monotonic_s"]
        if (
            meta["quality"] != "checked"
            or not math.isfinite(age)
            or not 0 <= age <= timing.observation_max_age_s
        ):
            raise ValueError("shadow input is stale or not timing checked")
        if last_sequence is not None and meta["sequence"] <= last_sequence:
            raise ValueError("shadow observation sequence did not advance")
        last_sequence = meta["sequence"]
        telemetry.append({"tick": tick, "acquired_monotonic_s": now, "observation": meta})
        if tick in selected:
            pair, kind = selected[tick]
            image = np.asarray(observation["d435"])
            state = np.asarray([observation[k] for k in ACTION_KEYS], dtype=np.float32)
            if (
                image.shape != (480, 640, 3)
                or image.dtype != np.uint8
                or not np.isfinite(state).all()
            ):
                raise ValueError("invalid actual RGB/state input")
            frames.setdefault(pair, {})[kind] = (image.copy(), state.copy(), now, meta["sequence"])
        sleep(max(0, 1 / fps - (clock() - start)))
    # Write only after capture; disk compression does not distort the sampled cadence.
    directory = Path(output) / "pairs"
    directory.mkdir()
    records = []
    for pair in range(pairs):
        first, second = frames[pair]["first"], frames[pair]["second"]
        np.savez_compressed(
            directory / f"{pair:03d}.npz",
            first_image=first[0],
            first_state=first[1],
            second_image=second[0],
            second_state=second[1],
        )
        records.append(
            {
                "pair": pair,
                "first_time": first[2],
                "second_time": second[2],
                "first_sequence": first[3],
                "second_sequence": second[3],
                "actual_spacing_s": second[2] - first[2],
                "observation_ticks": spacing,
            }
        )
    (Path(output) / "observations.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    (Path(output) / "pairs.json").write_text(json.dumps(records, indent=2) + "\n")
    return {
        "pairs": len(records),
        "observations": len(telemetry),
        "actual_spacing_s": [x["actual_spacing_s"] for x in records],
        "motion_commands_sent": 0,
        "motor_enable_executed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check-config", "capture"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs", type=int, default=8)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.mode == "check-config":
        print(
            json.dumps(
                {
                    "status": "config_valid",
                    "device_io": False,
                    "execution_mode": config.execution_mode,
                }
            )
        )
        return
    if args.output is None:
        parser.error("capture requires --output")
    if args.pairs <= 0:
        parser.error("pairs must be positive")
    from lerobot_robot_outcome_piper.robot import OutcomePiper

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2) + "\n"
    )
    result = {
        "status": "started",
        "domain": "live_read_only_capture",
        "hardware_io": True,
        "motion_commands_sent": 0,
        "motor_enable_executed": False,
        "ground_truth_actions_available": False,
    }
    robot = OutcomePiper(config)
    try:
        robot.connect()
        result.update(acquire_pairs(robot, args.output, pairs=args.pairs), status="captured")
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
