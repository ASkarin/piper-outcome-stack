"""Read-only joint/camera timing measurement and offline telemetry summaries."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def load_read_only_config(path):
    import draccus
    from lerobot_robot_outcome_piper.config import OutcomePiperConfig

    values = json.loads(Path(path).read_text())
    values = dict(values.get("robot", values))
    values.pop("type", None)
    cfg = draccus.decode(OutcomePiperConfig, values)
    if cfg.execution_mode != "read_only":
        raise ValueError(
            "timing measurement requires explicit execution_mode=read_only; motion is not rewritten"
        )
    if cfg.scene is None or not cfg.cameras:
        raise ValueError("measurement needs the current scene and configured camera streams")
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    measure = modes.add_parser("measure")
    measure.add_argument("--config", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--seconds", type=float, default=60.0)
    measure.add_argument("--fps", type=float, default=20.0)
    report = modes.add_parser("report")
    report.add_argument("--events", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from lerobot_robot_outcome_piper.raw_io import read_jsonl
    from lerobot_robot_outcome_piper.timing_report import summarize_events

    if args.mode == "report":
        events = read_jsonl(args.events)
        with args.output.open("x") as out:
            json.dump(summarize_events(events), out, indent=2, allow_nan=False)
        return 0
    import math

    if not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.fps)):
        parser.error("seconds and fps must be positive finite values")
    cfg = load_read_only_config(args.config)
    from lerobot_robot_outcome_piper.robot import OutcomePiper

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(asdict(cfg), default=str, indent=2))
    robot = OutcomePiper(cfg)
    events = []
    result = dict(scope="read-only timing; no enable or motion", status="started")
    started = time.monotonic()
    try:
        robot.connect()
        result["connect_and_warmup_s"] = time.monotonic() - started
        started = time.monotonic()
        with (args.output / "events.jsonl").open("x") as log:
            while time.monotonic() - started < args.seconds:
                tick = time.monotonic()
                robot.get_observation()
                row = dict(
                    event="measurement",
                    phase="startup" if not events else "steady",
                    observation=robot.last_observation_telemetry,
                    action=None,
                )
                events.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - tick)))
        result["status"] = "measured"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            result["timing"] = summarize_events(events)
            (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(args.output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
