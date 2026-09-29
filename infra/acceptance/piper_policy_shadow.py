"""Recorded software-path validation or operator-run read-only live shadow."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from piper_outcome_stack.policy_execution import ACTChunkPredictor, run_shadow


def metrics(values):
    values = np.asarray(values) * 1000
    return dict(
        samples=len(values),
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        max_ms=float(values.max()),
        over20ms=int(sum(values > 20)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("recorded", "live"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="explicit read_only robot JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=1500)
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("cycles must be positive")
    import draccus
    from lerobot.robots.config import RobotConfig
    from lerobot_robot_outcome_piper import OutcomePiper, OutcomePiperConfig
    from lerobot_robot_outcome_piper.safety import ACTION_KEYS, load_motion_safety
    from lerobot_robot_outcome_piper.execution_constraints import check_execution_target

    raw = json.loads(args.config.read_text())
    config = draccus.decode(RobotConfig, raw)
    if not isinstance(config, OutcomePiperConfig) or config.execution_mode != "read_only":
        raise ValueError("shadow configuration must explicitly select read_only PiPER")
    if config.capture_timing is None or config.safety_path is None:
        raise ValueError("shadow needs measured capture_timing and safety_path")
    safety = load_motion_safety(config.safety_path)
    # Reserve a new result before expensive work. Never replace earlier evidence.
    with args.output.open("x") as output:
        report = dict(
            status="started",
            mode=args.mode,
            robot_config=raw,
            safety=asdict(safety),
            checkpoint=str(args.checkpoint),
            reference_inputs=str(args.reference_inputs),
            device="cuda",
            fps=50,
            n_action_steps=1,
            rows=[],
            motion_commands_sent=False,
            scope="live acquisition through candidate validation, no action dispatch"
            if args.mode == "live"
            else "recorded inputs with synthetic receive times; no hardware I/O",
        )
        robot = None
        try:
            torch.set_num_threads(4)
            predictor = ACTChunkPredictor.from_checkpoint(args.checkpoint)
            with np.load(args.reference_inputs) as data:
                images, states = data["images"], data["states"]
                expected = data["expected_absolute_chunks"]
            errors = []
            for i in range(len(states)):
                difference = np.abs(predictor.predict(images[i], states[i]) - expected[i])
                joint, gripper = float(difference[:, :6].max()), float(difference[:, 6].max())
                if joint >= 1e-4 or gripper >= 1e-5:
                    raise ValueError(f"reference mismatch at sample {i}: {joint}, {gripper}")
                # Initialize FK/geometry and check real candidates before live input capture.
                check_execution_target(states[i], expected[i, 0], safety)
                errors.append(dict(sample=i, joint_rad=joint, gripper_m=gripper))
            report["reference_errors"] = errors
            for i in range(30):
                predictor.predict(images[i % len(images)], states[i % len(states)])

            if args.mode == "live":
                robot = OutcomePiper(config)
                robot.connect()
            else:

                class RecordedSource:
                    def __init__(self):
                        self.config = config
                        self.index = 0

                    def get_observation(self):
                        i = self.index % len(states)
                        self.index += 1
                        self.last_observation_telemetry = dict(
                            quality="checked",
                            sequence=self.index,
                            oldest_received_monotonic_s=time.monotonic(),
                            timestamp_source="synthetic_for_software_path_only",
                        )
                        return {**dict(zip(ACTION_KEYS, states[i])), "d435": images[i]}

                robot = RecordedSource()
            run_shadow(robot, predictor, safety, args.cycles, 50, report["rows"])
            report["status"] = "completed"
        except BaseException as exc:
            report.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            try:
                if args.mode == "live" and robot is not None:
                    robot.disconnect()
            except BaseException as exc:
                report.update(status="failed", disconnect_error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                if report["rows"]:
                    report["timing"] = {
                        name: metrics([r[name] for r in report["rows"]])
                        for name in (
                            "inference_s",
                            "validation_s",
                            "acquisition_s",
                            "cycle_work_s",
                            "cycle_elapsed_s",
                        )
                    }
                output.write(json.dumps(report, indent=2))
                output.flush()
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in report.items()
                            if k not in ("rows", "robot_config", "safety", "reference_errors")
                        },
                        indent=2,
                    )
                )


if __name__ == "__main__":
    main()
