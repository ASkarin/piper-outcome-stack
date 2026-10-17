"""A single bounded development run, followed by fixed offline and simulation checks."""

import argparse
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from .smolvla_training import write_json
from piper_policy.smolvla import TASK_A


def run(spec_path):
    spec = json.loads(spec_path.read_text())
    root = Path(spec["output_dir"]).parent
    status_path = root / "pipeline-status.json"
    if status_path.exists():
        raise FileExistsError(
            f"pipeline already has results/state: {root}; use a new run ID and output"
        )
    root.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "stages": [], "hardware_io": False}
    started = time.monotonic()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def stage(name, command):
        status["stage"] = name
        entry = {"name": name, "command": command, "status": "running"}
        status["stages"].append(entry)
        write_json(status_path, status)
        before = time.monotonic()
        with (root / f"{name}.log").open("x") as log:
            result = subprocess.run(
                command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
            )
        entry.update(
            exit_code=result.returncode,
            wall_seconds=time.monotonic() - before,
            status="complete" if result.returncode == 0 else "failed",
        )
        write_json(status_path, status)
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}")

    try:
        stage(
            "train",
            [
                sys.executable,
                "-m",
                "piper_outcome_stack.smolvla_training",
                "train",
                "--spec",
                str(spec_path),
            ],
        )
        candidates = [
            p / "pretrained_model"
            for p in (Path(spec["output_dir"]) / "checkpoints").iterdir()
            if p.is_dir() and p.name.isdigit() and int(p.name) == spec["budget"]["max_steps"]
        ]
        if len(candidates) != 1 or not candidates[0].is_dir():
            raise RuntimeError("completed final-step checkpoint is missing")
        checkpoint = candidates[0]
        status["checkpoint"] = str(checkpoint)
        for name, model_path, kind in [
            ("eval-initial", Path(spec["initial_checkpoint"]), "smolvla"),
            ("eval-finetuned", checkpoint, "smolvla"),
            ("eval-act40k", Path(spec["act_checkpoint"]), "act"),
        ]:
            stage(
                name,
                [
                    sys.executable,
                    "-m",
                    "piper_outcome_stack.smolvla_evaluate",
                    "--spec",
                    str(spec_path),
                    "--checkpoint",
                    str(model_path),
                    "--output",
                    str(root / name),
                    "--policy-type",
                    kind,
                ],
            )
        stage(
            "simulation",
            [
                spec["simulation_python"],
                "-m",
                "piper_outcome_stack.ops",
                "sim",
                "policy",
                "--policy-type",
                "smolvla",
                "--task",
                TASK_A,
                "--action-steps",
                "10",
                "--temporal-ensemble-coeff",
                "none",
                "--config",
                spec["simulation_config"],
                "--checkpoint",
                str(checkpoint),
                "--layout",
                "1",
                "--object",
                "green",
                "--region",
                "placement",
                "--duration",
                "40",
                "--seed",
                str(spec["optimization"]["seed"]),
                "--output",
                str(root / "simulation"),
            ],
        )
        simulation = json.loads((root / "simulation/summary.json").read_text())
        status.update(
            status="complete",
            simulated_task_success=simulation["success"],
            real_robot_tested=False,
            stop_reason="declared_pipeline_completed",
        )
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        status["wall_seconds"] = time.monotonic() - started
        write_json(status_path, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    run(args.spec)


if __name__ == "__main__":
    main()
