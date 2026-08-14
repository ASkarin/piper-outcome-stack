from __future__ import annotations

import json
import os
from pathlib import Path

import wandb
from torch.utils.tensorboard import SummaryWriter


def required_directory(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise RuntimeError(f"{name} is not a directory: {path}")
    return path


def main() -> int:
    if os.environ.get("WANDB_MODE") != "offline":
        raise RuntimeError("WANDB_MODE must be offline during formal runs")

    run_id = os.environ.get("PIPER_RUN_ID")
    if not run_id:
        raise RuntimeError("PIPER_RUN_ID is not set")

    run_dir = required_directory("PIPER_RUN_DIR")
    tensorboard_dir = required_directory("TENSORBOARD_LOG_DIR")
    wandb_dir = required_directory("WANDB_DIR")

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_scalar("acceptance/value", 1.0, 0)
    writer.flush()
    writer.close()

    event_files = sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("TensorBoard did not create an event file")

    run = wandb.init(
        project="piper-outcome-stack-acceptance",
        name=run_id,
        mode="offline",
        dir=str(wandb_dir),
        config={"schema_version": 1, "purpose": "environment acceptance"},
        settings=wandb.Settings(silent=True),
    )
    if run is None:
        raise RuntimeError("W&B did not create an offline run")
    try:
        run.log({"acceptance/value": 1.0}, step=0)
    finally:
        run.finish()

    wandb_files = sorted(wandb_dir.rglob("run-*.wandb"))
    if not wandb_files:
        raise RuntimeError("W&B did not create an offline run file")

    payload = {
        "schema_version": 1,
        "status": "pass",
        "tensorboard_files": [str(path.relative_to(run_dir)) for path in event_files],
        "wandb_files": [str(path.relative_to(run_dir)) for path in wandb_files],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
