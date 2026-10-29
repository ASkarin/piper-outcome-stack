"""Wait for the prepared data, verify it, then launch the authorized paired runs."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
REPO = Path("/workspace/projects/piper-outcome-stack")
PYTHON = str(REPO / ".venv/bin/python")
RUNS = Path("/workspace/piper/runs/csm")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def state(stage, **values):
    save(
        BASE / "pipeline-status.json",
        dict(
            stage=stage,
            updated_at=datetime.now(timezone.utc).isoformat(),
            hardware_io=False,
            **values,
        ),
    )


def free_gpus():
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    return [int(line.split(",")[0]) for line in lines if int(line.split(",")[1]) < 500]


def verify_data():
    ready = json.loads((BASE / "dataset-ready.json").read_text())
    data = json.loads((BASE / "dataset-manifest.json").read_text())
    digest = hashlib.sha256((BASE / "dataset-manifest.json").read_bytes()).hexdigest()
    require(
        ready["status"] == "complete" and digest == ready["dataset_manifest_sha256"],
        "dataset completion record/manifest identity differs",
    )
    require(
        data["new_original_demonstrations"] == 108 and data["new_retained_frames"] == 113473,
        "new-data source counts differ",
    )
    files = 0
    for entry in data["sources"]:
        root = Path(entry["root"])
        require(root.parent == BASE / "datasets", f"unexpected transferred dataset root: {root}")
        for row in entry["transferred_files"]:
            path = root / row["path"]
            require(
                path.is_file() and path.stat().st_size == row["size"],
                f"incomplete transfer: {path}",
            )
            files += 1
        meta = json.loads((root / "meta/info.json").read_text())
        require(
            meta["fps"] == 50 and meta["total_frames"] == entry["frames"],
            f"dataset metadata: {root}",
        )
    save(
        BASE / "transfer-verification.json",
        dict(status="passed", files_checked=files, dataset_manifest_sha256=digest),
    )
    return data, digest


def make_specs(data, digest):
    original = json.loads(
        Path("/workspace/piper/staging/csm/smolvla-b-20261017/spec.json").read_text()
    )
    inputs = json.loads((BASE / "training-inputs.json").read_text())
    common = {
        k: original[k]
        for k in [
            "source_checkpoint",
            "source_revision",
            "vlm_assets",
            "vlm_revision",
            "optimization",
            "evaluation",
            "input_contract",
            "action_encoding",
        ]
    }
    require(
        common["source_revision"] == "d9f33c94a60fb382c90dea2164c96845bd955e28",
        "wrong original base revision",
    )
    require(
        Path(common["source_checkpoint"]).name == "source-model",
        "source is not original pretrained model",
    )
    common.update(
        train_sources=inputs["train_sources"]
        + [{k: e[k] for k in ["root", "repo_id", "segments"]} for e in data["sources"]],
        development_sources=inputs["development_sources"],
        expected_train_frames=282138,
        expected_train_episodes=434,
        expected_development_episodes=72,
        data_version=data["data_version"],
        data_manifest_sha256=digest,
        temporal_weight=2.5221420342525227,
        padding=dict(
            real_dimensions=7,
            padded_dimensions=32,
            state_tail="zero",
            target_action_tail="zero",
            training_loss_dimensions=7,
            inference_tail="x_t[...,7:]=t*epsilon[...,7:]; use project SmolVLAChunkPredictor",
        ),
        authorization="2026-10-27 user: original and flat FM, both fresh from original pretrained base, each 50000 steps; structural padding handled as before",
        preregistration_id="PR-20261027-01",
        sampling_policy="equal task-layout cell mass, then equal source demo, then uniform complete50-frame starts; same deterministic seed+epoch sampler in both arms",
    )
    common["optimization"] = dict(
        common["optimization"],
        seed=42,
        batch_size=32,
        num_workers=4,
        learning_rate=1e-4,
        warmup_steps=1000,
    )
    common["budget"] = dict(
        max_steps=50000,
        max_wallclock_seconds=43200,
        save_interval_steps=5000,
        eval_interval_steps=5000,
    )
    specs = {}
    for witness in (True, False):
        for arm in ("original", "flat"):
            spec = json.loads(json.dumps(common))
            run_id = f"smolvla-b-{arm}-fresh-20261027-seed42" + ("-witness" if witness else "")
            spec.update(
                arm=arm,
                witness=witness,
                run_id=run_id,
                run_root=str(RUNS / run_id),
                output_dir=str(RUNS / run_id / "checkpoints-output"),
            )
            if witness:
                spec["budget"] = dict(
                    max_steps=4,
                    max_wallclock_seconds=3600,
                    save_interval_steps=2,
                    eval_interval_steps=2,
                )
            path = BASE / (run_id + ".json")
            require(not path.exists(), f"spec already exists: {path}")
            save(path, spec)
            specs[(witness, arm)] = (spec, path)
    return specs


def launch(spec, path, gpu):
    command = [
        "/workspace/piper/bin/piper-gpu-run",
        "--gpus",
        str(gpu),
        "--run-id",
        spec["run_id"],
        "--repo",
        str(REPO),
        "--python",
        PYTHON,
        "--config",
        str(path),
        "--",
        PYTHON,
        str(BASE / "train.py"),
        "--spec",
        str(path),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(BASE / "source"),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with (BASE / (spec["run_id"] + ".log")).open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    info = dict(run_id=spec["run_id"], pid=process.pid, gpu=gpu, command=command, spec=str(path))
    save(BASE / (spec["run_id"] + "-launch.json"), info)
    return process, info


def main():
    require(not (BASE / "pipeline-status.json").exists(), "this pipeline already has state")
    state("waiting_for_prepared_data")
    deadline = time.monotonic() + 24 * 3600
    while not (BASE / "dataset-ready.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "data preparation did not finish within24 hours; no training started"
            )
        time.sleep(30)
    state("verifying_data")
    data, digest = verify_data()
    specs = make_specs(data, digest)
    while len(free_gpus()) < 2:
        state("waiting_for_two_free_gpus")
        time.sleep(300)
    gpus = free_gpus()[:2]
    for witness in (True, False):
        active = [
            launch(*specs[(witness, arm)], gpu)
            for arm, gpu in zip(("original", "flat"), gpus, strict=True)
        ]
        state(
            "witness_running" if witness else "training_running",
            launches=[info for _, info in active],
            target_steps_per_model=4 if witness else 50000,
        )
        while any(process.poll() is None for process, _ in active):
            for process, info in active:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"run failed; inspect primary log: {info['run_id']}")
            time.sleep(30)
        for process, info in active:
            require(process.returncode == 0, f"run failed; inspect primary log: {info['run_id']}")
            report = json.loads((RUNS / info["run_id"] / "training-status.json").read_text())
            require(report["status"] == "complete", f"run did not finish: {info['run_id']}")
        if witness:
            reports = [
                json.loads((RUNS / info["run_id"] / "training-status.json").read_text())
                for _, info in active
            ]
            require(
                reports[0]["first_batch_indices"] == reports[1]["first_batch_indices"],
                "paired witness used different first batch order",
            )
            views = [
                json.loads((RUNS / info["run_id"] / "dataset-view.json").read_text())
                for _, info in active
            ]
            require(views[0] == views[1], "paired witness used different data/normalization")
            state("witness_passed", runs=[info["run_id"] for _, info in active])
            while any(gpu not in free_gpus() for gpu in gpus):
                state("waiting_for_witness_gpu_release")
                time.sleep(30)
    state(
        "complete",
        runs=[specs[(False, arm)][0]["run_id"] for arm in ("original", "flat")],
        steps_per_model=50000,
        hardware_motion_executed=False,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        previous = (
            json.loads((BASE / "pipeline-status.json").read_text())
            if (BASE / "pipeline-status.json").exists()
            else {}
        )
        state(
            "failed",
            failed_stage=previous.get("stage"),
            launches=previous.get("launches", []),
            error=repr(error),
        )
        raise
