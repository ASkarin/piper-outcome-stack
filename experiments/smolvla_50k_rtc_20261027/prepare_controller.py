"""Prepare a new isolated RTC bundle and test software only; never run motion."""

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent


def save(value):
    p = BASE / "preparation-status.json"
    t = p.with_suffix(".tmp")
    t.write_text(
        json.dumps(
            dict(updated_at=datetime.now(timezone.utc).isoformat(), hardware_io=False, **value),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    t.replace(p)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    require(not (BASE / "preparation-status.json").exists(), "this preparation already has state")
    spec = json.loads((BASE / "spec.json").read_text())
    require(str(BASE) == spec["controller_root"], "candidate root differs")
    profile = json.loads(
        (Path.home() / ".config/piper-outcome-stack/operator/profile.json").read_text()
    )
    old = json.loads((Path(spec["controller_base"]) / "operator-config.json").read_text())
    source = Path(profile["source"])
    save(dict(status="running", phase="source_and_configuration"))
    shutil.copytree(
        source / "packages/lerobot_robot_outcome_piper/src/lerobot_robot_outcome_piper",
        BASE / "source/lerobot_robot_outcome_piper",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ["execution_constraints.py", "gripper_reference.py"]:
        shutil.copy2(
            BASE / "plugin-overrides" / name, BASE / "source/lerobot_robot_outcome_piper" / name
        )
    # The pure policy and policy-loop sources were copied into this new bundle from Mac.
    shutil.copy2(BASE / "rtc_executor.py", BASE / "source/piper_policy/smolvla_execution.py")
    shutil.copy2(Path(spec["controller_base"]) / "label-trial.py", BASE / "label-trial.py")
    original = json.loads(Path(profile["record_b_config"]).read_text())
    safety_source = Path(original["robot"]["safety_path"])
    safety = BASE / "safety.json"
    shutil.copy2(safety_source, safety)
    (BASE / "configs").mkdir()
    policies = {}
    for skill, instruction in spec["tasks"].items():
        cfg = json.loads(json.dumps({k: original[k] for k in ["robot", "teleop"]}))
        cfg["robot"]["safety_path"] = str(safety)
        path = BASE / "configs" / (skill + ".json")
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
        policies[skill] = dict(instruction=instruction, config=str(path))
        restored = json.loads(json.dumps(cfg))
        restored["robot"]["safety_path"] = original["robot"]["safety_path"]
        require(
            restored == {k: original[k] for k in ["robot", "teleop"]}, "physical parameters changed"
        )
    require(safety.read_bytes() == safety_source.read_bytes(), "safety copy differs")
    require(
        Path(profile["python"]).is_file() and Path(old["vlm_assets"]).is_dir(),
        "existing runtime/assets unavailable",
    )
    operator = dict(
        python=profile["python"],
        vlm_assets=old["vlm_assets"],
        policies=policies,
        models={
            arm: dict(checkpoint=str(BASE / "checkpoints" / arm), run_id=row["run_id"])
            for arm, row in spec["models"].items()
        },
        runtime=spec["runtime"],
    )
    (BASE / "operator-config.json").write_text(
        json.dumps(operator, ensure_ascii=False, indent=2) + "\n"
    )
    provenance = dict(
        source=str(source),
        python=profile["python"],
        git_head=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        safety_source=str(safety_source),
        physical_parameters_preserved=True,
        model_sources=spec["models"],
        runtime=spec["runtime"],
        tested_plugin_overrides=["execution_constraints.py", "gripper_reference.py"],
    )
    (BASE / "source-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    with (BASE / "controller-source.diff").open("xb") as f:
        subprocess.run(["git", "-C", str(source), "diff", "HEAD"], stdout=f, check=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(BASE / "source"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    save(dict(status="running", phase="affected_offline_tests"))
    with (BASE / "offline-tests.log").open("xb") as log:
        subprocess.run(
            [profile["python"], "-m", "pytest", "-q", str(BASE / "tests")],
            cwd=BASE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    save(dict(status="running", phase="gripper_integration_tests"))
    with (BASE / "gripper-integration-tests.log").open("xb") as log:
        subprocess.run(
            [profile["python"], "-m", "pytest", "-q", str(BASE / "integration-tests")],
            cwd=BASE,
            env={**env, "PYTHONPATH": str(BASE / "source") + ":" + str(BASE / "test-support")},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    save(dict(status="running", phase="waiting_for_parallel_model_transfer"))
    while True:
        transfer = json.loads((BASE / "transfer-status.json").read_text())
        if transfer["status"] == "failed":
            raise RuntimeError(transfer["error"])
        if transfer["status"] == "complete":
            break
        time.sleep(5)
    save(dict(status="running", phase="reference_inputs_transfer"))
    (BASE / "reference-inputs").mkdir()
    subprocess.run(
        [
            "rsync",
            "-az",
            "piper-training:" + spec["training_stage"] + "/rtc-reference-20261027/",
            str(BASE / "reference-inputs") + "/",
        ],
        check=True,
    )
    finish_checks(operator, env)


def finish_checks(operator, env):
    save(dict(status="running", phase="device_free_model_checks"))
    checks = []
    for arm in ["original", "flat"]:
        # All four task instructions get reference/load checks through their actual entry.
        for skill in operator["policies"]:
            before = set((BASE / "checks").glob("*")) if (BASE / "checks").exists() else set()
            subprocess.run(
                [operator["python"], str(BASE / "run-policy.py"), "check", arm, skill, "1"],
                cwd=BASE,
                env=env,
                check=True,
                timeout=300,
            )
            created = set((BASE / "checks").glob("*")) - before
            require(len(created) == 1, "check result identity differs")
            trial = next(iter(created)) / "trial.json"
            d = json.loads(trial.read_text())
            require(
                d["status"] == "software_check_completed" and d["hardware_io"] is False,
                "device-free check did not close",
            )
            require(
                d["execution"]["rtc"]
                and d["execution"]["rtc_guidance"] == "native_joint6"
                and d["execution"]["lowpass"] is False,
                "actual runtime differs from RTC-only preparation",
            )
            chunks = d["execution_preflight"]["asynchronous_chunks"]
            prefix_count = sum(
                bool(row.get("rtc_request", {}).get("prefix_model_actions")) for row in chunks
            )
            checks.append(
                dict(
                    model=arm,
                    skill=skill,
                    trial=str(trial),
                    cross_gpu_reference_differences=sum(
                        r["reference_status"] == "different" for r in d["reference_errors"]
                    ),
                    completed_predictions=d["execution_preflight"]["inferences"],
                    rtc_prefix_publications=prefix_count,
                    recorded_preflight_rejections=sum(
                        r["status"] == "rejected"
                        for r in d["execution_preflight"]["recorded_target_checks"]
                    ),
                    cycle_work_seconds_p50_p95_max=d["execution_preflight"][
                        "cycle_work_seconds_p50_p95_max"
                    ],
                )
            )
            save(dict(status="running", phase="device_free_model_checks", checks=checks))
    save(dict(status="running", phase="actual_rtc_padding_and_timing", checks=checks))
    with (BASE / "rtc-padding-check.log").open("xb") as log:
        subprocess.run(
            [operator["python"], str(BASE / "validate_rtc.py")],
            cwd=BASE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )
    save(
        dict(
            status="complete",
            phase="prepared_for_operator_trials",
            checks=checks,
            rtc_validation=json.loads((BASE / "rtc-validation.json").read_text()),
            training_updates=0,
            hardware_motion_executed=False,
        )
    )


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["resume-checks"]:
            operator = json.loads((BASE / "operator-config.json").read_text())
            env = {
                **os.environ,
                "PYTHONPATH": str(BASE / "source"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            finish_checks(operator, env)
        elif sys.argv[1:]:
            raise ValueError("only the existing preparation or resume-checks is supported")
        else:
            main()
    except BaseException as error:
        old = (
            json.loads((BASE / "preparation-status.json").read_text())
            if (BASE / "preparation-status.json").exists()
            else {}
        )
        save(
            dict(
                status="failed",
                phase=old.get("phase"),
                checks=old.get("checks", []),
                error=repr(error),
            )
        )
        raise
