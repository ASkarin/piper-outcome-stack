from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_robot_doctor_reports_local_safety_and_permission_boundary(tmp_path: Path):
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["A3_LOCAL_DEPLOYMENT_ROOT"] = str(tmp_path / "missing-deployment")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "a3_outcome_stack.ops",
            "robot",
            "doctor",
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["hardware_available"] is False
    assert report["hardware_tests_executed"] is False
    assert report["motor_enable_executed"] is False
    assert report["real_can_traffic_executed"] is False
    assert report["hardware_verified"] is False
    assert report["deployment"] == {
        "exists": False,
        "writable_by_current_process": False,
    }
    assert report["roles"]["administrator"]["unique_highest_privilege"] is True
    assert report["roles"]["administrator"]["raw_hardware_authorized"] is True
    assert report["roles"]["collaborator"]["raw_hardware_authorized"] is False
    assert report["roles"]["collaborator"]["sudo_authorized"] is False
    assert report["execution_role"] in {"administrator", "collaborator", "unassigned"}
    assert set(report["dependencies"]) >= {
        "lerobot",
        "lerobot_robot_a3",
        "el_a3_sdk",
    }
    for package in ("lerobot", "lerobot_robot_a3", "el_a3_sdk"):
        assert set(report["dependencies"][package]) == {
            "installed",
            "version",
            "vcs_commit",
        }
    for role in ("administrator", "collaborator"):
        access = report["roles"][role]["enumerated_device_access"]
        assert set(access) == {"can", "d435", "ar0234", "xbox"}
        assert all(item["status"] in {"pass", "fail", "not_checked"} for item in access.values())


def test_robot_doctor_has_no_project_root_argument() -> None:
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "a3_outcome_stack.ops", "robot", "doctor", "--root", "."],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "unrecognized arguments: --root" in completed.stderr
