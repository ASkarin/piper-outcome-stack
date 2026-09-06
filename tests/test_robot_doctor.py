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
    environment["PIPER_LOCAL_DEPLOYMENT_ROOT"] = str(tmp_path / "missing-deployment")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "piper_outcome_stack.ops",
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
        "lerobot_robot_outcome_piper",
        "pyagxarm",
    }
    for package in ("lerobot", "lerobot_robot_outcome_piper", "pyagxarm"):
        assert set(report["dependencies"][package]) == {
            "installed",
            "version",
            "vcs_commit",
        }
    for role in ("administrator", "collaborator"):
        access = report["roles"][role]["enumerated_device_access"]
        assert set(access) == {"can", "d435", "xbox"}
        assert access["can"]["status"] in {
            "bind_succeeded",
            "bind_failed",
            "not_checked",
        }
        assert all(
            access[name]["status"] in {"pass", "fail", "not_checked"} for name in ("d435", "xbox")
        )


def test_robot_doctor_does_not_treat_visible_vcan_as_the_real_interface(
    monkeypatch,
) -> None:
    from piper_outcome_stack.ops import robot_doctor as doctor_module

    monkeypatch.delenv("PIPER_CAN_INTERFACE", raising=False)
    access, inventory = doctor_module._target_devices()
    assert access["can"]["status"] == "not_checked"
    assert inventory["can_interfaces"] == []


def test_doctor_binds_camera_inventory_to_one_serial_and_includes_usb(tmp_path, monkeypatch):
    from piper_outcome_stack.ops import robot_doctor as doctor_module

    def host_path(path):
        return tmp_path / str(path).lstrip("/")

    videos = host_path("/dev/v4l/by-id")
    videos.mkdir(parents=True)
    for serial in ("123456789012", "987654321098"):
        (videos / f"usb-Intel_RealSense_D435_{serial}-video-index0").touch()
    for name, serial, device in (("1-1", "123456789012", 2), ("1-2", "987654321098", 3)):
        usb = host_path(f"/sys/bus/usb/devices/{name}")
        usb.mkdir(parents=True)
        (usb / "serial").write_text(serial)
        (usb / "busnum").write_text("1")
        (usb / "devnum").write_text(str(device))
        node = host_path(f"/dev/bus/usb/001/{device:03d}")
        node.parent.mkdir(parents=True, exist_ok=True)
        node.touch()
    monkeypatch.setattr(doctor_module, "Path", host_path)
    monkeypatch.delenv("PIPER_CAN_INTERFACE", raising=False)
    monkeypatch.setenv("PIPER_D435_SERIAL", "123456789012")
    access, inventory = doctor_module._target_devices()
    assert inventory["d435_nodes"] == [
        str(videos / "usb-Intel_RealSense_D435_123456789012-video-index0"),
        str(host_path("/dev/bus/usb/001/002")),
    ]
    assert access["d435"]["status"] == "pass"
    monkeypatch.delenv("PIPER_D435_SERIAL")
    access, inventory = doctor_module._target_devices()
    assert inventory["d435_nodes"] == []
    assert access["d435"]["status"] == "not_checked"


def test_robot_doctor_has_no_project_root_argument() -> None:
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "piper_outcome_stack.ops", "robot", "doctor", "--root", "."],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "unrecognized arguments: --root" in completed.stderr
