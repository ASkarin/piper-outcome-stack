"""Read-only local robot-host diagnostics.

LeRobot and ``lerobot_robot_outcome_piper`` own actuator-facing behavior. This module only
reports installed dependencies, device enumeration, and the two human-role
permission boundary used by OutcomeStack operations.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import socket
from pathlib import Path
from typing import Any


def _ordinary_device_access(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        return {"status": "not_checked", "devices": []}
    devices = [
        {
            "path": str(path),
            "readable": os.access(path, os.R_OK),
            "writable": os.access(path, os.W_OK),
        }
        for path in paths
    ]
    return {
        "status": (
            "pass" if all(item["readable"] and item["writable"] for item in devices) else "fail"
        ),
        "devices": devices,
    }


def _can_access(interface: str | None) -> dict[str, Any]:
    if not interface:
        return {
            "status": "not_checked",
            "interface": None,
            "bind_succeeded": None,
            "error": None,
        }
    bind_succeeded = False
    error = None
    try:
        can_socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            can_socket.bind((interface,))
            bind_succeeded = True
        finally:
            can_socket.close()
    except (AttributeError, OSError) as exc:
        error = str(exc)
    return {
        "status": "bind_succeeded" if bind_succeeded else "bind_failed",
        "interface": interface,
        "bind_succeeded": bind_succeeded,
        "error": error,
    }


def _target_devices() -> tuple[dict[str, Any], dict[str, list[str]]]:
    target_can_interface = os.environ.get("PIPER_CAN_INTERFACE")
    network_root = Path("/sys/class/net")
    can_interfaces = []
    if target_can_interface and (network_root / target_can_interface).exists():
        can_interfaces.append(target_can_interface)
    by_id = Path("/dev/v4l/by-id")
    video_links = sorted(by_id.iterdir()) if by_id.is_dir() else []
    d435 = [
        path
        for path in video_links
        if "realsense" in path.name.lower() or "d435" in path.name.lower()
    ]
    ar0234 = [path for path in video_links if "ar0234" in path.name.lower()]
    configured_ar0234 = os.environ.get("PIPER_AR0234_DEVICE")
    if configured_ar0234:
        ar0234.append(Path(configured_ar0234))
    input_by_id = Path("/dev/input/by-id")
    xbox = sorted(input_by_id.glob("*-event-joystick")) if input_by_id.is_dir() else []
    access = {
        "can": _can_access(target_can_interface),
        "d435": _ordinary_device_access(d435),
        "ar0234": _ordinary_device_access(ar0234),
        "xbox": _ordinary_device_access(xbox),
    }
    inventory = {
        "can_interfaces": can_interfaces,
        "d435_nodes": [str(path) for path in d435],
        "ar0234_nodes": [str(path) for path in ar0234],
        "xbox_nodes": [str(path) for path in xbox],
    }
    return access, inventory


def _execution_role() -> str:
    groups: set[str] = set()
    if os.name == "posix":
        import grp

        for group_id in os.getgroups():
            try:
                groups.add(grp.getgrgid(group_id).gr_name)
            except KeyError:
                continue
    if (hasattr(os, "geteuid") and os.geteuid() == 0) or "sudo" in groups:
        return "administrator"
    if os.environ.get("PIPER_LOCAL_COLLAB_GROUP", "piper-collab") in groups:
        return "collaborator"
    return "unassigned"


def _not_checked_targets() -> dict[str, Any]:
    return {
        "can": {
            "status": "not_checked",
            "interface": None,
            "bind_succeeded": None,
            "error": None,
        },
        **{name: {"status": "not_checked", "devices": []} for name in ("d435", "ar0234", "xbox")},
    }


def _installed_distribution(distribution_name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None, "vcs_commit": None}
    vcs_commit = None
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            vcs_commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        except json.JSONDecodeError:
            pass
    return {
        "installed": True,
        "version": distribution.version,
        "vcs_commit": vcs_commit,
    }


def robot_doctor() -> dict[str, Any]:
    execution_role = _execution_role()
    current_access, inventory = _target_devices()
    administrator_access = (
        current_access if execution_role == "administrator" else _not_checked_targets()
    )
    collaborator_access = (
        current_access if execution_role == "collaborator" else _not_checked_targets()
    )

    deployment_root = Path(
        os.environ.get("PIPER_LOCAL_DEPLOYMENT_ROOT", "/opt/piper-outcome-stack")
    )
    return {
        "status": "ok",
        "dependencies": {
            "lerobot": _installed_distribution("lerobot"),
            "lerobot_robot_outcome_piper": _installed_distribution("lerobot_robot_outcome_piper"),
            "pyagxarm": _installed_distribution("pyAgxArm"),
        },
        "execution_role": execution_role,
        "roles": {
            "administrator": {
                "unique_highest_privilege": True,
                "raw_hardware_authorized": True,
                "current_process_is_role": execution_role == "administrator",
                "enumerated_device_access": administrator_access,
            },
            "collaborator": {
                "raw_hardware_authorized": False,
                "sudo_authorized": False,
                "current_process_is_role": execution_role == "collaborator",
                "enumerated_device_access": collaborator_access,
            },
        },
        "target_device_inventory": inventory,
        "deployment": {
            "exists": deployment_root.is_dir(),
            "writable_by_current_process": deployment_root.is_dir()
            and os.access(deployment_root, os.W_OK),
        },
        "hardware_available": False,
        "hardware_tests_executed": False,
        "motor_enable_executed": False,
        "real_can_traffic_executed": False,
        "hardware_verified": False,
        "hardware_branch": "not_executed",
    }
