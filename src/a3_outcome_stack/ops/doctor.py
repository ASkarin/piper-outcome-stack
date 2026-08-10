"""Read-only structural, release, and dependency diagnostics."""

from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from .canonical import load_json, sha256_text_file
from .errors import IntegrityError, ValidationError

SDK_COMMIT = "ea7231f784ebb37e4c4120f7be8e3670514dc9ee"
LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
ADAPTER_COMMIT = "bf188864ef3922f8caded5cc19cc43b8061c4b22"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")

REQUIRED_DIRECTORIES = (
    "src/a3_outcome_stack/ops",
    "docs/preregistration",
    "tests",
    "evidence",
)


def _source_identity(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        inside = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        top_level = Path(
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        ).resolve()
        if inside == "true" and top_level == root.resolve() and COMMIT_PATTERN.fullmatch(commit):
            branch = subprocess.check_output(
                ["git", "-C", str(root), "branch", "--show-current"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return {
                "kind": "git_checkout",
                "commit": commit,
                "branch": branch or None,
            }
    except (OSError, subprocess.CalledProcessError):
        pass

    marker = root / ".a3-release-complete"
    try:
        commit = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValidationError(
            "project root is neither a Git checkout nor a completed Git-archive release"
        ) from exc
    if not COMMIT_PATTERN.fullmatch(commit):
        raise IntegrityError("invalid .a3-release-complete commit marker")
    return {"kind": "git_archive_release", "commit": commit, "branch": None}


def _locked_package(root: Path, name: str, commit: str) -> dict[str, str]:
    lock_path = root / "uv.lock"
    try:
        with lock_path.open("rb") as handle:
            lock = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValidationError(f"cannot read uv.lock: {exc}") from exc
    matches = [package for package in lock.get("package", []) if package.get("name") == name]
    if len(matches) != 1:
        raise IntegrityError(f"uv.lock must contain exactly one {name} package")
    package = matches[0]
    source = package.get("source", {})
    git_source = source.get("git", "")
    if not isinstance(git_source, str) or f"#{commit}" not in git_source:
        raise IntegrityError(f"uv.lock does not resolve {name} to {commit}")
    return {"version": str(package.get("version", "")), "commit": commit}


def _installed_package(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None, "vcs_commit": None}
    vcs_commit = None
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            vcs_commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        except json.JSONDecodeError:
            vcs_commit = None
    return {
        "installed": True,
        "version": distribution.version,
        "vcs_commit": vcs_commit,
    }


def doctor_project(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    missing = [relative for relative in REQUIRED_DIRECTORIES if not (root_path / relative).is_dir()]
    if missing:
        raise ValidationError(f"missing project directories: {missing}")

    project = load_json(root_path / "configs/project.json")
    if project.get("project_id") != "a3-outcome-stack":
        raise IntegrityError(
            f"expected project_id a3-outcome-stack, got {project.get('project_id')}"
        )
    if project.get("display_name") != "A3 OutcomeStack":
        raise IntegrityError(
            f"expected display_name A3 OutcomeStack, got {project.get('display_name')}"
        )
    prereg = project.get("preregistration", {})
    snapshot = root_path / prereg.get("path", "")
    if not snapshot.is_file():
        raise ValidationError(f"missing preregistration snapshot: {snapshot}")
    actual_prereg_hash = sha256_text_file(snapshot)
    if actual_prereg_hash != prereg.get("sha256"):
        raise IntegrityError(
            f"preregistration snapshot mismatch: expected {prereg.get('sha256')}, "
            f"got {actual_prereg_hash}"
        )

    source = _source_identity(root_path)
    sdk_lock = _locked_package(root_path, "el-a3-sdk", SDK_COMMIT)
    lerobot_lock = _locked_package(root_path, "lerobot", LEROBOT_COMMIT)
    adapter_lock = _locked_package(root_path, "lerobot-robot-a3", ADAPTER_COMMIT)
    return {
        "status": "ok",
        "project_id": project["project_id"],
        "display_name": project["display_name"],
        "project_root": str(root_path),
        "source": source,
        "python": sys.version.split()[0],
        "preregistration": {
            "id": prereg.get("id"),
            "sha256": actual_prereg_hash,
        },
        "dependencies": {
            "uv_lock": {
                "el_a3_sdk": sdk_lock,
                "lerobot": lerobot_lock,
                "lerobot_robot_a3": adapter_lock,
            },
            "installed": {
                "el_a3_sdk": _installed_package("el-a3-sdk"),
                "lerobot": _installed_package("lerobot"),
                "lerobot_robot_a3": _installed_package("lerobot_robot_a3"),
            },
        },
        "hardware_verified": False,
    }
