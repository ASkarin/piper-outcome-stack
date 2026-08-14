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

SDK_COMMIT = "799b8412fbe8b9156bc9892d3dbeb2df7e98be71"
LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
PLUGIN_VERSION = "0.1.0"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")

REQUIRED_DIRECTORIES = (
    "src/piper_outcome_stack/ops",
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

    marker = root / ".piper-release-complete"
    try:
        commit = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValidationError(
            "project root is neither a Git checkout nor a completed Git-archive release"
        ) from exc
    if not COMMIT_PATTERN.fullmatch(commit):
        raise IntegrityError("invalid .piper-release-complete commit marker")
    return {"kind": "git_archive_release", "commit": commit, "branch": None}


def _load_lock(root: Path) -> dict[str, Any]:
    lock_path = root / "uv.lock"
    try:
        with lock_path.open("rb") as handle:
            lock = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValidationError(f"cannot read uv.lock: {exc}") from exc
    return lock


def _locked_git_package(lock: dict[str, Any], name: str, commit: str) -> dict[str, str]:
    matches = [package for package in lock.get("package", []) if package.get("name") == name]
    if len(matches) != 1:
        raise IntegrityError(f"uv.lock must contain exactly one {name} package")
    package = matches[0]
    source = package.get("source", {})
    git_source = source.get("git", "")
    if not isinstance(git_source, str) or f"#{commit}" not in git_source:
        raise IntegrityError(f"uv.lock does not resolve {name} to {commit}")
    return {"version": str(package.get("version", "")), "commit": commit}


def _locked_workspace_package(lock: dict[str, Any], name: str, version: str) -> dict[str, str]:
    matches = [package for package in lock.get("package", []) if package.get("name") == name]
    if len(matches) != 1:
        raise IntegrityError(f"uv.lock must contain exactly one {name} package")
    package = matches[0]
    if package.get("source") != {"editable": "packages/lerobot_robot_outcome_piper"}:
        raise IntegrityError(f"uv.lock does not resolve {name} to the workspace package")
    if package.get("version") != version:
        raise IntegrityError(f"uv.lock resolves {name} to an unexpected version")
    return {"version": version, "source": "workspace"}


def _installed_package(
    name: str,
    *,
    expected_version: str,
    expected_vcs_commit: str | None = None,
) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise IntegrityError(f"required distribution {name} is not installed") from exc
    if distribution.version != expected_version:
        raise IntegrityError(
            f"installed {name} version {distribution.version!r} does not match "
            f"locked version {expected_version!r}"
        )

    vcs_commit = None
    direct_url_value = None
    if expected_vcs_commit is not None:
        direct_url = distribution.read_text("direct_url.json")
        if not direct_url:
            raise IntegrityError(f"installed {name} has no direct_url.json source identity")
        try:
            direct_url_data = json.loads(direct_url)
        except json.JSONDecodeError as exc:
            raise IntegrityError(f"installed {name} has invalid direct_url.json") from exc
        direct_url_value = direct_url_data.get("url")
        vcs_info = direct_url_data.get("vcs_info")
        if not isinstance(vcs_info, dict):
            raise IntegrityError(f"installed {name} is not identified as a VCS dependency")
        vcs_commit = vcs_info.get("commit_id")
        if vcs_commit != expected_vcs_commit:
            raise IntegrityError(
                f"installed {name} commit {vcs_commit!r} does not match "
                f"locked commit {expected_vcs_commit}"
            )
    return {
        "installed": True,
        "version": distribution.version,
        "vcs_commit": vcs_commit,
        "direct_url": direct_url_value,
    }


def doctor_project(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    missing = [relative for relative in REQUIRED_DIRECTORIES if not (root_path / relative).is_dir()]
    if missing:
        raise ValidationError(f"missing project directories: {missing}")

    project = load_json(root_path / "configs/project.json")
    if project.get("project_id") != "piper-outcome-stack":
        raise IntegrityError(
            f"expected project_id piper-outcome-stack, got {project.get('project_id')}"
        )
    if project.get("display_name") != "PiPER OutcomeStack":
        raise IntegrityError(
            f"expected display_name PiPER OutcomeStack, got {project.get('display_name')}"
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
    lock = _load_lock(root_path)
    sdk_lock = _locked_git_package(lock, "pyagxarm", SDK_COMMIT)
    lerobot_lock = _locked_git_package(lock, "lerobot", LEROBOT_COMMIT)
    plugin_lock = _locked_workspace_package(lock, "lerobot-robot-outcome-piper", PLUGIN_VERSION)
    installed = {
        "pyagxarm": _installed_package(
            "pyAgxArm",
            expected_version=sdk_lock["version"],
            expected_vcs_commit=SDK_COMMIT,
        ),
        "lerobot": _installed_package(
            "lerobot",
            expected_version=lerobot_lock["version"],
            expected_vcs_commit=LEROBOT_COMMIT,
        ),
        "lerobot_robot_outcome_piper": _installed_package(
            "lerobot_robot_outcome_piper",
            expected_version=plugin_lock["version"],
        ),
    }
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
                "pyagxarm": sdk_lock,
                "lerobot": lerobot_lock,
                "lerobot_robot_outcome_piper": plugin_lock,
            },
            "installed": installed,
        },
        "hardware_verified": False,
    }
