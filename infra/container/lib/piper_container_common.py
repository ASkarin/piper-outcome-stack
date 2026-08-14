from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path("/workspace")
PIPER_ROOT = WORKSPACE_ROOT / "piper"
RUNTIME_CONFIG = Path("/etc/piper/runtime.json")
SHARED_PYTHON_ENV = PIPER_ROOT / "python-env"
PYTHON_HISTORY_ROOT = PIPER_ROOT / "python-env-history"
ARTIFACT_MANIFEST_NAME = "piper-artifact-manifest.json"
MIRROR_ENDPOINT = "https://hf-mirror.com"
MIN_FREE_BYTES = 200 * 1024**3
WARN_FREE_BYTES = 300 * 1024**3
MIN_SHM_BYTES = 16 * 1024**3

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PiperContainerError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any, mode: int = 0o640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode())
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PiperContainerError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PiperContainerError(f"expected a JSON object in {path}")
    return payload


def load_runtime_config(path: Path = RUNTIME_CONFIG) -> dict[str, Any]:
    payload = load_json(path)
    required = {
        "schema_version",
        "workspace_root",
        "admin_user",
        "collaborator_user",
        "group_name",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise PiperContainerError(f"runtime config is missing keys: {', '.join(missing)}")
    if payload["schema_version"] != 2:
        raise PiperContainerError("runtime config schema_version must be 2")
    for key in ("workspace_root", "admin_user", "collaborator_user", "group_name"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise PiperContainerError(f"runtime config {key} must be a non-empty string")
    expected_workspace = WORKSPACE_ROOT.as_posix()
    if payload["workspace_root"] != expected_workspace:
        raise PiperContainerError(
            f"runtime config workspace_root must be {expected_workspace}, "
            f"found {payload['workspace_root']}"
        )
    return payload


def validate_revision(value: str) -> str:
    if not _REVISION_RE.fullmatch(value):
        raise PiperContainerError("revision must be an exact 40-character lowercase commit SHA")
    return value


def validate_repo_id(value: str) -> str:
    components = value.split("/")
    if (
        not _REPO_ID_RE.fullmatch(value)
        or any(component.endswith((".", "-")) for component in components)
        or any(".." in component or "--" in component for component in components)
    ):
        raise PiperContainerError("repo must have the form owner/name using safe characters")
    return value


def validate_run_id(value: str) -> str:
    if not _RUN_ID_RE.fullmatch(value):
        raise PiperContainerError("run ID contains unsupported characters or is too long")
    return value


def ensure_under(path: Path, root: Path) -> Path:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if not resolved_path.is_relative_to(resolved_root):
        raise PiperContainerError(f"{resolved_path} is outside allowed root {resolved_root}")
    return resolved_path


def free_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def total_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return stats.f_blocks * stats.f_frsize


def file_inventory(root: Path) -> list[dict[str, Any]]:
    root = root.resolve(strict=True)
    root_manifest = root / ARTIFACT_MANIFEST_NAME
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PiperContainerError(f"symbolic links are forbidden in artifacts: {path}")
        if path == root_manifest:
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise PiperContainerError(f"unsupported artifact entry: {path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def inventory_identity(files: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(files))


def git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current")
        dirty_output = run("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PiperContainerError(f"{repo} is not an inspectable Git worktree") from exc
    return {"commit": commit, "branch": branch, "dirty": bool(dirty_output)}


def gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PiperContainerError("nvidia-smi GPU inventory failed") from exc

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=4)]
        if len(fields) != 5:
            raise PiperContainerError(f"unexpected nvidia-smi output: {line}")
        index, uuid, name, memory_mib, pci_bus_id = fields
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_mib": int(memory_mib),
                "pci_bus_id": pci_bus_id,
            }
        )
    return gpus


def manifest_hash(path: Path) -> str:
    return sha256_file(path.resolve(strict=True))
