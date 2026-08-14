from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = ROOT / "infra" / "container"
LIB = CONTAINER / "lib"
sys.path.insert(0, str(LIB))

from piper_container_common import (  # noqa: E402
    PiperContainerError,
    atomic_write_json,
    file_inventory,
    inventory_identity,
    load_json,
    load_runtime_config,
    validate_repo_id,
    validate_revision,
    validate_run_id,
)


def test_compose_has_required_isolation() -> None:
    compose = (CONTAINER / "compose.yaml").read_text(encoding="utf-8")
    assert 'shm_size: "16gb"' in compose
    assert "driver: nvidia" in compose
    assert "count: 3" in compose
    assert "/var/run/docker.sock" not in compose
    assert "privileged:" not in compose
    assert "ipc:" not in compose
    assert "init:" not in compose
    assert "${PIPER_WORKSPACE_HOST" in compose
    assert "${PIPER_SSH_HOST_KEYS_HOST" in compose
    assert "${PIPER_AUTH_KEYS_HOST" in compose
    assert "${PIPER_IMAGE:" in compose
    assert "/opt/piper/container-runtime/entrypoint.sh" in compose
    assert compose.count("create_host_path: false") == 8


def test_dockerfile_pins_the_training_stack() -> None:
    dockerfile = (CONTAINER / "Dockerfile").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:"
        "ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85" in dockerfile
    )
    assert "ghcr.io/astral-sh/uv:0.11.32@sha256:" in dockerfile
    assert "COPY packages/lerobot_robot_outcome_piper/pyproject.toml" in dockerfile
    assert "ARG PYTHON_VERSION=3.12.13" in dockerfile
    assert '"tensorboard==2.21.0 ' in pyproject
    assert "m.version('tensorboard') == '2.21.0'" in dockerfile
    assert "--frozen" in dockerfile
    assert "--no-install-project" in dockerfile
    assert "piper-init-shared-python" in dockerfile
    assert "chmod -R a-w" not in dockerfile
    assert "PYTHONNOUSERSITE" not in dockerfile
    assert "UV_LOCK_SHA256" not in dockerfile
    assert 'ENTRYPOINT ["/usr/bin/tini"' in dockerfile


def test_gpu_acceptance_writes_real_offline_logs() -> None:
    runner = (CONTAINER / "acceptance" / "run_gpu_acceptance.sh").read_text(encoding="utf-8")
    smoke = (CONTAINER / "acceptance" / "logging_smoke.py").read_text(encoding="utf-8")
    assert 'python "${SCRIPT_DIR}/logging_smoke.py"' in runner
    assert "events.out.tfevents.*" in smoke
    assert 'os.environ.get("WANDB_MODE") != "offline"' in smoke
    assert 'wandb_dir.rglob("run-*.wandb")' in smoke


def test_sshd_disables_root_and_password_login() -> None:
    config = (CONTAINER / "sshd_config").read_text(encoding="utf-8")
    assert "PermitRootLogin no" in config
    assert "PasswordAuthentication no" in config
    assert "KbdInteractiveAuthentication no" in config
    assert "AuthenticationMethods publickey" in config


def test_ssh_shells_load_the_mutable_shared_environment() -> None:
    entrypoint = (CONTAINER / "entrypoint.sh").read_text(encoding="utf-8")
    profile = (CONTAINER / "profile.sh").read_text(encoding="utf-8")
    initializer = (CONTAINER / "init-shared-python.sh").read_text(encoding="utf-8")
    assert "install_shell_startup" in entrypoint
    assert "enable_public_key_account" in entrypoint
    assert "| chpasswd" in entrypoint
    assert '"u:${PIPER_ADMIN_USER}:rwx,u:${PIPER_COLLAB_USER}:r-x,m::rwx"' in entrypoint
    assert "source /workspace/piper/profile.sh" in entrypoint
    assert 'export PATH="/workspace/piper/bin:${PIPER_SHARED_PYTHON_ENV}/bin:${PATH}"' in profile
    assert "unset PYTHONNOUSERSITE" in profile
    assert 'SHARED_ENV="${PIPER_ROOT}/python-env"' in initializer
    assert 'rsync -a "${SEED_ENV}/" "${temporary}/"' in initializer
    assert '"${temporary}/bin/python" -m ensurepip --upgrade' in initializer
    assert 'chown -R "${PIPER_ADMIN_USER}:${PIPER_GROUP_NAME}"' in initializer
    assert "g-w" in initializer


def test_collaborator_supplementary_groups_are_reset() -> None:
    entrypoint = (CONTAINER / "entrypoint.sh").read_text(encoding="utf-8")
    assert 'usermod --groups "${PIPER_GROUP_NAME}" "${PIPER_COLLAB_USER}"' in entrypoint
    assert 'usermod --groups "${PIPER_GROUP_NAME},sudo" "${PIPER_ADMIN_USER}"' in entrypoint


def test_deployment_wrapper_uses_an_ordinary_image_reference() -> None:
    wrapper = (CONTAINER / "piper-compose").read_text(encoding="utf-8")
    assert 'source "${ENV_FILE}"' in wrapper
    assert "docker compose --env-file" in wrapper
    assert 'restart "$@"' in wrapper
    assert "recreate)" in wrapper
    assert "up -d --force-recreate" in wrapper
    assert '--user "${PIPER_ADMIN_USER}"' in wrapper
    assert "must name an existing host directory" in wrapper
    assert "project GID must differ from both private user IDs" in wrapper
    assert "missing regular non-empty public-key file" in wrapper


def test_release_area_is_root_owned_and_user_read_only() -> None:
    entrypoint = (CONTAINER / "entrypoint.sh").read_text(encoding="utf-8")
    promote = (CONTAINER / "bin" / "piper-artifact-promote").read_text(encoding="utf-8")
    doctor = (CONTAINER / "bin" / "piper-env-doctor").read_text(encoding="utf-8")
    assert 'install -d -m 2750 -o root -g "${PIPER_GROUP_NAME}"' in entrypoint
    assert "set_release_ownership(temporary, 0, group.gr_gid)" in promote
    assert '(PIPER_ROOT / "releases" / "datasets", False)' in doctor


def test_python_admin_command_and_gpu_runs_record_live_packages() -> None:
    python_admin = (CONTAINER / "bin" / "piper-python").read_text(encoding="utf-8")
    gpu_run = (CONTAINER / "bin" / "piper-gpu-run").read_text(encoding="utf-8")
    assert "only the project administrator" in python_admin
    assert '"install", "uninstall", "list", "snapshot"' in python_admin
    assert '"resolved_packages"' in python_admin
    assert '"after_freeze_sha256"' in python_admin
    assert '"python-packages.txt"' in gpu_run
    assert '"packages_sha256"' in gpu_run
    assert '"executable": os.path.abspath(sys.executable)' in gpu_run


def test_pr_container_build_has_no_registry_write_credentials() -> None:
    workflow = (ROOT / ".github" / "workflows" / "remote-training-container.yml").read_text(
        encoding="utf-8"
    )
    container_job = workflow.split("\n  container:", maxsplit=1)[1].split(
        "\n  publish:", maxsplit=1
    )[0]
    publish_job = workflow.split("\n  publish:", maxsplit=1)[1]
    assert "packages: write" not in container_job
    assert "id-token: write" not in container_job
    assert "docker/build-push-action" not in container_job
    assert "infra/container/ci/reclaim_runner_space.sh" in container_job
    assert "docker compose -f infra/container/compose.yaml config --quiet" in container_job
    assert "docker build --file infra/container/Dockerfile" in container_job
    assert "packages: write" in publish_job
    assert "push: true" in publish_job
    assert "github.event_name == 'workflow_dispatch'" in publish_job
    assert "push:" not in workflow.split("\npermissions:", maxsplit=1)[0]


def test_image_publication_requires_local_controller_integration() -> None:
    workflow = (ROOT / ".github" / "workflows" / "remote-training-container.yml").read_text(
        encoding="utf-8"
    )
    publish_job = workflow.split("\n  publish:", maxsplit=1)[1]
    assert "needs: [validate, local-controller-integration, container]" in publish_job


def test_retired_name_scan_checks_paths_and_contents() -> None:
    workflow = (ROOT / ".github" / "workflows" / "remote-training-container.yml").read_text(
        encoding="utf-8"
    )
    local_job = workflow.split("\n  local-controller-integration:", maxsplit=1)[1].split(
        "\n  container:", maxsplit=1
    )[0]
    assert "relative = path.as_posix()" in local_job
    assert "retired term in path" in local_job
    assert "if term in text" in local_job
    assert '"30da8e687a6dfc617fcd94afc367ac7071c376ce"' in local_job


def test_local_controller_ci_verifies_retained_license_texts() -> None:
    workflow = (ROOT / ".github" / "workflows" / "remote-training-container.yml").read_text(
        encoding="utf-8"
    )
    local_job = workflow.split("\n  local-controller-integration:", maxsplit=1)[1].split(
        "\n  container:", maxsplit=1
    )[0]
    for relative in (
        "licenses/pyAgxArm-LGPL-3.0-only.txt",
        "licenses/GPL-3.0-only.txt",
        "licenses/agx_arm_urdf-MIT.txt",
        "licenses/agx_arm_ros-MIT.txt",
    ):
        assert f"test -s {relative}" in local_job
    assert "f6642ce0d7872c686f29c99e9e10cd23d1d49313" in local_job
    assert "22a9cf6c5ad2fd2e0743531936bc5dab007fa5bc" in local_job


def test_workflow_actions_are_commit_pinned() -> None:
    workflow = (ROOT / ".github" / "workflows" / "remote-training-container.yml").read_text(
        encoding="utf-8"
    )
    action_lines = [line.strip() for line in workflow.splitlines() if "uses:" in line]
    assert action_lines
    for line in action_lines:
        assert re.search(r"@[0-9a-f]{40}(?:\s|$)", line), line


@pytest.mark.parametrize(
    ("function", "valid", "invalid"),
    [
        (validate_revision, "a" * 40, "main"),
        (validate_repo_id, "owner/model-name", "../model"),
        (validate_run_id, "EXP-TASK-A001", "bad run/id"),
    ],
)
def test_identifier_validation(function, valid: str, invalid: str) -> None:
    assert function(valid) == valid
    with pytest.raises(PiperContainerError):
        function(invalid)


def test_runtime_config_has_no_image_identity_gate(tmp_path: Path) -> None:
    runtime_config = tmp_path / "runtime.json"
    payload = {
        "schema_version": 2,
        "workspace_root": "/workspace",
        "admin_user": "admin",
        "collaborator_user": "collaborator",
        "group_name": "piper",
    }
    runtime_config.write_text(json.dumps(payload), encoding="utf-8")
    assert load_runtime_config(runtime_config) == payload

    del payload["group_name"]
    runtime_config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PiperContainerError, match="missing keys"):
        load_runtime_config(runtime_config)


def test_artifact_inventory_and_atomic_json(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"weights")
    (artifact / "config.json").write_text('{"model":"test"}\n', encoding="utf-8")

    files = file_inventory(artifact)
    assert [record["path"] for record in files] == ["config.json", "weights.bin"]
    assert len(inventory_identity(files)) == 64

    output = tmp_path / "payload.json"
    atomic_write_json(output, {"files": files})
    assert load_json(output) == {"files": files}


def test_artifact_inventory_only_skips_the_root_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    nested = artifact / "nested"
    nested.mkdir(parents=True)
    (artifact / "piper-artifact-manifest.json").write_text("{}\n", encoding="utf-8")
    (nested / "piper-artifact-manifest.json").write_text("payload\n", encoding="utf-8")

    files = file_inventory(artifact)

    assert [record["path"] for record in files] == ["nested/piper-artifact-manifest.json"]


def test_artifact_inventory_rejects_manifest_named_symlinks(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    target = tmp_path / "outside"
    target.write_text("outside\n", encoding="utf-8")
    link = artifact / "piper-artifact-manifest.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic-link creation is unavailable on this platform")

    with pytest.raises(PiperContainerError, match="symbolic links are forbidden"):
        file_inventory(artifact)


def test_env_example_contains_no_real_identity() -> None:
    env_example = (CONTAINER / "env.example").read_text(encoding="utf-8")
    assert "CHANGE_ME" in env_example
    assert "csm" not in env_example.lower()
    assert "password" not in env_example.lower()


def test_current_tree_does_not_carry_retired_acceptance_evidence() -> None:
    assert list((ROOT / "evidence").rglob("*.json")) == []


def test_json_module_is_available_for_ci_smoke() -> None:
    assert json.loads('{"ok": true}') == {"ok": True}
