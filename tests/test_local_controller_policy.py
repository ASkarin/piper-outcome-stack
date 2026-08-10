from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
LOCAL = ROOT / "infra" / "local-controller"


def test_local_controller_dependency_set_is_pinned_and_device_scoped():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    local_extra = "\n".join(pyproject["project"]["optional-dependencies"]["local-controller"])
    assert (
        "lerobot_robot_a3 @ git+ssh://git@github.com/ASkarin/"
        "lerobot-robot-edulite-a3.git@bf188864ef3922f8caded5cc19cc43b8061c4b22" in local_extra
    )
    assert "el_a3_sdk @" not in local_extra
    assert "lerobot[core-scripts,gamepad,intelrealsense,smolvla]" in local_extra
    assert "torch==2.11.0+cu128" in local_extra
    assert "torchvision==0.26.0+cu128" in local_extra
    assert "pyserial" not in local_extra


def test_socket_control_service_and_runtime_account_are_absent():
    assert not (LOCAL / "a3-local-control.service").exists()
    assert not (LOCAL / "local-control.env.example").exists()
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in LOCAL.iterdir() if path.is_file()
    )
    assert "robot control serve-mock" not in source
    assert "a3-runtime" not in source


def test_bootstrap_creates_only_the_collaborator_group_and_no_placeholder_account():
    bootstrap = (LOCAL / "bootstrap-host.sh").read_text(encoding="utf-8")
    manager = (LOCAL / "manage-collaborator.sh").read_text(encoding="utf-8")
    assert "groupadd --force" in bootstrap
    assert "a3-collab" in bootstrap
    assert "a3-operator" not in bootstrap
    assert "a3-hardware" not in bootstrap
    assert 'usermod --append --groups "${collab_group}" "${administrator}"' in bootstrap
    assert '"${state_root}/permits"' not in bootstrap
    assert "a3-local-control.service" not in bootstrap
    assert "useradd" not in bootstrap
    assert "adduser" not in bootstrap
    assert "A3_TAILNET_GRANT_CONFIRMED" in manager
    assert "A3_TAILNET_GRANT_REVOKED" in manager
    assert "--operator" not in manager
    assert 'usermod --shell /bin/bash --groups "${collab_group}"' in manager
    assert 'usermod --groups "" --lock' in manager
    assert "authorized_keys" in manager
    assert "account already exists" in manager
    assert "public-key file must contain exactly one key" in manager
    assert "account was not provisioned by this manager" in manager
    assert '"${account_uid}" -ge 1000' in manager


def test_deployment_uses_clean_commit_scoped_immutable_environments():
    deploy = (LOCAL / "deploy-release.sh").read_text(encoding="utf-8")
    assert 'git -c "safe.directory=${source_root}" -C "${source_root}" "$@"' in deploy
    assert "git config --global" not in deploy
    assert "git_source diff --quiet" in deploy
    assert "git_source diff --cached --quiet" in deploy
    assert "status --porcelain --untracked-files=all" in deploy
    assert 'archive --format=tar "${commit}"' in deploy
    assert "rsync" not in deploy
    assert "private_uv sync" in deploy
    assert deploy.count("--all-packages") == 2
    assert "--extra local-controller --no-dev" in deploy
    assert deploy.count("--no-editable") == 1
    assert 'preinstall_registry_from_mirror "${temporary}"' not in deploy
    assert 'private_uv sync --project "${temporary}"' not in deploy
    move_to_destination = deploy.index('mv -- "${temporary}" "${destination}"')
    preinstall_destination = deploy.index('preinstall_registry_from_mirror "${destination}"')
    sync_destination = deploy.index('private_uv sync --project "${destination}"')
    assert move_to_destination < preinstall_destination < sync_destination
    assert 'private_uv sync --project "${destination}"' in deploy
    assert ".a3-release-complete" in deploy
    assert "release is incomplete" in deploy
    assert "A3_PYPI_MIRROR must be an HTTPS package index" in deploy
    assert "uv export" in deploy
    assert "--require-hashes" in deploy
    assert "--no-emit-package torch" in deploy
    assert "--no-emit-package torchvision" in deploy
    assert "UV_DEFAULT_INDEX" in deploy
    assert "UV_HTTP_TIMEOUT=600" in deploy
    assert "administrator=${SUDO_USER:-}" in deploy
    assert '"${SUDO_UID:-}" == "$(id -u "${administrator}")"' in deploy
    assert "runuser --user" in deploy
    assert "/usr/bin/env -i" in deploy
    assert "HOME=%q USER=%q LOGNAME=%q" in deploy
    assert "GIT_LFS_SKIP_SMUDGE=1" in deploy
    assert "BatchMode=yes" in deploy
    assert "StrictHostKeyChecking=yes" in deploy
    assert "HostKeyAlgorithms=ssh-ed25519" in deploy
    assert "UserKnownHostsFile=${private_git_known_hosts}" in deploy
    assert "GlobalKnownHostsFile=/dev/null" in deploy
    assert "IdentityAgent=none" in deploy
    assert "IdentitiesOnly=yes" in deploy
    assert 'mktemp -d "${deployment_root}/.a3-git-transport.XXXXXX"' in deploy
    assert 'chmod 0711 "${private_git_runtime}"' in deploy
    assert "private_git_known_hosts=${private_git_runtime}/known_hosts" in deploy
    assert "private_git_ssh_wrapper=${private_git_runtime}/ssh" in deploy
    assert "mktemp /run" not in deploy
    assert 'rmdir -- "${private_git_runtime}"' in deploy
    assert "pinned GitHub host key fingerprint verification failed" in deploy
    assert "StrictHostKeyChecking=no" not in deploy
    assert "accept-new" not in deploy
    assert "SSH_AUTH_SOCK" not in deploy
    assert "ssh-add" not in deploy
    assert "IdentityFile" not in deploy
    assert "--no-emit-package lerobot-robot-a3" in deploy
    assert "release already exists; releases are immutable" in deploy
    assert "incomplete_release_owned=true" in deploy
    assert 'mv -- "${incomplete_release_destination}" "${incomplete_release_temporary}"' in deploy
    assert "chown -R root:" in deploy
    assert "mv -Tf" in deploy


def test_host_doctor_requires_exact_planned_runtime_versions():
    doctor = (LOCAL / "a3-local-doctor.sh").read_text(encoding="utf-8")
    assert '"${driver_version}" == 595.*' in doctor
    assert '"${uv_version}" == "0.11.32"' in doctor
    assert '"${python_version}" == "3.12.13"' in doctor
    assert "ufw status verbose" in doctor
    assert "Default: deny (incoming), allow (outgoing)" in doctor
    assert "on tailscale0 to any port [0-9]+ proto tcp" in doctor
    assert '"a3-local-environment-v2"' in doctor
    assert "raw_hardware_authorized" in doctor
    assert "enumerated_device_access" in doctor
    assert "raw_hardware_access_granted_to_humans" not in doctor


def test_host_security_is_public_key_only_and_has_timed_rollback():
    sshd = (LOCAL / "sshd-hardening.conf").read_text(encoding="utf-8")
    bootstrap = (LOCAL / "bootstrap-host.sh").read_text(encoding="utf-8")
    assert "PermitRootLogin no" in sshd
    assert "PasswordAuthentication no" in sshd
    assert "KbdInteractiveAuthentication no" in sshd
    assert "AuthenticationMethods publickey" in sshd
    assert "current_ssh_uses_tailscale" in bootstrap
    assert "local connection=${SSH_CONNECTION:-}" in bootstrap
    assert "preserve SSH_CONNECTION through sudo" in bootstrap
    assert "| grep -q" not in bootstrap
    assert "{print $2; exit}" not in bootstrap
    assert "{print $2; seen=1}" in bootstrap
    assert "--on-active=10m" in bootstrap
    assert "confirm-security" in bootstrap
    assert "ufw allow in on tailscale0" in bootstrap
    assert "ufw --force enable" in bootstrap
    assert "ufw --force reset" in bootstrap
    assert "assert_administrator_key" in bootstrap
    assert "assert_sshd_policy" in bootstrap
    assert "assert_ufw_policy" in bootstrap
    assert "ufw status verbose" in bootstrap
    assert "UFW default policy mismatch" in bootstrap
    assert "on tailscale0 to any port [0-9]+ proto tcp" in bootstrap
    assert "AllowTcpForwarding no" in sshd
    assert "AllowStreamLocalForwarding no" in sshd

    local_readme = (LOCAL / "README.md").read_text(encoding="utf-8")
    assert local_readme.count("--preserve-env=SSH_CONNECTION") == 2


def test_local_controller_templates_contain_no_literal_network_address():
    text = "\n".join(path.read_text(encoding="utf-8") for path in LOCAL.iterdir() if path.is_file())
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    assert "PRIVATE KEY" not in text
    assert "ssh-rsa " not in text
    assert text.count("github.com ssh-ed25519 ") == 1
    assert "AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" in text
    assert text.count("SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU") == 1
