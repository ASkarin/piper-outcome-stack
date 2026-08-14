from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
LOCAL = ROOT / "infra" / "local-controller"


def test_local_controller_dependency_set_is_public_pinned_and_device_scoped():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    local_extra = "\n".join(pyproject["project"]["optional-dependencies"]["local-controller"])
    assert "lerobot_robot_outcome_piper" in local_extra
    assert "30da8e687a6dfc617fcd94afc367ac7071c376ce" in local_extra
    assert "799b8412fbe8b9156bc9892d3dbeb2df7e98be71" in local_extra
    assert "git+https://" in local_extra
    assert "git+ssh" not in local_extra
    assert "lerobot[core-scripts,gamepad,intelrealsense,smolvla]" in local_extra
    assert "torch==2.11.0+cu128" in local_extra
    assert "torchvision==0.26.0+cu128" in local_extra


def test_no_resident_control_service_or_runtime_account():
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in LOCAL.iterdir() if path.is_file()
    )
    assert "robot control serve-mock" not in source
    assert "piper-runtime" not in source
    isolate_unit = (LOCAL / "piper-socketcan-isolate@.service").read_text(encoding="utf-8")
    assert "Type=oneshot" in isolate_unit
    assert [line for line in isolate_unit.splitlines() if line.startswith("ExecStart=")] == [
        "ExecStart=/usr/local/sbin/piper-socketcan isolate %I"
    ]
    assert "RemainAfterExit" not in isolate_unit


def test_bootstrap_creates_project_and_collaborator_groups_without_placeholder_account():
    bootstrap = (LOCAL / "bootstrap-host.sh").read_text(encoding="utf-8")
    manager = (LOCAL / "manage-collaborator.sh").read_text(encoding="utf-8")
    assert "project_group=piper" in bootstrap
    assert "collab_group=piper-collab" in bootstrap
    assert 'groupadd --force "${project_group}"' in bootstrap
    assert 'groupadd --force "${collab_group}"' in bootstrap
    assert (
        'usermod --append --groups "${project_group},${collab_group}" "${administrator}"'
        in bootstrap
    )
    assert "useradd" not in bootstrap
    assert "adduser" not in bootstrap
    assert "PIPER_TAILNET_GRANT_CONFIRMED" in manager
    assert "PIPER_TAILNET_GRANT_REVOKED" in manager
    assert "adopt)" in manager
    assert 'fail "adopt requires one existing account"' in manager
    assert 'fail "existing authorized_keys is unavailable"' in manager
    assert 'fail "refusing to adopt an account with a privileged group"' in manager
    assert 'usermod --shell /bin/bash --groups "${collab_group}" "${account}"' in manager
    assert "adopted_at_utc" in manager
    assert "--operator" not in manager
    assert 'usermod --shell /bin/bash --groups "${collab_group}"' in manager
    assert 'usermod --groups "" --lock' in manager
    assert "authorized_keys" in manager


def test_bootstrap_normalizes_managed_python_permissions_after_install():
    bootstrap = (LOCAL / "bootstrap-host.sh").read_text(encoding="utf-8")
    install_python = "UV_PYTHON_INSTALL_DIR=/opt/piper/python uv python install 3.12.13"
    normalize_permissions = "chmod -R u=rwX,go=rX /opt/piper/python"
    assert bootstrap.count(install_python) == 1
    assert bootstrap.count(normalize_permissions) == 1
    assert bootstrap.index(normalize_permissions) > bootstrap.index(install_python)


def test_deployment_is_commit_scoped_accepted_and_immutable():
    deploy = (LOCAL / "deploy-release.sh").read_text(encoding="utf-8")
    assert 'git -c "safe.directory=${source_root}" -C "${source_root}" "$@"' in deploy
    assert "git config --global" not in deploy
    assert "git_source diff --quiet" in deploy
    assert "git_source diff --cached --quiet" in deploy
    assert "status --porcelain --untracked-files=all" in deploy
    assert 'archive --format=tar "${commit}"' in deploy
    assert "rsync" not in deploy
    assert 'locked_uv sync --project "${destination}"' in deploy
    assert "--all-packages" in deploy
    assert "--extra local-controller --no-dev --no-editable" in deploy
    assert ".piper-release-complete" in deploy
    assert ".piper-release-accepted" in deploy
    assert ".piper-release-acceptance.json" in deploy
    assert "release has not passed acceptance" in deploy
    assert "PIPER_PYPI_MIRROR must be an HTTPS package index" in deploy
    assert "--no-emit-project --no-emit-workspace" in deploy
    assert "--require-hashes" in deploy
    assert 'UV_PROJECT_ENVIRONMENT="${acceptance_environment}"' in deploy
    assert '"${acceptance_python}" -m pytest --quiet' in deploy
    assert '"${runtime_python}" -m pytest' not in deploy
    assert "\"${runtime_python}\" - <<'PIPER_RUNTIME_PLUGIN_DISCOVERY'" in deploy
    assert 'metadata.distribution("lerobot_robot_outcome_piper")' in deploy
    assert 'RobotConfig._choice_registry.get("outcome_piper")' in deploy
    assert 'TeleoperatorConfig._choice_registry.get("outcome_piper_xbox")' in deploy
    assert '"runtime_plugin_discovery": {' in deploy
    assert "runtime plugin discovery did not pass" in deploy
    assert 'locked_uv pip freeze --python "${runtime_python}"' in deploy
    assert '"packages": runtime_packages' in deploy
    assert "release package list is missing or invalid" in deploy
    assert "infra/acceptance/lerobot_dataset_replay_smoke.py" in deploy
    dataset_acceptance = deploy.split("local dataset_replay_root", maxsplit=1)[1].split(
        "local dataset_replay_report", maxsplit=1
    )[0]
    assert 'PATH="${release}/.venv/bin:/usr/bin:/bin"' in dataset_acceptance
    assert '"${runtime_python}"' in dataset_acceptance
    assert '"${acceptance_python}"' not in dataset_acceptance
    assert '"dataset_round_trip": dataset_replay' in deploy
    assert "Dataset round-trip smoke did not pass" in deploy
    assert '"kind": "ephemeral_locked_dev"' in deploy
    assert "doctor = json.loads(Path(doctor_report).read_text" in deploy
    assert 'doctor.get("source", {}).get("commit") != commit' in deploy
    assert '"lerobot": ("0.6.0", "30da8e687a6dfc617fcd94afc367ac7071c376ce")' in deploy
    assert '"pyagxarm": ("1.0.0", "799b8412fbe8b9156bc9892d3dbeb2df7e98be71")' in deploy
    assert "runtime dependency identity mismatch" in deploy
    assert "runtime dependency commit mismatch" in deploy
    assert 'summary_hash=$(sha256sum "${summary_temporary}"' in deploy
    assert 'actual_summary_hash=$(sha256sum "${release}/${acceptance_summary}"' in deploy
    assert "release acceptance summary hash does not match marker" in deploy
    assert "release lock no longer matches acceptance summary" in deploy
    assert "release already exists; releases are immutable" in deploy
    assert "chown -R root:" in deploy
    assert "mv -Tf" in deploy
    assert "GIT_SSH_COMMAND" not in deploy
    assert "IdentityFile" not in deploy


def test_host_doctor_requires_exact_planned_runtime_versions():
    doctor = (LOCAL / "piper-local-doctor.sh").read_text(encoding="utf-8")
    assert '"${driver_version}" == 595.*' in doctor
    assert '"${uv_version}" == "0.11.32"' in doctor
    assert '"${python_version}" == "3.12.13"' in doctor
    assert "ufw status verbose" in doctor
    assert "Default: deny (incoming), allow (outgoing)" in doctor
    assert "on tailscale0 to any port [0-9]+ proto tcp" in doctor
    assert '"piper-local-environment-v3"' in doctor
    assert "raw_hardware_authorized" in doctor
    assert "enumerated_device_access" in doctor
    assert "PIPER_CAN_INTERFACE" in doctor
    assert "can_namespace=piper-can" in doctor
    assert "target_absent_from_host=true" in doctor
    assert "target_present_in_namespace=true" in doctor
    assert '[[ "${socketcan_status}" == pass ]] || status=incomplete' in doctor
    assert "project_group_present" in doctor
    assert "collaborator_group_present" in doctor
    assert "administrator_project_group_member" in doctor
    assert "project_group_member: $administrator_project_group_member" in doctor
    assert 'glob("can*")' not in doctor


def test_socketcan_uses_one_explicit_namespace_and_drops_to_the_administrator():
    helper = (LOCAL / "piper-socketcan.sh").read_text(encoding="utf-8")
    bootstrap = (LOCAL / "bootstrap-host.sh").read_text(encoding="utf-8")
    assert "namespace=piper-can" in helper
    assert "isolate <interface> | up <interface> <bitrate>" in helper
    assert "exec -- <command> [args...] | down <interface>" in helper
    assert 'require_interface_down host "${interface}"' in helper
    assert 'require_interface_down namespace "${interface}"' in helper
    assert 'ip -n "${namespace}" link set dev "${interface}" down' in helper
    assert 'ip link set dev "${interface}" netns "${namespace}"' in helper
    assert 'type can bitrate "${bitrate}"' in helper
    assert 'setpriv --reuid="${uid}" --regid="${gid}" --init-groups' in helper
    assert "--inh-caps=-all --ambient-caps=-all --bounding-set=-all" in helper
    assert 'ip netns pids "${namespace}"' in helper
    assert "iproute2" in bootstrap
    assert "util-linux" in bootstrap
    assert "/usr/local/sbin/piper-socketcan" in bootstrap
    assert "/etc/systemd/system/piper-socketcan-isolate@.service" in bootstrap


def test_host_security_is_public_key_only_and_has_timed_rollback():
    sshd = (LOCAL / "sshd-hardening.conf").read_text(encoding="utf-8")
    bootstrap = (LOCAL / "bootstrap-host.sh").read_text(encoding="utf-8")
    doctor = (LOCAL / "piper-local-doctor.sh").read_text(encoding="utf-8")
    assert "export LC_ALL=C\nexport LANGUAGE=C" in bootstrap
    assert "export LC_ALL=C\nexport LANGUAGE=C" in doctor
    assert "PermitRootLogin no" in sshd
    assert "PasswordAuthentication no" in sshd
    assert "KbdInteractiveAuthentication no" in sshd
    assert "AuthenticationMethods publickey" in sshd
    assert "current_ssh_uses_tailscale" in bootstrap
    assert "--on-active=10m" in bootstrap
    assert "confirm-security" in bootstrap
    assert "ufw allow in on tailscale0" in bootstrap
    assert "ufw --force enable" in bootstrap
    assert "ufw --force reset" in bootstrap
    assert "AllowTcpForwarding no" in sshd
    assert "AllowStreamLocalForwarding no" in sshd
    local_readme = (LOCAL / "README.md").read_text(encoding="utf-8")
    assert local_readme.count("--preserve-env=SSH_CONNECTION") == 2


def test_local_controller_templates_contain_no_literal_identity_or_network_address():
    text = "\n".join(path.read_text(encoding="utf-8") for path in LOCAL.iterdir() if path.is_file())
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    assert "PRIVATE KEY" not in text
    assert "ssh-rsa " not in text
    assert "github.com ssh-ed25519 " not in text


def test_required_upstream_license_texts_are_retained():
    notices = (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8")
    expected = {
        "licenses/pyAgxArm-LGPL-3.0-only.txt": "GNU LESSER GENERAL PUBLIC LICENSE",
        "licenses/GPL-3.0-only.txt": "GNU GENERAL PUBLIC LICENSE",
        "licenses/agx_arm_urdf-MIT.txt": "Copyright (c) 2026 aalicecc",
        "licenses/agx_arm_ros-MIT.txt": "Copyright (c) 2026 aalicecc",
    }
    for relative, required_text in expected.items():
        assert relative in notices
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert required_text in text
        assert len(text) > 1_000
