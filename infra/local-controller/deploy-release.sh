#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

deployment_root=/opt/piper-outcome-stack
release_root=${deployment_root}/releases
collab_group=piper-collab
release_marker=.piper-release-complete
acceptance_marker=.piper-release-accepted
acceptance_summary=.piper-release-acceptance.json
incomplete_release_destination=
incomplete_release_temporary=
incomplete_release_owned=false
acceptance_temporary=

fail() {
    echo "error: $*" >&2
    exit 1
}

cleanup() {
    local status=$?
    if [[ "${status}" -ne 0 && "${incomplete_release_owned}" == true && \
        "${incomplete_release_destination}" =~ ^${release_root}/[0-9a-f]{40}$ && \
        "${incomplete_release_temporary}" =~ ^${release_root}/\.[0-9a-f]{40}\.installing$ && \
        -d "${incomplete_release_destination}" && \
        ! -L "${incomplete_release_destination}" && \
        ! -e "${incomplete_release_destination}/${release_marker}" && \
        ! -e "${incomplete_release_temporary}" && \
        ! -L "${incomplete_release_temporary}" ]]; then
        mv -- "${incomplete_release_destination}" "${incomplete_release_temporary}" || true
    fi
    if [[ -n "${acceptance_temporary}" && \
        "${acceptance_temporary}" =~ ^${deployment_root}/\.acceptance\.[0-9a-f]{40}\.[[:alnum:]]+$ && \
        -d "${acceptance_temporary}" && ! -L "${acceptance_temporary}" ]]; then
        rm -rf -- "${acceptance_temporary}"
    fi
    return "${status}"
}
trap cleanup EXIT

[[ "${EUID}" -eq 0 ]] || fail "run as root"
action=${1:-}

git_source() {
    GIT_OPTIONAL_LOCKS=0 \
        git -c "safe.directory=${source_root}" -C "${source_root}" "$@"
}

locked_uv() {
    UV_PYTHON_INSTALL_DIR=/opt/piper/python command uv "$@"
}

preinstall_registry_from_mirror() {
    local project=$1
    local mirror=${PIPER_PYPI_MIRROR:-}
    [[ -n "${mirror}" ]] || return 0
    [[ "${mirror}" =~ ^https://[^[:space:]]+$ ]] || \
        fail "PIPER_PYPI_MIRROR must be an HTTPS package index"
    local requirements=${project}/.piper-mirror-requirements.txt
    locked_uv export --project "${project}" --all-packages --frozen \
        --extra local-controller --no-dev \
        --no-emit-project --no-emit-workspace --no-emit-package lerobot \
        --no-emit-package pyagxarm \
        --no-emit-package torch --no-emit-package torchvision \
        --format requirements-txt --output-file "${requirements}"
    locked_uv venv --python 3.12.13 "${project}/.venv"
    UV_DEFAULT_INDEX="${mirror}" UV_CONCURRENT_DOWNLOADS=8 \
        UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 \
        uv pip sync --python "${project}/.venv/bin/python" \
        --require-hashes "${requirements}"
    rm -f -- "${requirements}"
}

release_path() {
    local commit=$1
    [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || fail "release must be a full Git commit"
    printf '%s/%s\n' "${release_root}" "${commit}"
}

verify_release() {
    local commit=$1
    local release
    release=$(release_path "${commit}")
    [[ -d "${release}" && ! -L "${release}" ]] || fail "release does not exist"
    [[ -f "${release}/${release_marker}" && ! -L "${release}/${release_marker}" ]] || \
        fail "release is incomplete"
    [[ "$(<"${release}/${release_marker}")" == "${commit}" ]] || \
        fail "release completion marker does not match commit"
    printf '%s\n' "${release}"
}

run_acceptance() {
    local commit=$1
    local release
    release=$(verify_release "${commit}")
    local runtime_python=${release}/.venv/bin/python
    [[ -x "${runtime_python}" ]] || fail "release Python is unavailable"
    local administrator=${SUDO_USER:-}
    [[ -n "${administrator}" && "${administrator}" != root ]] || \
        fail "run through sudo from the administrator account"
    local administrator_home
    administrator_home=$(getent passwd "${administrator}" | cut -d: -f6)
    [[ -n "${administrator_home}" && "${administrator_home}" != / ]] || \
        fail "administrator home is unavailable"
    local summary=${release}/${acceptance_summary}
    local marker=${release}/${acceptance_marker}
    [[ ! -e "${summary}" && ! -L "${summary}" && \
        ! -e "${marker}" && ! -L "${marker}" ]] || \
        fail "release acceptance already exists"

    acceptance_temporary=$(mktemp -d \
        "${deployment_root}/.acceptance.${commit}.XXXXXXXX")
    local acceptance_environment=${acceptance_temporary}/venv
    UV_PYTHON_INSTALL_DIR=/opt/piper/python \
        UV_PROJECT_ENVIRONMENT="${acceptance_environment}" \
        UV_LINK_MODE=copy \
        command uv sync --project "${release}" --all-packages --frozen \
        --extra local-controller --group dev --no-editable
    chown -R "${administrator}:${collab_group}" "${acceptance_temporary}"
    chmod -R u=rwX,g=rX,o= "${acceptance_temporary}"
    local acceptance_python=${acceptance_environment}/bin/python
    [[ -x "${acceptance_python}" ]] || fail "acceptance Python is unavailable"

    local doctor_report=${acceptance_temporary}/runtime-doctor.json
    if ! sudo -u "${administrator}" \
        env -i HOME="${administrator_home}" \
        PATH="${release}/.venv/bin:/usr/bin:/bin" \
        PYTHONDONTWRITEBYTECODE=1 \
        "${runtime_python}" -m piper_outcome_stack.ops doctor --root "${release}" \
        >"${doctor_report}"; then
        fail "release runtime doctor failed"
    fi
    chown "${administrator}:${collab_group}" "${doctor_report}"
    chmod 0640 "${doctor_report}"
    sudo -u "${administrator}" env -i \
        HOME="${administrator_home}" \
        PATH="${release}/.venv/bin:/usr/bin:/bin" \
        PYTHONDONTWRITEBYTECODE=1 \
        "${runtime_python}" - <<'PIPER_RUNTIME_PLUGIN_DISCOVERY'
import importlib.metadata as metadata

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()

from lerobot_robot_outcome_piper.config import OutcomePiperConfig, OutcomePiperXboxConfig

plugin = metadata.distribution("lerobot_robot_outcome_piper")
if plugin.version != "0.1.0" or list(plugin.entry_points):
    raise SystemExit("runtime plugin distribution identity is invalid")
if RobotConfig._choice_registry.get("outcome_piper") is not OutcomePiperConfig:
    raise SystemExit("runtime robot plugin was not auto-discovered")
if (
    TeleoperatorConfig._choice_registry.get("outcome_piper_xbox")
    is not OutcomePiperXboxConfig
):
    raise SystemExit("runtime teleoperator plugin was not auto-discovered")
PIPER_RUNTIME_PLUGIN_DISCOVERY
    sudo -u "${administrator}" env -i \
        HOME="${administrator_home}" \
        PATH="${acceptance_environment}/bin:/usr/bin:/bin" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTEST_ADDOPTS="-p no:cacheprovider" \
        "${acceptance_python}" -m pytest --quiet \
        "${release}/tests/test_plugin.py" \
        "${release}/tests/test_local_controller_policy.py"

    local dataset_replay_root=${acceptance_temporary}/dataset-replay
    sudo -u "${administrator}" env -i \
        HOME="${administrator_home}" \
        PATH="${release}/.venv/bin:/usr/bin:/bin" \
        PYTHONDONTWRITEBYTECODE=1 \
        "${runtime_python}" \
        "${release}/infra/acceptance/lerobot_dataset_replay_smoke.py" \
        --work-dir "${dataset_replay_root}" >/dev/null
    local dataset_replay_report=${dataset_replay_root}/summary.json
    [[ -f "${dataset_replay_report}" ]] || fail "Dataset round-trip report is unavailable"

    local runtime_packages=${acceptance_temporary}/runtime-packages.txt
    locked_uv pip freeze --python "${runtime_python}" | LC_ALL=C sort >"${runtime_packages}"
    [[ -s "${runtime_packages}" ]] || fail "release package list is empty"

    local lock_hash doctor_hash runtime_version uv_version
    lock_hash=$(sha256sum "${release}/uv.lock" | awk '{print $1}')
    doctor_hash=$(sha256sum "${doctor_report}" | awk '{print $1}')
    runtime_version=$("${runtime_python}" -c \
        'import platform; print(platform.python_version())')
    uv_version=$(locked_uv --version)
    local summary_temporary=${release}/${acceptance_summary}.tmp
    local marker_temporary=${release}/${acceptance_marker}.tmp
    [[ ! -e "${summary_temporary}" && ! -L "${summary_temporary}" && \
        ! -e "${marker_temporary}" && ! -L "${marker_temporary}" ]] || \
        fail "stale release acceptance temporary file exists"
    "${runtime_python}" - "${summary_temporary}" "${commit}" "${lock_hash}" \
        "${doctor_report}" "${doctor_hash}" "${runtime_version}" "${uv_version}" \
        "${runtime_packages}" "${dataset_replay_report}" \
        <<'PIPER_ACCEPTANCE_SUMMARY'
import json
import sys
from pathlib import Path

(
    path,
    commit,
    lock_hash,
    doctor_report,
    doctor_hash,
    runtime_version,
    uv_version,
    runtime_packages_path,
    dataset_replay_report,
) = sys.argv[1:]
doctor = json.loads(Path(doctor_report).read_text(encoding="utf-8"))
if doctor.get("status") != "ok" or doctor.get("source", {}).get("commit") != commit:
    raise SystemExit("runtime doctor report does not match the accepted release")
installed = doctor.get("dependencies", {}).get("installed", {})
expected_dependencies = {
    "lerobot": ("0.6.0", "30da8e687a6dfc617fcd94afc367ac7071c376ce"),
    "pyagxarm": ("1.0.0", "799b8412fbe8b9156bc9892d3dbeb2df7e98be71"),
    "lerobot_robot_outcome_piper": ("0.1.0", None),
}
for name, (version, vcs_commit) in expected_dependencies.items():
    actual = installed.get(name, {})
    if actual.get("installed") is not True or actual.get("version") != version:
        raise SystemExit(f"runtime dependency identity mismatch for {name}")
    if vcs_commit is not None and actual.get("vcs_commit") != vcs_commit:
        raise SystemExit(f"runtime dependency commit mismatch for {name}")
runtime_packages = Path(runtime_packages_path).read_text(encoding="utf-8").splitlines()
if not runtime_packages or runtime_packages != sorted(runtime_packages):
    raise SystemExit("runtime package list is empty or unsorted")
dataset_replay = json.loads(Path(dataset_replay_report).read_text(encoding="utf-8"))
if (
    dataset_replay.get("schema_version") != "piper-lerobot-dataset-replay-smoke-v1"
    or dataset_replay.get("status") != "passed"
):
    raise SystemExit("Dataset round-trip smoke did not pass")
summary = {
    "schema_version": "piper-release-acceptance-v1",
    "commit": commit,
    "lock": {"path": "uv.lock", "sha256": lock_hash},
    "runtime": {
        "python_version": runtime_version,
        "packages": runtime_packages,
    },
    "test_environment": {
        "kind": "ephemeral_locked_dev",
        "uv_version": uv_version,
        "sync": [
            "--all-packages",
            "--frozen",
            "--extra=local-controller",
            "--group=dev",
            "--no-editable",
        ],
    },
    "checks": {
        "runtime_doctor": {
            "status": "passed",
            "report_sha256": doctor_hash,
            "report": doctor,
        },
        "runtime_plugin_discovery": {
            "status": "passed",
            "distribution": "lerobot_robot_outcome_piper",
            "version": "0.1.0",
            "entry_points": [],
            "robot_type": "outcome_piper",
            "teleoperator_type": "outcome_piper_xbox",
        },
        "pytest": {
            "status": "passed",
            "tests": [
                "tests/test_plugin.py",
                "tests/test_local_controller_policy.py",
            ],
        },
        "dataset_round_trip": dataset_replay,
    },
}
Path(path).write_text(
    json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
PIPER_ACCEPTANCE_SUMMARY
    local summary_hash
    summary_hash=$(sha256sum "${summary_temporary}" | awk '{print $1}')
    printf '%s\n%s\n' "${commit}" "${summary_hash}" >"${marker_temporary}"
    chown root:"${collab_group}" "${summary_temporary}" "${marker_temporary}"
    chmod 0440 "${summary_temporary}" "${marker_temporary}"
    mv -- "${summary_temporary}" "${summary}"
    mv -- "${marker_temporary}" "${marker}"
}

activate_release() {
    local commit=${1:-}
    local release
    release=$(verify_release "${commit}")
    [[ -f "${release}/${acceptance_marker}" && ! -L "${release}/${acceptance_marker}" ]] || \
        fail "release has not passed acceptance"
    [[ -f "${release}/${acceptance_summary}" && ! -L "${release}/${acceptance_summary}" ]] || \
        fail "release acceptance summary is unavailable"
    local accepted
    mapfile -t accepted <"${release}/${acceptance_marker}"
    [[ "${#accepted[@]}" -eq 2 && "${accepted[0]}" == "${commit}" && \
        "${accepted[1]}" =~ ^[0-9a-f]{64}$ ]] || \
        fail "release acceptance marker is invalid"
    local actual_summary_hash
    actual_summary_hash=$(sha256sum "${release}/${acceptance_summary}" | awk '{print $1}')
    [[ "${actual_summary_hash}" == "${accepted[1]}" ]] || \
        fail "release acceptance summary hash does not match marker"
    local summary_values
    if ! summary_values=$("${release}/.venv/bin/python" - \
        "${release}/${acceptance_summary}" <<'PIPER_VERIFY_ACCEPTANCE'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("schema_version") != "piper-release-acceptance-v1":
    raise SystemExit("unexpected acceptance summary schema")
checks = summary.get("checks", {})
if checks.get("runtime_doctor", {}).get("status") != "passed":
    raise SystemExit("runtime doctor did not pass")
if checks.get("runtime_plugin_discovery", {}).get("status") != "passed":
    raise SystemExit("runtime plugin discovery did not pass")
if checks.get("pytest", {}).get("status") != "passed":
    raise SystemExit("acceptance tests did not pass")
if (
    checks.get("dataset_round_trip", {}).get("schema_version")
    != "piper-lerobot-dataset-replay-smoke-v1"
    or checks.get("dataset_round_trip", {}).get("status") != "passed"
):
    raise SystemExit("Dataset round-trip smoke did not pass")
packages = summary.get("runtime", {}).get("packages")
if (
    not isinstance(packages, list)
    or not packages
    or packages != sorted(packages)
    or any(not isinstance(package, str) or not package for package in packages)
):
    raise SystemExit("release package list is missing or invalid")
print(summary.get("commit", ""))
print(summary.get("lock", {}).get("sha256", ""))
PIPER_VERIFY_ACCEPTANCE
    ); then
        fail "release acceptance summary is invalid"
    fi
    local summary_fields
    mapfile -t summary_fields <<<"${summary_values}"
    [[ "${#summary_fields[@]}" -eq 2 && "${summary_fields[0]}" == "${commit}" ]] || \
        fail "release acceptance summary does not match commit"
    local actual_lock_hash
    actual_lock_hash=$(sha256sum "${release}/uv.lock" | awk '{print $1}')
    [[ "${summary_fields[1]}" =~ ^[0-9a-f]{64}$ && \
        "${summary_fields[1]}" == "${actual_lock_hash}" ]] || \
        fail "release lock no longer matches acceptance summary"
    local temporary_link=${deployment_root}/.current.${commit}
    [[ ! -e "${temporary_link}" && ! -L "${temporary_link}" ]] || \
        fail "temporary activation link already exists"
    ln -s "releases/${commit}" "${temporary_link}"
    mv -Tf -- "${temporary_link}" "${deployment_root}/current"
}

case "${action}" in
    install)
        source_root=${2:-}
        [[ -d "${source_root}/.git" ]] || fail "source must be a Git checkout"
        git_source diff --quiet || fail "source worktree is dirty"
        git_source diff --cached --quiet || fail "source index is dirty"
        [[ -z "$(git_source status --porcelain --untracked-files=all)" ]] || \
            fail "source checkout contains untracked files"
        commit=$(git_source rev-parse HEAD)
        [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || fail "source commit is invalid"
        destination=${release_root}/${commit}
        [[ ! -e "${destination}" && ! -L "${destination}" ]] || \
            fail "release already exists; releases are immutable"
        temporary=${release_root}/.${commit}.installing
        [[ ! -e "${temporary}" ]] || fail "stale release installation exists"
        install -d -m 0750 -o root -g "${collab_group}" "${temporary}"
        git_source archive --format=tar "${commit}" | tar -xf - -C "${temporary}"
        mv -- "${temporary}" "${destination}"
        incomplete_release_destination=${destination}
        incomplete_release_temporary=${temporary}
        incomplete_release_owned=true
        preinstall_registry_from_mirror "${destination}"
        locked_uv sync --project "${destination}" --all-packages --frozen \
            --extra local-controller --no-dev --no-editable
        printf '%s\n' "${commit}" >"${destination}/${release_marker}"
        chown -R root:"${collab_group}" "${destination}"
        chmod -R u=rwX,g=rX,o= "${destination}"
        incomplete_release_owned=false
        ;;
    accept)
        run_acceptance "${2:-}"
        ;;
    activate)
        activate_release "${2:-}"
        ;;
    *)
        fail "usage: $0 install <clean-source-checkout> | accept <full-commit> | activate <full-commit>"
        ;;
esac
