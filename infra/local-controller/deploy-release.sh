#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

deployment_root=/opt/a3-outcome-stack
release_root=${deployment_root}/releases
collab_group=a3-collab
release_marker=.a3-release-complete
administrator=
administrator_home=
private_git_known_hosts=
private_git_ssh_wrapper=
private_git_runtime=
incomplete_release_destination=
incomplete_release_temporary=
incomplete_release_owned=false
github_host_key='github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl'
github_host_key_fingerprint='SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU'

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
    if [[ -n "${private_git_known_hosts}" ]]; then
        rm -f -- "${private_git_known_hosts}"
    fi
    if [[ -n "${private_git_ssh_wrapper}" ]]; then
        rm -f -- "${private_git_ssh_wrapper}"
    fi
    if [[ -n "${private_git_runtime}" && \
        "${private_git_runtime}" =~ ^${deployment_root}/\.a3-git-transport\.[[:alnum:]]+$ && \
        -d "${private_git_runtime}" && ! -L "${private_git_runtime}" ]]; then
        rmdir -- "${private_git_runtime}" || true
    fi
    return "${status}"
}
trap cleanup EXIT

[[ "${EUID}" -eq 0 ]] || fail "run as root"
action=${1:-}

git_source() {
    git -c "safe.directory=${source_root}" -C "${source_root}" "$@"
}

prepare_administrator_git_transport() {
    administrator=${SUDO_USER:-}
    [[ -n "${administrator}" && "${administrator}" != root ]] || \
        fail "install must run through sudo from the administrator account"
    id "${administrator}" >/dev/null 2>&1 || fail "administrator account does not exist"
    [[ "${SUDO_UID:-}" == "$(id -u "${administrator}")" ]] || \
        fail "sudo caller identity does not match the administrator account"
    administrator_home=$(getent passwd "${administrator}" | cut -d: -f6)
    [[ "${administrator_home}" == /* && "${administrator_home}" != / && \
        -d "${administrator_home}" ]] || fail "administrator home is invalid"
    command -v runuser >/dev/null 2>&1 || \
        fail "runuser is required to isolate administrator Git access"
    command -v ssh-keygen >/dev/null 2>&1 || \
        fail "ssh-keygen is required to verify the private-repository host key"
    private_git_runtime=$(mktemp -d "${deployment_root}/.a3-git-transport.XXXXXX")
    chmod 0711 "${private_git_runtime}"
    private_git_known_hosts=${private_git_runtime}/known_hosts
    printf '%s\n' "${github_host_key}" >"${private_git_known_hosts}"
    ssh-keygen -lf "${private_git_known_hosts}" -E sha256 | \
        grep -F -- "${github_host_key_fingerprint}" >/dev/null || \
        fail "pinned GitHub host key fingerprint verification failed"
    chown "${administrator}":root "${private_git_known_hosts}"
    chmod 0600 "${private_git_known_hosts}"

    private_git_ssh_wrapper=${private_git_runtime}/ssh
    {
        printf '#!/bin/bash\n'
        printf 'exec /usr/sbin/runuser --user %q -- /usr/bin/env -i ' "${administrator}"
        printf 'HOME=%q USER=%q LOGNAME=%q PATH=/usr/bin:/bin ' \
            "${administrator_home}" "${administrator}" "${administrator}"
        printf '/usr/bin/ssh -o BatchMode=yes -o StrictHostKeyChecking=yes '
        printf '%s ' '-o HostKeyAlgorithms=ssh-ed25519'
        printf '%s ' "-o UserKnownHostsFile=${private_git_known_hosts}"
        printf '%s ' '-o GlobalKnownHostsFile=/dev/null'
        printf '%s ' '-o IdentityAgent=none' '-o IdentitiesOnly=yes'
        printf '"$@"\n'
    } >"${private_git_ssh_wrapper}"
    chmod 0700 "${private_git_ssh_wrapper}"
}

private_uv() {
    [[ -n "${administrator}" ]] || fail "administrator Git identity is unavailable"
    [[ -f "${private_git_known_hosts}" ]] || fail "private Git host key is unavailable"
    [[ -x "${private_git_ssh_wrapper}" ]] || fail "private Git SSH wrapper is unavailable"
    GIT_SSH_COMMAND="${private_git_ssh_wrapper}" GIT_SSH_VARIANT=ssh \
        GIT_LFS_SKIP_SMUDGE=1 command uv "$@"
}

preinstall_registry_from_mirror() {
    local project=$1
    local mirror=${A3_PYPI_MIRROR:-}
    [[ -n "${mirror}" ]] || return 0
    [[ "${mirror}" =~ ^https://[^[:space:]]+$ ]] || \
        fail "A3_PYPI_MIRROR must be an HTTPS package index"
    local requirements=${project}/.a3-mirror-requirements.txt
    UV_PYTHON_INSTALL_DIR=/opt/a3/python \
        private_uv export --project "${project}" --all-packages --frozen \
            --extra local-controller --no-dev \
        --no-emit-project --no-emit-package lerobot-robot-a3 \
        --no-emit-package el-a3-sdk --no-emit-package lerobot \
        --no-emit-package torch --no-emit-package torchvision \
        --format requirements-txt --output-file "${requirements}"
    UV_PYTHON_INSTALL_DIR=/opt/a3/python \
        uv venv --python 3.12.13 "${project}/.venv"
    UV_DEFAULT_INDEX="${mirror}" UV_CONCURRENT_DOWNLOADS=8 \
        UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 \
        uv pip sync --python "${project}/.venv/bin/python" \
        --require-hashes "${requirements}"
    rm -f -- "${requirements}"
}

activate_release() {
    local commit=${1:-}
    [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || fail "release must be a full Git commit"
    local release=${release_root}/${commit}
    [[ -d "${release}" && ! -L "${release}" ]] || fail "release does not exist"
    [[ -f "${release}/${release_marker}" && ! -L "${release}/${release_marker}" ]] || \
        fail "release is incomplete"
    [[ "$(<"${release}/${release_marker}")" == "${commit}" ]] || \
        fail "release completion marker does not match commit"
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
        prepare_administrator_git_transport
        commit=$(git_source rev-parse HEAD)
        [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || fail "source commit is invalid"
        destination=${release_root}/${commit}
        [[ ! -e "${destination}" && ! -L "${destination}" ]] || \
            fail "release already exists; releases are immutable"
        temporary=${release_root}/.${commit}.installing
        [[ ! -e "${temporary}" ]] || fail "stale release installation exists"
        install -d -m 0750 -o root -g "${collab_group}" "${temporary}"
        git_source archive --format=tar "${commit}" | \
            tar -xf - -C "${temporary}"
        # Virtual environments embed absolute interpreter paths in their launchers.
        # Put the source at its final path before creating .venv; activation remains
        # gated by the completion marker below.
        mv -- "${temporary}" "${destination}"
        incomplete_release_destination=${destination}
        incomplete_release_temporary=${temporary}
        incomplete_release_owned=true
        preinstall_registry_from_mirror "${destination}"
        UV_PYTHON_INSTALL_DIR=/opt/a3/python \
            private_uv sync --project "${destination}" --all-packages --frozen \
            --extra local-controller --no-dev --no-editable
        printf '%s\n' "${commit}" >"${destination}/${release_marker}"
        chown -R root:"${collab_group}" "${destination}"
        chmod -R u=rwX,g=rX,o= "${destination}"
        incomplete_release_owned=false
        activate_release "${commit}"
        ;;
    activate)
        activate_release "${2:-}"
        ;;
    *)
        fail "usage: $0 install <clean-source-checkout> | activate <full-commit>"
        ;;
esac
