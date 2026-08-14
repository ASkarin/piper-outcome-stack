#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKSPACE_ROOT="/workspace"
readonly CONFIG_ROOT="/run/piper-config"
readonly SSH_HOST_KEY_ROOT="/etc/ssh/piper_host_keys"
readonly RUNTIME_CONFIG="/etc/piper/runtime.json"

if [[ -x "${SCRIPT_DIR}/init-shared-python.sh" && -d "${SCRIPT_DIR}/bin" ]]; then
    readonly PYTHON_INITIALIZER="${SCRIPT_DIR}/init-shared-python.sh"
    readonly COMMAND_SOURCE="${SCRIPT_DIR}/bin"
    readonly LIB_SOURCE="${SCRIPT_DIR}/lib"
    readonly PROFILE_SOURCE="${SCRIPT_DIR}/profile.sh"
else
    readonly PYTHON_INITIALIZER="/usr/local/sbin/piper-init-shared-python"
    readonly COMMAND_SOURCE="/usr/local/bin"
    readonly LIB_SOURCE="/usr/local/lib/piper-container"
    readonly PROFILE_SOURCE="/etc/profile.d/piper.sh"
fi

fail() {
    echo "piper-entrypoint: $*" >&2
    exit 2
}

require_var() {
    local name="$1"
    test -n "${!name:-}" || fail "required environment variable ${name} is missing"
}

validate_user_name() {
    local value="$1"
    [[ "${value}" =~ ^[a-z_][a-z0-9_-]{0,30}$ ]] \
        || fail "invalid Linux user name: ${value}"
}

validate_id() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] && (( value >= 1000 && value <= 60000 )) \
        || fail "${name} must be an integer from 1000 through 60000"
}

ensure_project_group() {
    local name="$1"
    local gid="$2"
    local existing

    existing="$(getent group "${name}" || true)"
    if [[ -n "${existing}" ]]; then
        [[ "$(cut -d: -f3 <<<"${existing}")" == "${gid}" ]] \
            || fail "group ${name} exists with a different GID"
        return
    fi

    if getent group "${gid}" >/dev/null; then
        fail "requested project GID ${gid} is already in use"
    fi
    groupadd --gid "${gid}" "${name}"
}

ensure_user() {
    local name="$1"
    local uid="$2"
    local home="$3"
    local project_group="$4"
    local existing
    local private_group

    existing="$(getent passwd "${name}" || true)"
    if [[ -n "${existing}" ]]; then
        [[ "$(cut -d: -f3 <<<"${existing}")" == "${uid}" ]] \
            || fail "user ${name} exists with a different UID"
        [[ "$(cut -d: -f4 <<<"${existing}")" == "${uid}" ]] \
            || fail "user ${name} exists with a different primary GID"
        [[ "$(cut -d: -f6 <<<"${existing}")" == "${home}" ]] \
            || fail "user ${name} exists with a different home"
        [[ "$(cut -d: -f7 <<<"${existing}")" == "/bin/bash" ]] \
            || fail "user ${name} exists with a different shell"
        private_group="$(getent group "${name}" || true)"
        [[ -n "${private_group}" && "$(cut -d: -f3 <<<"${private_group}")" == "${uid}" ]] \
            || fail "private group ${name} is missing or has a different GID"
    else
        if getent passwd "${uid}" >/dev/null; then
            fail "requested UID ${uid} is already in use"
        fi
        if getent group "${name}" >/dev/null; then
            fail "private group ${name} already exists"
        fi
        if getent group "${uid}" >/dev/null; then
            fail "private GID ${uid} is already in use"
        fi
        groupadd --gid "${uid}" "${name}"
        useradd \
            --uid "${uid}" \
            --gid "${name}" \
            --groups "${project_group}" \
            --home-dir "${home}" \
            --shell /bin/bash \
            --no-create-home \
            "${name}"
    fi

    usermod --append --groups "${project_group}" "${name}"
    install -d -m 0700 -o "${name}" -g "${name}" "${home}"
}

enable_public_key_account() {
    local user="$1"
    local random_password

    # OpenSSH rejects a shadow-locked account before it evaluates authorized_keys.
    # Give the account an unknown, short-lived random password so public-key
    # authentication can proceed. Password and keyboard-interactive authentication
    # remain disabled in sshd_config, and the random plaintext is immediately lost.
    random_password="$(head -c 48 /dev/urandom | base64 | tr -d '\n')"
    printf '%s:%s\n' "${user}" "${random_password}" | chpasswd
    unset random_password
}

install_authorized_keys() {
    local user="$1"
    local source="$2"
    local home
    local ssh_directory
    local destination

    [[ -s "${source}" ]] || fail "authorized_keys file is missing or empty: ${source}"
    home="$(getent passwd "${user}" | cut -d: -f6)"
    ssh_directory="${home}/.ssh"
    destination="${ssh_directory}/authorized_keys"
    [[ ! -L "${ssh_directory}" ]] || fail "refusing symbolic-link SSH directory: ${ssh_directory}"
    [[ ! -e "${ssh_directory}" || -d "${ssh_directory}" ]] \
        || fail "SSH path is not a directory: ${ssh_directory}"
    [[ ! -L "${destination}" ]] || fail "refusing symbolic-link authorized_keys: ${destination}"
    [[ ! -e "${destination}" || -f "${destination}" ]] \
        || fail "authorized_keys path is not a regular file: ${destination}"
    install -d -m 0700 -o "${user}" -g "${user}" "${ssh_directory}"
    install -m 0600 -o "${user}" -g "${user}" "${source}" "${destination}"
}

install_shell_startup() {
    local user="$1"
    local home
    local startup
    local source_line='source /workspace/piper/profile.sh'

    home="$(getent passwd "${user}" | cut -d: -f6)"
    for startup in "${home}/.profile" "${home}/.bashrc"; do
        [[ ! -L "${startup}" ]] || fail "refusing symbolic-link shell startup file: ${startup}"
        [[ ! -e "${startup}" || -f "${startup}" ]] \
            || fail "shell startup path is not a regular file: ${startup}"
        if [[ ! -e "${startup}" ]]; then
            install -m 0644 -o "${user}" -g "${user}" /dev/null "${startup}"
        fi
        sed -i '\|^source /etc/profile.d/piper.sh$|d' "${startup}"
        if ! grep -Fqx "${source_line}" "${startup}"; then
            printf '\n%s\n' "${source_line}" >>"${startup}"
        fi
        chown "${user}:${user}" "${startup}"
        chmod 0644 "${startup}"
    done
}

set_read_acl() {
    local path="$1"
    setfacl -m "u::rwx,g::r-x,o::---,m::r-x" "${path}"
    setfacl -m "d:u::rwx,d:g::r-x,d:o::---,d:m::r-x" "${path}"
}

set_write_acl() {
    local path="$1"
    setfacl -m "u::rwx,g::rwx,o::---,m::rwx" "${path}"
    setfacl -m "d:u::rwx,d:g::rwx,d:o::---,d:m::rwx" "${path}"
}

for variable in \
    PIPER_ADMIN_USER \
    PIPER_ADMIN_UID \
    PIPER_COLLAB_USER \
    PIPER_COLLAB_UID \
    PIPER_GROUP_GID \
    PIPER_GROUP_NAME
do
    require_var "${variable}"
done

validate_user_name "${PIPER_ADMIN_USER}"
validate_user_name "${PIPER_COLLAB_USER}"
[[ "${PIPER_ADMIN_USER}" != "${PIPER_COLLAB_USER}" ]] || fail "admin and collaborator must differ"
validate_id PIPER_ADMIN_UID "${PIPER_ADMIN_UID}"
validate_id PIPER_COLLAB_UID "${PIPER_COLLAB_UID}"
validate_id PIPER_GROUP_GID "${PIPER_GROUP_GID}"

ensure_project_group "${PIPER_GROUP_NAME}" "${PIPER_GROUP_GID}"
ensure_user \
    "${PIPER_ADMIN_USER}" \
    "${PIPER_ADMIN_UID}" \
    "${WORKSPACE_ROOT}/users/${PIPER_ADMIN_USER}" \
    "${PIPER_GROUP_NAME}"
ensure_user \
    "${PIPER_COLLAB_USER}" \
    "${PIPER_COLLAB_UID}" \
    "${WORKSPACE_ROOT}/users/${PIPER_COLLAB_USER}" \
    "${PIPER_GROUP_NAME}"

enable_public_key_account "${PIPER_ADMIN_USER}"
enable_public_key_account "${PIPER_COLLAB_USER}"

usermod --groups "${PIPER_GROUP_NAME},sudo" "${PIPER_ADMIN_USER}"
usermod --groups "${PIPER_GROUP_NAME}" "${PIPER_COLLAB_USER}"

[[ -d "${WORKSPACE_ROOT}" && ! -L "${WORKSPACE_ROOT}" ]] \
    || fail "workspace mount must be a real directory: ${WORKSPACE_ROOT}"
setfacl -m \
    "u:${PIPER_ADMIN_USER}:rwx,u:${PIPER_COLLAB_USER}:r-x,m::rwx" \
    "${WORKSPACE_ROOT}"

{
    printf '%s\n' \
        'Defaults secure_path="/workspace/piper/bin:/workspace/piper/python-env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"'
    printf '%s ALL=(ALL:ALL) NOPASSWD: ALL\n' "${PIPER_ADMIN_USER}"
} >"/etc/sudoers.d/piper-admin"
chmod 0440 /etc/sudoers.d/piper-admin
visudo --check --file /etc/sudoers.d/piper-admin >/dev/null

install_authorized_keys \
    "${PIPER_ADMIN_USER}" \
    "${CONFIG_ROOT}/admin_authorized_keys"
install_authorized_keys \
    "${PIPER_COLLAB_USER}" \
    "${CONFIG_ROOT}/collaborator_authorized_keys"

install -d -m 2750 -o "${PIPER_ADMIN_USER}" -g "${PIPER_GROUP_NAME}" \
    "${WORKSPACE_ROOT}/projects" \
    "${WORKSPACE_ROOT}/projects/piper-outcome-stack"

for user in "${PIPER_ADMIN_USER}" "${PIPER_COLLAB_USER}"; do
    install -d -m 2750 -o "${user}" -g "${PIPER_GROUP_NAME}" \
        "${WORKSPACE_ROOT}/piper/staging/${user}" \
        "${WORKSPACE_ROOT}/piper/runs/${user}"
    set_read_acl "${WORKSPACE_ROOT}/piper/staging/${user}"
    set_read_acl "${WORKSPACE_ROOT}/piper/runs/${user}"
    install -d -m 0750 -o "${user}" -g "${PIPER_GROUP_NAME}" \
        "${WORKSPACE_ROOT}/users/${user}/src"
done

install -d -m 2750 -o root -g "${PIPER_GROUP_NAME}" \
    "${WORKSPACE_ROOT}/piper/releases" \
    "${WORKSPACE_ROOT}/piper/releases/datasets" \
    "${WORKSPACE_ROOT}/piper/releases/models"
set_read_acl "${WORKSPACE_ROOT}/piper/releases"
set_read_acl "${WORKSPACE_ROOT}/piper/releases/datasets"
set_read_acl "${WORKSPACE_ROOT}/piper/releases/models"

install -d -m 2770 -o "${PIPER_ADMIN_USER}" -g "${PIPER_GROUP_NAME}" \
    "${WORKSPACE_ROOT}/piper/cache" \
    "${WORKSPACE_ROOT}/piper/cache/huggingface" \
    "${WORKSPACE_ROOT}/piper/cache/torch" \
    "${WORKSPACE_ROOT}/piper/locks"
set_write_acl "${WORKSPACE_ROOT}/piper/cache"
set_write_acl "${WORKSPACE_ROOT}/piper/cache/huggingface"
set_write_acl "${WORKSPACE_ROOT}/piper/cache/torch"
set_write_acl "${WORKSPACE_ROOT}/piper/locks"

install -d -m 0700 -o "${PIPER_ADMIN_USER}" -g "${PIPER_ADMIN_USER}" \
    "${WORKSPACE_ROOT}/piper/admin"

env \
    PIPER_ADMIN_USER="${PIPER_ADMIN_USER}" \
    PIPER_GROUP_NAME="${PIPER_GROUP_NAME}" \
    PIPER_CONTAINER_BIN_SOURCE="${COMMAND_SOURCE}" \
    PIPER_CONTAINER_LIB_SOURCE="${LIB_SOURCE}" \
    PIPER_PROFILE_SOURCE="${PROFILE_SOURCE}" \
    "${PYTHON_INITIALIZER}"
install_shell_startup "${PIPER_ADMIN_USER}"
install_shell_startup "${PIPER_COLLAB_USER}"

install -d -m 0755 /etc/piper
/workspace/piper/python-env/bin/python - "${RUNTIME_CONFIG}" <<'PY'
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 2,
    "workspace_root": "/workspace",
    "admin_user": os.environ["PIPER_ADMIN_USER"],
    "collaborator_user": os.environ["PIPER_COLLAB_USER"],
    "group_name": os.environ["PIPER_GROUP_NAME"],
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
output.chmod(0o644)
PY

install -d -m 0700 "${SSH_HOST_KEY_ROOT}"
if [[ ! -s "${SSH_HOST_KEY_ROOT}/ssh_host_ed25519_key" ]]; then
    ssh-keygen -q -t ed25519 -N '' -f "${SSH_HOST_KEY_ROOT}/ssh_host_ed25519_key"
fi
if [[ ! -s "${SSH_HOST_KEY_ROOT}/ssh_host_rsa_key" ]]; then
    ssh-keygen -q -t rsa -b 3072 -N '' -f "${SSH_HOST_KEY_ROOT}/ssh_host_rsa_key"
fi
chmod 0600 "${SSH_HOST_KEY_ROOT}"/ssh_host_*_key
chmod 0644 "${SSH_HOST_KEY_ROOT}"/ssh_host_*_key.pub

cp /etc/ssh/sshd_config.piper /run/sshd_config
printf '\nAllowUsers %s %s\n' "${PIPER_ADMIN_USER}" "${PIPER_COLLAB_USER}" >>/run/sshd_config
mkdir -p /run/sshd
/usr/sbin/sshd -t -f /run/sshd_config

exec /usr/sbin/sshd -D -e -f /run/sshd_config
