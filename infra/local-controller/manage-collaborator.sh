#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

collab_group=piper-collab
archive_root=/var/lib/piper-outcome-stack/admin/revocations
managed_root=/var/lib/piper-outcome-stack/admin/collaborators

fail() {
    echo "error: $*" >&2
    exit 1
}

[[ "${EUID}" -eq 0 ]] || fail "run as root"
action=${1:-}
account=${2:-}
[[ "${account}" =~ ^[a-z_][a-z0-9_-]*$ ]] || fail "invalid account name"

case "${action}" in
    provision)
        key_file=${3:-}
        [[ $# -eq 3 ]] || fail "provision requires an account and one public-key file"
        [[ "${PIPER_TAILNET_GRANT_CONFIRMED:-}" == "YES" ]] || \
            fail "confirm the individual tailnet Grant first"
        [[ -f "${key_file}" && -s "${key_file}" ]] || fail "missing public-key file"
        [[ "$(grep -cve '^[[:space:]]*$' "${key_file}")" -eq 1 ]] || \
            fail "public-key file must contain exactly one key"
        key_type=$(awk 'NF {print $1}' "${key_file}")
        case "${key_type}" in
            ssh-ed25519|ssh-rsa|ecdsa-sha2-*|sk-ssh-ed25519@openssh.com|sk-ecdsa-sha2-*) ;;
            *) fail "public-key file has an unsupported key type" ;;
        esac
        ssh-keygen -l -f "${key_file}" >/dev/null || fail "invalid public-key file"
        getent group "${collab_group}" >/dev/null || fail "collaborator group is absent"
        ! id "${account}" >/dev/null 2>&1 || fail "account already exists"
        adduser --disabled-password --gecos "" "${account}"
        account_uid=$(id -u "${account}")
        [[ "${account_uid}" -ge 1000 ]] || fail "collaborator UID is outside the human range"
        random_password=$(openssl rand -base64 48)
        password_hash=$(openssl passwd -6 "${random_password}")
        unset random_password
        usermod --password "${password_hash}" "${account}"
        unset password_hash
        usermod --shell /bin/bash --groups "${collab_group}" "${account}"
        home=$(getent passwd "${account}" | cut -d: -f6)
        [[ -n "${home}" && "${home}" != "/" ]] || fail "unsafe home directory"
        install -d -m 0700 -o "${account}" -g "${account}" "${home}/.ssh"
        install -m 0600 -o "${account}" -g "${account}" \
            "${key_file}" "${home}/.ssh/authorized_keys"
        privileged=$(id -nG "${account}" | tr ' ' '\n' | \
            grep -E '^(sudo|adm|lxd|docker|disk|dialout|input|video|render|plugdev)$' || true)
        [[ -z "${privileged}" ]] || fail "account retained a privileged group"
        install -d -m 0700 -o root -g root "${managed_root}"
        printf 'account=%s\nuid=%s\nprovisioned_at_utc=%s\n' \
            "${account}" "${account_uid}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"${managed_root}/${account}"
        chmod 0600 "${managed_root}/${account}"
        ;;
    revoke)
        [[ "${PIPER_TAILNET_GRANT_REVOKED:-}" == "YES" ]] || \
            fail "remove and confirm the individual tailnet Grant first"
        id "${account}" >/dev/null 2>&1 || fail "account does not exist"
        [[ -f "${managed_root}/${account}" && ! -L "${managed_root}/${account}" ]] || \
            fail "account was not provisioned by this manager"
        account_uid=$(id -u "${account}")
        [[ "${account_uid}" -ge 1000 ]] || fail "refusing to revoke a system account"
        recorded_account=$(awk -F= '$1 == "account" {print $2}' "${managed_root}/${account}")
        recorded_uid=$(awk -F= '$1 == "uid" {print $2}' "${managed_root}/${account}")
        [[ "${recorded_account}" == "${account}" && "${recorded_uid}" == "${account_uid}" ]] || \
            fail "managed-account record does not match the live account"
        home=$(getent passwd "${account}" | cut -d: -f6)
        [[ -n "${home}" && "${home}" != "/" ]] || fail "unsafe home directory"
        timestamp=$(date -u +%Y%m%dT%H%M%SZ)
        archive="${archive_root}/${timestamp}-${account}"
        install -d -m 0700 -o root -g root "${archive}"
        install -m 0600 -o root -g root \
            "${managed_root}/${account}" "${archive}/account-record"
        if [[ -f "${home}/.ssh/authorized_keys" ]]; then
            install -m 0600 -o root -g root \
                "${home}/.ssh/authorized_keys" "${archive}/authorized_keys"
            : >"${home}/.ssh/authorized_keys"
            chown "${account}:${account}" "${home}/.ssh/authorized_keys"
            chmod 0600 "${home}/.ssh/authorized_keys"
        fi
        usermod --groups "" --lock --shell /usr/sbin/nologin "${account}"
        rm -f -- "${managed_root}/${account}"
        ;;
    *)
        fail "usage: $0 provision <account> <public-key-file> | revoke <account>"
        ;;
esac
