#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export LANGUAGE=C
export DEBIAN_FRONTEND=noninteractive

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_group=piper
collab_group=piper-collab
project_root=/opt/piper-outcome-stack
state_root=/var/lib/piper-outcome-stack
log_root=/var/log/piper-outcome-stack
config_root=/etc/piper-outcome-stack
runtime_root=/run/piper-outcome-stack
dropin=/etc/ssh/sshd_config.d/60-piper-local-hardening.conf
rollback_unit=piper-local-security-rollback

fail() {
    echo "error: $*" >&2
    exit 1
}

require_root() {
    [[ "${EUID}" -eq 0 ]] || fail "run this phase as root"
}

administrator_account() {
    local account=${SUDO_USER:-}
    [[ -n "${account}" && "${account}" != "root" ]] || \
        fail "run through sudo from the administrator account"
    id "${account}" >/dev/null 2>&1 || fail "administrator account does not exist"
    printf '%s\n' "${account}"
}

current_ssh_uses_tailscale() {
    local connection=${SSH_CONNECTION:-}
    [[ -n "${connection}" ]] || \
        fail "preserve SSH_CONNECTION through sudo for the security phase"
    local peer=${connection%% *}
    local route
    [[ -n "${peer}" ]] || return 1
    if [[ "${peer}" == *:* ]]; then
        route=$(ip -6 route get "${peer}" 2>/dev/null) || return 1
    else
        route=$(ip route get "${peer}" 2>/dev/null) || return 1
    fi
    grep -q 'dev tailscale0' <<<"${route}"
}

assert_administrator_key() {
    local administrator=$1
    local home
    home=$(getent passwd "${administrator}" | cut -d: -f6)
    [[ -n "${home}" && "${home}" != "/" ]] || fail "unsafe administrator home"
    [[ -s "${home}/.ssh/authorized_keys" ]] || \
        fail "administrator has no authorized SSH public key"
    ssh-keygen -l -f "${home}/.ssh/authorized_keys" >/dev/null || \
        fail "administrator authorized_keys is invalid"
}

assert_sshd_policy() {
    local effective expected
    effective=$(/usr/sbin/sshd -T)
    for expected in \
        "permitrootlogin no" \
        "passwordauthentication no" \
        "kbdinteractiveauthentication no" \
        "authenticationmethods publickey" \
        "allowagentforwarding yes" \
        "allowtcpforwarding no" \
        "allowstreamlocalforwarding no" \
        "permittunnel no"; do
        grep -Fqx "${expected}" <<<"${effective}" || \
            fail "effective sshd policy mismatch: ${expected}"
    done
}

assert_ufw_policy() {
    local added status_output unexpected
    status_output=$(ufw status verbose)
    grep -q '^Status: active' <<<"${status_output}" || fail "UFW is not active"
    grep -q '^Default: deny (incoming), allow (outgoing)' <<<"${status_output}" || \
        fail "UFW default policy mismatch"
    added=$(ufw show added)
    grep -Eq '^ufw allow in on tailscale0 to any port [0-9]+ proto tcp' <<<"${added}" || \
        fail "UFW has no Tailscale-only SSH rule"
    unexpected=$(grep -E '^ufw allow' <<<"${added}" | \
        grep -Ev '^ufw allow in on tailscale0 to any port [0-9]+ proto tcp( comment .*)?$' || true)
    [[ -z "${unexpected}" ]] || fail "UFW contains an unexpected allow rule"
}

install_base() {
    require_root
    local administrator
    administrator=$(administrator_account)
    hostnamectl set-hostname piper-local
    apt-get update
    apt-get install -y --no-install-recommends \
        acl build-essential ca-certificates can-utils cmake curl evtest ffmpeg git \
        iproute2 jq joystick ninja-build openssl pciutils pkg-config rsync ufw \
        usbutils util-linux v4l-utils
    groupadd --force "${project_group}"
    groupadd --force "${collab_group}"
    usermod --append --groups "${project_group},${collab_group}" "${administrator}"
    install -d -m 0755 -o root -g root "${project_root}"
    install -d -m 2750 -o root -g "${collab_group}" \
        "${project_root}/releases" "${log_root}"
    install -d -m 0750 -o root -g root "${config_root}" "${state_root}/admin"
    install -m 0755 "${script_dir}/security-rollback.sh" \
        /usr/local/sbin/piper-local-security-rollback
    install -m 0755 "${script_dir}/deploy-release.sh" \
        /usr/local/sbin/piper-local-deploy-release
    install -m 0755 "${script_dir}/piper-socketcan.sh" \
        /usr/local/sbin/piper-socketcan
    install -m 0644 "${script_dir}/piper-socketcan-isolate@.service" \
        /etc/systemd/system/piper-socketcan-isolate@.service
    systemctl daemon-reload
    if [[ "$(uv --version 2>/dev/null || true)" != "uv 0.11.32"* ]]; then
        local temporary
        temporary=$(mktemp -d)
        trap 'rm -rf -- "${temporary}"' RETURN
        curl -LsSf https://astral.sh/uv/0.11.32/install.sh \
            -o "${temporary}/install-uv.sh"
        UV_INSTALL_DIR=/usr/local/bin sh "${temporary}/install-uv.sh"
        rm -rf -- "${temporary}"
        trap - RETURN
    fi
    install -d -m 0755 -o root -g root /opt/piper/python
    UV_PYTHON_INSTALL_DIR=/opt/piper/python uv python install 3.12.13
    chmod -R u=rwX,go=rX /opt/piper/python
}

install_gpu() {
    require_root
    administrator_account >/dev/null
    apt-get update
    apt-get install -y nvidia-driver-595
    install -d -m 0700 -o root -g root "${state_root}/admin"
    date -u +%Y-%m-%dT%H:%M:%SZ >"${state_root}/admin/gpu-reboot-required"
    chmod 0600 "${state_root}/admin/gpu-reboot-required"
    echo "GPU driver installed; reboot was not performed automatically."
}

install_security() {
    require_root
    local administrator
    administrator=$(administrator_account)
    assert_administrator_key "${administrator}"
    [[ "${PIPER_CONFIRM_CONSOLE:-}" == "YES" ]] || \
        fail "set PIPER_CONFIRM_CONSOLE=YES only with physical-console recovery available"
    ip link show tailscale0 >/dev/null 2>&1 || fail "tailscale0 is unavailable"
    current_ssh_uses_tailscale || fail "current SSH session is not routed through tailscale0"
    install -d -m 0755 -o root -g root /etc/ssh/sshd_config.d "${runtime_root}"
    [[ ! -e "${runtime_root}/security-pending" ]] || \
        fail "a security change is already pending confirmation"
    local timestamp backup ssh_port ufw_status ufw_was_active had_dropin rollback_instance
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    backup="${state_root}/admin/security-backups/${timestamp}"
    rollback_instance="${rollback_unit}-${timestamp}"
    install -d -m 0700 -o root -g root "${backup}"
    cp -a -- /etc/default/ufw "${backup}/default-ufw"
    cp -a -- /etc/ufw "${backup}/ufw"
    ufw_was_active=no
    ufw_status=$(ufw status)
    grep -q '^Status: active' <<<"${ufw_status}" && ufw_was_active=yes || true
    had_dropin=no
    if [[ -f "${dropin}" ]]; then
        had_dropin=yes
        cp -a -- "${dropin}" "${backup}/sshd-hardening.conf"
    fi
    {
        printf 'ufw_was_active=%q\n' "${ufw_was_active}"
        printf 'had_dropin=%q\n' "${had_dropin}"
        printf 'rollback_instance=%q\n' "${rollback_instance}"
    } >"${backup}/metadata"
    chmod 0600 "${backup}/metadata"

    systemd-run --quiet --unit="${rollback_instance}" --on-active=10m \
        /usr/local/sbin/piper-local-security-rollback "${backup}"
    printf '%s\n' "${backup}" >"${runtime_root}/security-pending"
    chmod 0600 "${runtime_root}/security-pending"

    install -m 0644 "${script_dir}/sshd-hardening.conf" "${dropin}"
    /usr/sbin/sshd -t
    assert_sshd_policy
    ssh_port=$(/usr/sbin/sshd -T | \
        awk '$1 == "port" && !seen {print $2; seen=1}')
    [[ "${ssh_port}" =~ ^[0-9]+$ ]] || fail "could not determine the SSH service port"
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow in on tailscale0 proto tcp to any port "${ssh_port}" comment 'PiPER Tailscale SSH'
    ufw --force enable
    systemctl reload ssh
    assert_ufw_policy
    echo "Security changes are pending. Confirm from a second Tailscale SSH session."
}

confirm_security() {
    require_root
    current_ssh_uses_tailscale || fail "confirmation session is not routed through tailscale0"
    [[ -f "${runtime_root}/security-pending" ]] || fail "no pending security change"
    local backup rollback_instance
    backup=$(<"${runtime_root}/security-pending")
    [[ "${backup}" == "${state_root}/admin/security-backups/"* ]] || \
        fail "pending security backup path is invalid"
    [[ -f "${backup}/metadata" ]] || fail "pending security backup metadata is absent"
    # shellcheck disable=SC1090
    source "${backup}/metadata"
    [[ "${rollback_instance}" =~ ^piper-local-security-rollback-[0-9]{8}T[0-9]{6}Z$ ]] || \
        fail "pending rollback unit is invalid"
    /usr/sbin/sshd -t
    assert_sshd_policy
    assert_ufw_policy
    systemctl stop "${rollback_instance}.timer"
    ! systemctl is-active --quiet "${rollback_instance}.service" || \
        fail "security rollback is already running"
    rm -f -- "${runtime_root}/security-pending"
    systemctl reset-failed "${rollback_instance}.service" >/dev/null 2>&1 || true
    echo "Security rollback timer cancelled after successful second-session check."
}

case "${1:-}" in
    base) install_base ;;
    gpu) install_gpu ;;
    security) install_security ;;
    confirm-security) confirm_security ;;
    check) exec bash "${script_dir}/piper-local-doctor.sh" ;;
    *) fail "usage: $0 base|gpu|security|confirm-security|check" ;;
esac
