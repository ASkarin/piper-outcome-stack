#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

backup=${1:?security backup directory is required}
dropin=/etc/ssh/sshd_config.d/60-piper-local-hardening.conf

if [[ ! -d "${backup}" || ! -f "${backup}/metadata" ]]; then
    echo "invalid security backup" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${backup}/metadata"

if [[ "${had_dropin}" == "yes" ]]; then
    install -m 0644 "${backup}/sshd-hardening.conf" "${dropin}"
else
    rm -f -- "${dropin}"
fi
cp -a -- "${backup}/default-ufw" /etc/default/ufw
cp -a -- "${backup}/ufw/." /etc/ufw/

/usr/sbin/sshd -t
systemctl reload ssh
if [[ "${ufw_was_active}" == "yes" ]]; then
    ufw --force enable
else
    ufw --force disable
fi
rm -f -- /run/piper-outcome-stack/security-pending
