#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

namespace=piper-can

fail() {
    echo "error: $*" >&2
    exit 1
}

require_root() {
    [[ "${EUID}" -eq 0 ]] || fail "run through sudo from the administrator account"
}

administrator_account() {
    local account=${SUDO_USER:-}
    [[ -n "${account}" && "${account}" != root ]] || \
        fail "SUDO_USER must identify the administrator account"
    id "${account}" >/dev/null 2>&1 || fail "administrator account does not exist"
    printf '%s\n' "${account}"
}

validate_interface() {
    local interface=$1
    [[ "${interface}" =~ ^[[:alnum:]][[:alnum:]_.-]{0,14}$ ]] || \
        fail "invalid CAN interface name"
}

require_namespace() {
    ip netns exec "${namespace}" true >/dev/null 2>&1 || \
        fail "SocketCAN namespace is absent; isolate the verified interface first"
}

require_isolated_interface() {
    local interface=$1
    require_namespace
    ! ip link show dev "${interface}" >/dev/null 2>&1 || \
        fail "target CAN interface is still visible in the host namespace"
    ip -n "${namespace}" link show dev "${interface}" >/dev/null 2>&1 || \
        fail "target CAN interface is absent from the SocketCAN namespace"
}

require_interface_down() {
    local context=$1
    local interface=$2
    local link
    case "${context}" in
        host) link=$(ip -o link show dev "${interface}") ;;
        namespace) link=$(ip -n "${namespace}" -o link show dev "${interface}") ;;
        *) fail "invalid interface context" ;;
    esac
    ! grep -Eq '<([^>]*,)?UP(,|>)' <<<"${link}" || \
        fail "target CAN interface must already be down"
}

isolate_interface() {
    local interface=${1:-}
    [[ $# -eq 1 ]] || fail "isolate requires one explicit CAN interface"
    validate_interface "${interface}"
    ip link show dev "${interface}" >/dev/null 2>&1 || \
        fail "target CAN interface is absent from the host namespace"
    [[ -r "/sys/class/net/${interface}/type" && \
        "$(<"/sys/class/net/${interface}/type")" == 280 ]] || \
        fail "target interface is not a CAN network device"
    local namespace_created=false
    if ip netns exec "${namespace}" true >/dev/null 2>&1; then
        [[ -z "$(ip netns pids "${namespace}")" ]] || \
            fail "a process is still running in the SocketCAN namespace"
        local existing_links
        existing_links=$(ip -n "${namespace}" -o link show | \
            grep -Ev '^[0-9]+: lo(:|@)' || true)
        [[ -z "${existing_links}" ]] || \
            fail "SocketCAN namespace already contains a network interface"
    else
        ip netns add "${namespace}"
        namespace_created=true
    fi
    require_interface_down host "${interface}"
    if ! ip link set dev "${interface}" netns "${namespace}"; then
        if [[ "${namespace_created}" == true ]]; then
            ip netns delete "${namespace}" >/dev/null 2>&1 || true
        fi
        fail "could not isolate the target CAN interface"
    fi
    require_isolated_interface "${interface}"
}

enable_interface() {
    local interface=${1:-}
    local bitrate=${2:-}
    [[ $# -eq 2 ]] || fail "up requires an explicit CAN interface and bitrate"
    validate_interface "${interface}"
    [[ "${bitrate}" =~ ^[1-9][0-9]*$ ]] || fail "invalid CAN bitrate"
    require_isolated_interface "${interface}"
    require_interface_down namespace "${interface}"
    [[ -z "$(ip netns pids "${namespace}")" ]] || \
        fail "a process is still running in the SocketCAN namespace"
    ip -n "${namespace}" link set dev "${interface}" type can bitrate "${bitrate}"
    ip -n "${namespace}" link set dev "${interface}" up
}

disable_interface() {
    local interface=${1:-}
    [[ $# -eq 1 ]] || fail "down requires one explicit CAN interface"
    validate_interface "${interface}"
    require_isolated_interface "${interface}"
    [[ -z "$(ip netns pids "${namespace}")" ]] || \
        fail "a process is still running in the SocketCAN namespace"
    ip -n "${namespace}" link set dev "${interface}" down
}

run_as_administrator() {
    [[ "${1:-}" == -- && $# -ge 2 ]] || fail "exec requires -- followed by a command"
    shift
    require_namespace
    local administrator uid gid home
    administrator=$(administrator_account)
    uid=$(id -u "${administrator}")
    gid=$(id -g "${administrator}")
    home=$(getent passwd "${administrator}" | cut -d: -f6)
    [[ -n "${home}" && "${home}" != / ]] || fail "administrator home is unavailable"
    exec ip netns exec "${namespace}" \
        setpriv --reuid="${uid}" --regid="${gid}" --init-groups \
        --inh-caps=-all --ambient-caps=-all --bounding-set=-all \
        env HOME="${home}" USER="${administrator}" LOGNAME="${administrator}" "$@"
}

require_root
case "${1:-}" in
    isolate) shift; isolate_interface "$@" ;;
    up) shift; enable_interface "$@" ;;
    exec) shift; run_as_administrator "$@" ;;
    down) shift; disable_interface "$@" ;;
    *) fail "usage: $0 isolate <interface> | up <interface> <bitrate> | exec -- <command> [args...] | down <interface>" ;;
esac
