#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
helper=${script_dir}/../piper-socketcan.sh
namespace=piper-can
interface=piper-ci-can

fail() {
    echo "error: $*" >&2
    exit 1
}

cleanup() {
    sudo ip netns delete "${namespace}" >/dev/null 2>&1 || true
    sudo ip link delete dev "${interface}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

bind_and_send() {
    python3 - "${1}" <<'PIPER_VCAN_PROBE'
import socket
import struct
import sys

handle = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
try:
    handle.bind((sys.argv[1],))
    frame = struct.pack("=IB3x8s", 0x123, 1, b"\x01" + b"\x00" * 7)
    if handle.send(frame) != len(frame):
        raise SystemExit("short vcan frame write")
finally:
    handle.close()
PIPER_VCAN_PROBE
}

[[ "${EUID}" -ne 0 ]] || fail "run this smoke as a normal sudo-capable account"
sudo -v || fail "administrator sudo is required for the SocketCAN smoke"
! sudo ip netns exec "${namespace}" true >/dev/null 2>&1 || \
    fail "the fixed SocketCAN namespace already exists"
! ip link show dev "${interface}" >/dev/null 2>&1 || \
    fail "the smoke interface already exists"

sudo modprobe vcan
sudo ip link add dev "${interface}" type vcan
sudo ip link set dev "${interface}" up

# Lock in the original failure: a normal user can control a visible vcan interface.
bind_and_send "${interface}"
sudo ip link set dev "${interface}" down

sudo bash "${helper}" isolate "${interface}"
if bind_and_send "${interface}" >/dev/null 2>&1; then
    fail "the isolated CAN interface remained reachable from the host namespace"
fi

sudo ip -n "${namespace}" link set dev "${interface}" up
administrator_uid=$(id -u)
sudo bash "${helper}" exec -- \
    python3 - "${interface}" "${administrator_uid}" <<'PIPER_NAMESPACE_PROBE'
import os
import socket
import struct
import sys

interface = sys.argv[1]
expected_uid = int(sys.argv[2])
if os.geteuid() != expected_uid or os.geteuid() == 0:
    raise SystemExit("SocketCAN launcher did not restore the administrator UID")

cap_eff = None
with open("/proc/self/status", encoding="utf-8") as status_file:
    for line in status_file:
        if line.startswith("CapEff:"):
            cap_eff = line.split()[1]
            break
if cap_eff is None or int(cap_eff, 16) != 0:
    raise SystemExit("SocketCAN launcher retained effective capabilities")

handle = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
try:
    handle.bind((interface,))
    frame = struct.pack("=IB3x8s", 0x124, 1, b"\x02" + b"\x00" * 7)
    if handle.send(frame) != len(frame):
        raise SystemExit("short isolated vcan frame write")
finally:
    handle.close()
PIPER_NAMESPACE_PROBE

! ip link show dev "${interface}" >/dev/null 2>&1 || \
    fail "the target CAN interface returned to the host namespace"
sudo ip -n "${namespace}" link show dev "${interface}" >/dev/null
sudo bash "${helper}" down "${interface}"
sudo ip -n "${namespace}" -details link show dev "${interface}" | \
    grep -Fq 'state DOWN'
