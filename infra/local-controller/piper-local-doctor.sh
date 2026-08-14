#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

status=pass
hostname_ok=false
[[ "$(hostname)" == "piper-local" ]] && hostname_ok=true || status=incomplete

driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
uv_version=$(uv --version 2>/dev/null | awk '{print $2}' || true)
python_executable=$(find /opt/piper/python -type f -path '*/bin/python3.12' -print -quit 2>/dev/null || true)
python_version=
if [[ -n "${python_executable}" ]]; then
    python_version=$("${python_executable}" -c 'import platform; print(platform.python_version())')
fi
gpu_ok=false
[[ -n "${gpu_name}" && "${driver_version}" == 595.* ]] && gpu_ok=true || status=incomplete
uv_ok=false
[[ "${uv_version}" == "0.11.32" ]] && uv_ok=true || status=incomplete
python_ok=false
[[ "${python_version}" == "3.12.13" ]] && python_ok=true || status=incomplete

collaborator_group_present=false
getent group piper-collab >/dev/null && collaborator_group_present=true || status=incomplete

deployment_exists=false
[[ -d /opt/piper-outcome-stack ]] && deployment_exists=true || status=incomplete

sshd_policy_ok=false
if sshd_effective=$(/usr/sbin/sshd -T 2>/dev/null); then
    sshd_policy_ok=true
    for expected in \
        "permitrootlogin no" \
        "passwordauthentication no" \
        "kbdinteractiveauthentication no" \
        "authenticationmethods publickey" \
        "allowagentforwarding yes" \
        "allowtcpforwarding no" \
        "allowstreamlocalforwarding no" \
        "permittunnel no"; do
        grep -Fqx "${expected}" <<<"${sshd_effective}" || sshd_policy_ok=false
    done
fi
[[ "${sshd_policy_ok}" == "true" ]] || status=incomplete

ufw_policy_ok=false
ufw_status=$(ufw status verbose 2>/dev/null || true)
if grep -q '^Status: active' <<<"${ufw_status}" && \
    grep -q '^Default: deny (incoming), allow (outgoing)' <<<"${ufw_status}"; then
    ufw_added=$(ufw show added 2>/dev/null || true)
    unexpected_ufw=$(grep -E '^ufw allow' <<<"${ufw_added}" | \
        grep -Ev '^ufw allow in on tailscale0 to any port [0-9]+ proto tcp( comment .*)?$' || true)
    if grep -Eq '^ufw allow in on tailscale0 to any port [0-9]+ proto tcp' <<<"${ufw_added}" && \
        [[ -z "${unexpected_ufw}" ]]; then
        ufw_policy_ok=true
    fi
fi
[[ "${ufw_policy_ok}" == "true" ]] || status=incomplete

torch_json='{"installed":false,"version":"","build_cuda":"","cuda_available":false,"device_name":"","target_satisfied":false}'
deployment_python=/opt/piper-outcome-stack/current/.venv/bin/python
if [[ -x "${deployment_python}" ]]; then
    if torch_json=$("${deployment_python}" - <<'PIPER_LOCAL_TORCH_PROBE'
import json
import torch

cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else ""
version = torch.__version__
build_cuda = torch.version.cuda or ""
print(json.dumps({
    "installed": True,
    "version": version,
    "build_cuda": build_cuda,
    "cuda_available": cuda_available,
    "device_name": device_name,
    "target_satisfied": (
        version == "2.11.0+cu128"
        and build_cuda == "12.8"
        and cuda_available
        and "RTX 3060" in device_name
    ),
}, sort_keys=True))
PIPER_LOCAL_TORCH_PROBE
    ); then
        [[ "$(jq -r '.target_satisfied' <<<"${torch_json}")" == "true" ]] || \
            status=incomplete
    else
        status=incomplete
        torch_json='{"installed":true,"probe_failed":true,"target_satisfied":false}'
    fi
else
    status=incomplete
fi

can_not_checked='{"status":"not_checked","interface":"","expected_access":null,"bind_succeeded":null,"error":null,"effective_uid":null,"effective_capabilities_zero":null}'
not_checked_access='{"can":{"status":"not_checked","interface":"","expected_access":null,"bind_succeeded":null,"error":null,"effective_uid":null,"effective_capabilities_zero":null},"d435":{"status":"not_checked","devices":[]},"ar0234":{"status":"not_checked","devices":[]},"xbox":{"status":"not_checked","devices":[]}}'

probe_access_as() {
    local account=$1
    [[ -n "${account}" && "${account}" != "root" ]] || {
        printf '%s\n' "${not_checked_access}"
        return
    }
    id "${account}" >/dev/null 2>&1 || {
        printf '%s\n' "${not_checked_access}"
        return
    }
    [[ -x "${deployment_python}" ]] || {
        printf '%s\n' "${not_checked_access}"
        return
    }
    sudo -u "${account}" env PIPER_AR0234_DEVICE="${PIPER_AR0234_DEVICE:-}" \
        "${deployment_python}" - <<'PIPER_DEVICE_ACCESS_PROBE'
import json
import os
from pathlib import Path

def ordinary(paths):
    if not paths:
        return {"status": "not_checked", "devices": []}
    devices = [
        {"path": str(path), "readable": os.access(path, os.R_OK), "writable": os.access(path, os.W_OK)}
        for path in paths
    ]
    return {
        "status": "pass" if all(item["readable"] and item["writable"] for item in devices) else "fail",
        "devices": devices,
    }

video_root = Path("/dev/v4l/by-id")
videos = sorted(video_root.iterdir()) if video_root.is_dir() else []
d435 = [path for path in videos if "realsense" in path.name.lower() or "d435" in path.name.lower()]
ar0234 = [path for path in videos if "ar0234" in path.name.lower()]
if os.environ.get("PIPER_AR0234_DEVICE"):
    ar0234.append(Path(os.environ["PIPER_AR0234_DEVICE"]))
input_root = Path("/dev/input/by-id")
xbox = sorted(input_root.glob("*-event-joystick")) if input_root.is_dir() else []
print(json.dumps({"d435": ordinary(d435), "ar0234": ordinary(ar0234), "xbox": ordinary(xbox)}, sort_keys=True))
PIPER_DEVICE_ACCESS_PROBE
}

probe_can_bind_as() {
    local account=$1
    local context=$2
    local interface=$3
    local -a runner
    [[ -n "${account}" && "${account}" != root ]] || return 1
    id "${account}" >/dev/null 2>&1 || return 1
    [[ -x "${deployment_python}" ]] || return 1
    case "${context}" in
        namespace)
            [[ -x /usr/local/sbin/piper-socketcan ]] || return 1
            runner=(/usr/local/sbin/piper-socketcan exec -- env)
            ;;
        host)
            runner=(sudo -u "${account}" env)
            ;;
        *) return 1 ;;
    esac
    "${runner[@]}" PIPER_CAN_INTERFACE="${interface}" "${deployment_python}" - \
        <<'PIPER_CAN_BIND_PROBE'
import json
import os
import socket

interface = os.environ["PIPER_CAN_INTERFACE"]
bind_succeeded = False
error = None
try:
    handle = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        handle.bind((interface,))
        bind_succeeded = True
    finally:
        handle.close()
except (AttributeError, OSError) as exc:
    error = str(exc)

cap_eff = None
with open("/proc/self/status", encoding="utf-8") as status_file:
    for line in status_file:
        if line.startswith("CapEff:"):
            cap_eff = line.split()[1]
            break

print(json.dumps({
    "bind_succeeded": bind_succeeded,
    "effective_uid": os.geteuid(),
    "effective_capabilities_zero": cap_eff is not None and int(cap_eff, 16) == 0,
    "error": error,
}, sort_keys=True))
PIPER_CAN_BIND_PROBE
}

administrator=${SUDO_USER:-}
administrator_identified=false
[[ -n "${administrator}" && "${administrator}" != "root" ]] && \
    administrator_identified=true || true
administrator_access=$(probe_access_as "${administrator}")
collaborator=${PIPER_COLLABORATOR_ACCOUNT:-}
collaborator_identified=false
[[ -n "${collaborator}" && "${collaborator}" != "root" ]] && \
collaborator_identified=true || true
collaborator_access=$(probe_access_as "${collaborator}")

can_namespace=piper-can
target_can_interface=${PIPER_CAN_INTERFACE:-}
socketcan_status=not_checked
namespace_present=false
target_absent_from_host=false
target_present_in_namespace=false
collaborator_namespace_entry_succeeded=null
administrator_can=${can_not_checked}
collaborator_can=${can_not_checked}

if [[ -n "${target_can_interface}" ]]; then
    socketcan_status=fail
    if [[ "${target_can_interface}" =~ ^[[:alnum:]][[:alnum:]_.-]{0,14}$ ]]; then
        ip netns exec "${can_namespace}" true >/dev/null 2>&1 && \
            namespace_present=true || true
        ! ip link show dev "${target_can_interface}" >/dev/null 2>&1 && \
            target_absent_from_host=true || true
        ip -n "${can_namespace}" link show dev "${target_can_interface}" \
            >/dev/null 2>&1 && target_present_in_namespace=true || true

        administrator_probe=
        if administrator_probe=$(probe_can_bind_as \
            "${administrator}" namespace "${target_can_interface}"); then
            administrator_uid=$(id -u "${administrator}")
            administrator_can_status=fail
            jq -e --argjson expected_uid "${administrator_uid}" \
                '.bind_succeeded == true and
                 .effective_uid == $expected_uid and
                 .effective_capabilities_zero == true' \
                <<<"${administrator_probe}" >/dev/null && \
                administrator_can_status=pass || true
            administrator_can=$(jq -c \
                --arg status "${administrator_can_status}" \
                --arg interface "${target_can_interface}" \
                '. + {
                    status: $status,
                    interface: $interface,
                    expected_access: true
                }' <<<"${administrator_probe}")
        else
            administrator_can=$(jq -nc --arg interface "${target_can_interface}" \
                '{status:"fail", interface:$interface, expected_access:true,
                  bind_succeeded:null, error:"probe_failed", effective_uid:null,
                  effective_capabilities_zero:null}')
        fi

        collaborator_probe=
        if collaborator_probe=$(probe_can_bind_as \
            "${collaborator}" host "${target_can_interface}"); then
            collaborator_uid=$(id -u "${collaborator}")
            collaborator_namespace_entry_succeeded=false
            if sudo -u "${collaborator}" ip netns exec "${can_namespace}" true \
                >/dev/null 2>&1; then
                collaborator_namespace_entry_succeeded=true
            fi
            collaborator_can_status=fail
            jq -e --argjson expected_uid "${collaborator_uid}" \
                '.bind_succeeded == false and
                 .effective_uid == $expected_uid and
                 .effective_capabilities_zero == true' \
                <<<"${collaborator_probe}" >/dev/null && \
                [[ "${collaborator_namespace_entry_succeeded}" == false ]] && \
                collaborator_can_status=pass || true
            collaborator_can=$(jq -c \
                --arg status "${collaborator_can_status}" \
                --arg interface "${target_can_interface}" \
                --argjson namespace_entry_succeeded \
                    "${collaborator_namespace_entry_succeeded}" \
                '. + {
                    status: $status,
                    interface: $interface,
                    expected_access: false,
                    namespace_entry_succeeded: $namespace_entry_succeeded
                }' <<<"${collaborator_probe}")
        else
            collaborator_can=$(jq -nc --arg interface "${target_can_interface}" \
                '{status:"fail", interface:$interface, expected_access:false,
                  bind_succeeded:null, error:"probe_failed", effective_uid:null,
                  effective_capabilities_zero:null,
                  namespace_entry_succeeded:null}')
        fi

        if [[ "${namespace_present}" == true && \
            "${target_absent_from_host}" == true && \
            "${target_present_in_namespace}" == true && \
            "$(jq -r '.status' <<<"${administrator_can}")" == pass && \
            "$(jq -r '.status' <<<"${collaborator_can}")" == pass ]]; then
            socketcan_status=pass
        fi
    fi
    [[ "${socketcan_status}" == pass ]] || status=incomplete
fi

administrator_access=$(jq -c --argjson can "${administrator_can}" \
    '. + {can: $can}' <<<"${administrator_access}")
collaborator_access=$(jq -c --argjson can "${collaborator_can}" \
    '. + {can: $can}' <<<"${collaborator_access}")

jq -n \
    --arg schema_version "piper-local-environment-v3" \
    --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg status "${status}" \
    --argjson hostname_ok "${hostname_ok}" \
    --arg gpu_name "${gpu_name}" \
    --arg driver_version "${driver_version}" \
    --arg uv_version "${uv_version}" \
    --arg python_version "${python_version}" \
    --argjson gpu_ok "${gpu_ok}" \
    --argjson uv_ok "${uv_ok}" \
    --argjson python_ok "${python_ok}" \
    --argjson collaborator_group_present "${collaborator_group_present}" \
    --argjson deployment_exists "${deployment_exists}" \
    --argjson sshd_policy_ok "${sshd_policy_ok}" \
    --argjson ufw_policy_ok "${ufw_policy_ok}" \
    --argjson torch "${torch_json}" \
    --argjson administrator_identified "${administrator_identified}" \
    --argjson administrator_access "${administrator_access}" \
    --argjson collaborator_identified "${collaborator_identified}" \
    --argjson collaborator_access "${collaborator_access}" \
    --arg socketcan_status "${socketcan_status}" \
    --arg can_namespace "${can_namespace}" \
    --arg target_can_interface "${target_can_interface}" \
    --argjson namespace_present "${namespace_present}" \
    --argjson target_absent_from_host "${target_absent_from_host}" \
    --argjson target_present_in_namespace "${target_present_in_namespace}" \
    '{
        schema_version: $schema_version,
        generated_at_utc: $generated_at_utc,
        status: $status,
        host_identity: {hostname_matches_alias: $hostname_ok},
        gpu: {name: $gpu_name, driver_version: $driver_version, target_satisfied: $gpu_ok},
        runtime: {
            uv_version: $uv_version,
            uv_target_satisfied: $uv_ok,
            python_version: $python_version,
            python_target_satisfied: $python_ok,
            pytorch: $torch
        },
        network_security: {
            openssh_target_satisfied: $sshd_policy_ok,
            ufw_target_satisfied: $ufw_policy_ok
        },
        permissions: {
            socketcan_isolation: {
                status: $socketcan_status,
                namespace: $can_namespace,
                target_interface: $target_can_interface,
                namespace_present: $namespace_present,
                target_absent_from_host: $target_absent_from_host,
                target_present_in_namespace: $target_present_in_namespace
            },
            unique_administrator: {
                account_identified: $administrator_identified,
                highest_privilege: true,
                raw_hardware_authorized: true,
                enumerated_device_access: $administrator_access
            },
            collaborator: {
                account_identified: $collaborator_identified,
                sudo_authorized: false,
                raw_hardware_authorized: false,
                enumerated_device_access: $collaborator_access
            },
            collaborator_group_present: $collaborator_group_present,
            deployment_root_exists: $deployment_exists
        },
        hardware_available: false,
        hardware_tests_executed: false,
        motor_enable_executed: false,
        real_can_traffic_executed: false,
        hardware_verified: false
    }'
