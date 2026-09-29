"""Bounded PiPER diagnostics. Standard library only; no control SDK is loaded.

Wire definitions: pyAgxArm@799b8412, Piper Codec 2A1/2A5-7/473/47C and
ArmMsgSearchMotorMaxAngleSpdAccLimit; gripper Codec 2A8; LowSpd 261-266.
"""

import fcntl
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time

MODES = ("link", "status", "firmware", "limits", "acceleration")
CONFIG = Path("/etc/piper-outcome-stack/query-access.json")
NAMESPACE = "/run/netns/piper-can"
IP = "/usr/sbin/ip"
HELPER = "/usr/local/libexec/piper-query.py"
SAFE_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"}


def requests(mode):
    if mode in ("link", "status"):
        return ()
    if mode == "firmware":
        return ((0x4AF, b"\x01"),)
    if mode == "limits":
        return tuple((0x472, bytes([i, 1, 0, 0, 0, 0, 0, 0])) for i in range(1, 7))
    if mode == "acceleration":
        return tuple((0x472, bytes([i, 2, 0, 0, 0, 0, 0, 0])) for i in range(1, 7))
    raise ValueError("unsupported operation")


def interface_info(interface, namespace=False):
    cmd = [IP]
    if namespace:
        cmd += ["-n", "piper-can"]
    values = json.loads(
        subprocess.check_output(
            cmd + ["-j", "-details", "-statistics", "link", "show", "dev", interface], text=True
        )
    )
    if len(values) != 1 or values[0].get("link_type") != "can":
        raise RuntimeError("configured interface is not a CAN device")
    return values[0]


def usb_identity(info):
    if info.get("parentbus") != "usb":
        raise RuntimeError("CAN USB parent is unavailable")
    parent = (Path("/sys/bus/usb/devices") / info["parentdev"]).resolve().parent
    return {key: (parent / key).read_text().strip() for key in ("idVendor", "idProduct", "serial")}


def raw_decode(frames):
    def data(cid):
        row = frames.get(cid)
        return None if row is None else bytes.fromhex(row["data"])

    state = {}
    d = data(0x2A1)
    if d is not None:
        state["controller"] = dict(
            zip(
                (
                    "ctrl_mode",
                    "arm_status",
                    "mode_feedback",
                    "teach_status",
                    "motion_status",
                    "trajectory_num",
                ),
                d[:6],
            )
        )
        state["controller"]["err_code"] = int.from_bytes(d[6:8], "big")
    joints = [data(cid) for cid in (0x2A5, 0x2A6, 0x2A7)]
    if all(d is not None for d in joints):
        state["joint_deg"] = [v / 1000 for d in joints for v in struct.unpack(">ii", d)]
    d = data(0x2A8)
    if d is not None:
        state["gripper"] = {
            "raw_value": int.from_bytes(d[:4], "big", signed=True),
            "mode_byte": d[7],
            "status_code": d[6],
            "enabled": bool(d[6] & 64),
        }
        if d[7] == 0:
            state["gripper"]["width_m"] = state["gripper"]["raw_value"] * 1e-6
    state["drivers"] = {}
    for joint in range(1, 7):
        d = data(0x260 + joint)
        if d is not None:
            state["drivers"][joint] = {
                "enabled": bool(d[5] & 64),
                "driver_error": bool(d[5] & 32),
                "status_byte": d[5],
                "voltage": int.from_bytes(d[:2], "big") / 10,
            }
    return state


def collect(sock, seconds, frames, counts, reply_id=None, joint=None):
    deadline = time.monotonic() + seconds
    replies = []
    while time.monotonic() < deadline:
        sock.settimeout(min(0.1, max(0.001, deadline - time.monotonic())))
        try:
            packet = sock.recv(16)
        except socket.timeout:
            continue
        cid, length, payload = struct.unpack("=IB3x8s", packet)
        # Only standard eight-byte feedback is decoded. Preserve counts otherwise.
        counts[cid] = counts.get(cid, 0) + 1
        if length != 8 or cid > 0x7FF:
            continue
        row = {"data": payload.hex(), "received_monotonic_s": time.monotonic()}
        frames[cid] = row
        if cid == reply_id and (joint is None or payload[0] == joint):
            replies.append(row)
            if reply_id in (0x473, 0x47C) or len(replies) == 11:
                break
    return replies


def query(mode, interface, sock, report):
    frames, counts = {}, {}
    report["started_monotonic_s"] = time.monotonic()
    report["sent_frames"] = []
    report["replies"] = []
    try:
        if mode == "status":
            collect(sock, 3, frames, counts)
        else:
            for cid, payload in requests(mode):
                event = {
                    "id": hex(cid),
                    "data": payload.hex(),
                    "started_monotonic_s": time.monotonic(),
                }
                report["sent_frames"].append(event)
                size = sock.send(struct.pack("=IB3x8s", cid, len(payload), payload))
                event["send_returned"] = size == 16  # Kernel acceptance, not arm acknowledgement.
                if size != 16:
                    raise RuntimeError("CAN send incomplete")
                rows = collect(
                    sock,
                    1,
                    frames,
                    counts,
                    0x4AF if mode == "firmware" else 0x47C if mode == "acceleration" else 0x473,
                    None if mode == "firmware" else payload[0],
                )
                report["replies"].append(rows)
                if len(rows) != (11 if mode == "firmware" else 1):
                    raise RuntimeError("query reply incomplete; no retry")
        report["status"] = "received" if counts else "no_feedback"
        if mode == "status":
            required = {0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A8, *range(0x261, 0x267)}
            report["missing_feedback_ids"] = [hex(i) for i in sorted(required - frames.keys())]
            if counts and report["missing_feedback_ids"]:
                report["status"] = "partial_feedback"
        if mode == "firmware":
            data = b"".join(bytes.fromhex(r["data"]) for r in report["replies"][0])
            if len(data) != 88 or not data.startswith(b"H-V"):
                raise RuntimeError("invalid firmware reply")
            report["firmware"] = {
                k: data[a:b].decode("ascii").rstrip("\0")
                for k, a, b in (
                    ("hardware_version", 0, 8),
                    ("motor_ratio_and_batch", 16, 18),
                    ("node_type", 32, 38),
                    ("software_version", 60, 68),
                    ("production_date", 68, 74),
                    ("node_number", 76, 78),
                )
            }
        elif mode == "acceleration":
            report["joint_acceleration_limits"] = []
            for rows in report["replies"]:
                d = bytes.fromhex(rows[0]["data"])
                report["joint_acceleration_limits"].append(
                    {
                        "joint": d[0],
                        "max_joint_acc_rad_s2": int.from_bytes(d[1:3], "big") / 100,
                    }
                )
        elif mode == "limits":
            report["joint_limits"] = []
            for rows in report["replies"]:
                d = bytes.fromhex(rows[0]["data"])
                report["joint_limits"].append(
                    {
                        "joint": d[0],
                        "max_deg": int.from_bytes(d[1:3], "big", signed=True) / 10,
                        "min_deg": int.from_bytes(d[3:5], "big", signed=True) / 10,
                        "max_speed_rad_s": int.from_bytes(d[5:7], "big") / 100,
                    }
                )
    finally:
        report["frame_counts"] = {hex(k): v for k, v in counts.items()}
        report["latest_frames"] = {hex(k): v for k, v in frames.items()}
        report["decoded_latest"] = raw_decode(frames)
        report["finished_monotonic_s"] = time.monotonic()


def worker(mode, cfg):
    if (
        os.geteuid() == 0
        or os.geteuid() != cfg["uid"]
        or not os.path.samefile("/proc/self/ns/net", NAMESPACE)
    ):
        raise RuntimeError("query worker must run as the configured administrator in piper-can")
    caps = next(
        line.split()[1]
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("CapEff:")
    )
    if int(caps, 16) != 0:
        raise RuntimeError("query worker retained capabilities")
    report = {
        "operation": mode,
        "interface": cfg["interface"],
        "uid": os.geteuid(),
        "effective_capabilities": caps,
        "scope": "diagnostic only; not motion or safety acceptance",
    }
    try:
        info = interface_info(cfg["interface"])
        report["link_before"] = info
        if usb_identity(info) != cfg["usb"]:
            raise RuntimeError("CAN adapter identity changed; administrator review required")
        if mode == "link":
            report["status"] = "read_complete"
        else:
            if (
                "UP" not in info["flags"]
                or info["linkinfo"]["info_data"].get("bittiming", {}).get("bitrate") != 1000000
            ):
                raise RuntimeError(
                    "CAN must already be UP at its verified bitrate; no automatic configuration"
                )
            with socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as sock:
                sock.bind((cfg["interface"],))
                query(mode, cfg["interface"], sock, report)
            report["link_after"] = interface_info(cfg["interface"])
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, allow_nan=False, indent=2))
    return 0 if report["status"] not in ("failed", "no_feedback", "partial_feedback") else 1


def main(args=None):
    args = sys.argv[1:] if args is None else args
    if len(args) == 2 and args[1] == "--worker" and os.geteuid() != 0 and args[0] in MODES:
        return worker(args[0], json.load(sys.stdin))
    if len(args) != 1 or args[0] not in MODES:
        raise ValueError(
            "allowed operations: link | status | firmware | limits | acceleration (no extra arguments)"
        )
    if os.geteuid() != 0:
        raise PermissionError("use the installed sudo -n piper-query entry")
    mode = args[0]
    with CONFIG.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = json.load(lock)
        if (
            os.environ.get("SUDO_UID") != str(cfg["uid"])
            or os.environ.get("SUDO_USER") != cfg["administrator"]
        ):
            raise PermissionError("only the configured administrator may use this entry")
        if (
            mode in ("firmware", "limits", "acceleration")
            and subprocess.check_output([IP, "netns", "pids", "piper-can"], text=True).strip()
        ):
            raise RuntimeError(
                "CAN namespace has an active session; only passive status/link inspection is allowed"
            )
        cmd = [
            "/usr/bin/nsenter",
            "--net=" + NAMESPACE,
            "--",
            "/usr/bin/setpriv",
            "--reuid=" + str(cfg["uid"]),
            "--regid=" + str(cfg["gid"]),
            "--clear-groups",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--bounding-set=-all",
            "--no-new-privs",
            "/usr/bin/python3",
            "-I",
            "-S",
            HELPER,
            mode,
            "--worker",
        ]
        result = subprocess.run(cmd, input=json.dumps(cfg), text=True, env=SAFE_ENV, timeout=20)
        return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}),
            file=sys.stderr,
        )
        raise SystemExit(1)
