"""One-shot PiPER firmware and feedback inspection; never enables or moves the arm."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


SDK_COMMIT = "799b8412fbe8b9156bc9892d3dbeb2df7e98be71"


def firmware_driver(identity: dict) -> str:
    if identity.get("node_type") != "ARM_MC":
        raise ValueError("firmware response is not an ARM_MC controller")
    match = re.fullmatch(r"S-V(\d+)\.(\d+)-(\d+)", identity.get("software_version", ""))
    if match is None:
        raise ValueError(f"unrecognized software version: {identity!r}")
    version = tuple(map(int, match.groups()))
    if version <= (1, 8, 2):
        return "default"
    if version <= (1, 8, 7):
        return "v183"
    if version == (1, 8, 8):
        return "v188"
    return "v189"


def check_communication(arm) -> None:
    if arm.has_comm_error():
        raise RuntimeError(f"CAN communication error: {arm.get_comm_error()}")


def frame(message, payload):
    if message is None:
        return None
    return {
        "timestamp_s": float(message.timestamp),
        "hz": float(message.hz),
        "value": payload(message.msg),
    }


def snapshot(arm, gripper) -> dict:
    check_communication(arm)
    status = frame(
        arm.get_arm_status(),
        lambda msg: {
            key: int(getattr(msg, key))
            for key in ("ctrl_mode", "arm_status", "mode_feedback", "err_code")
        },
    )
    drivers = [
        frame(
            arm.get_driver_states(index),
            lambda msg: {
                "enabled": bool(msg.foc_status.driver_enable_status),
                "driver_error": bool(msg.foc_status.driver_error_status),
                "voltage": float(msg.vol),
                "driver_temperature_c": float(msg.foc_temp),
                "motor_temperature_c": float(msg.motor_temp),
            },
        )
        for index in range(1, 7)
    ]
    sample = {
        "joints_rad": frame(arm.get_joint_angles(), lambda msg: list(map(float, msg))),
        "joint_group_timestamps_s": [
            None if (item := getattr(arm._parser, name, None)) is None else float(item.timestamp)
            for name in ("joint_12", "joint_34", "joint_56")
        ],
        "status": status,
        "drivers": drivers,
        "gripper": frame(
            gripper.get_gripper_status(),
            lambda msg: {
                "value": float(msg.value),
                "mode": str(msg.mode),
                "status_code": int(msg.status_code),
                "enabled": bool(msg.foc_status.driver_enable_status),
            },
        ),
    }
    sample["host_monotonic_s"] = time.monotonic()
    sample["host_wall_s"] = time.time()
    return sample


def validate_snapshot(sample: dict, received_after_s: float) -> None:
    messages = [sample["joints_rad"], sample["status"], sample["gripper"], *sample["drivers"]]
    if any(item is None for item in messages):
        raise RuntimeError("incomplete feedback; missing data is not a disabled-motor result")
    timestamps = [item["timestamp_s"] for item in messages] + sample["joint_group_timestamps_s"]
    if any(
        value is None or not math.isfinite(value) or value < received_after_s
        for value in timestamps
    ):
        raise RuntimeError("feedback did not refresh during the diagnostic window")
    if any(value > sample["host_wall_s"] for value in timestamps):
        raise RuntimeError("feedback timestamp is in the future")
    values = sample["joints_rad"]["value"]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("joint feedback must contain six finite positions")
    gripper = sample["gripper"]["value"]
    if gripper["mode"] != "width" or not math.isfinite(gripper["value"]):
        raise RuntimeError("gripper feedback is not a finite width in metres")
    if gripper["status_code"] & 0x3F:
        raise RuntimeError("gripper reports a fault")
    if gripper["enabled"] or any(item["value"]["enabled"] for item in sample["drivers"]):
        raise RuntimeError("a motor was already enabled; no disable or stop command was sent")
    status = sample["status"]["value"]
    if status["arm_status"] != 0 or status["err_code"] != 0:
        raise RuntimeError(f"controller reports a non-normal status: {status}")
    if any(item["value"]["driver_error"] for item in sample["drivers"]):
        raise RuntimeError("a driver reports an error")


def probe(factory, interface: str, report: dict, duration_s: float = 3.0) -> None:
    # All pinned PiPER driver versions inherit this same firmware query. The
    # base profile is used only to read identity, never as a guessed motion driver.
    discovery = factory(interface, "default")
    try:
        discovery.connect()
        # The SDK discards firmware segments while its receive FPS is still zero.
        # Prefer a warmed-up receiver, but a quiet bus still needs its first query.
        reception_deadline = time.monotonic() + 2.0
        while True:
            passive_feedback = discovery.get_fps() > 0
            check_communication(discovery)
            if passive_feedback or time.monotonic() >= reception_deadline:
                break
            time.sleep(0.05)
        report["passive_feedback_before_query"] = passive_feedback
        identity = discovery.get_firmware(timeout=1.0, min_interval=0.0)
        check_communication(discovery)
        if not isinstance(identity, dict):
            raise RuntimeError("firmware query returned no identity; no retry performed")
        report["firmware_identity"] = identity
    finally:
        discovery.disconnect()
    driver = firmware_driver(identity)
    report["selected_driver"] = driver
    arm = factory(interface, driver)
    samples = report["samples"] = []
    try:
        arm.connect()
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            sample = snapshot(arm, gripper)
            samples.append(sample)
            if any(item is not None and item["value"]["enabled"] for item in sample["drivers"]):
                raise RuntimeError("a motor was already enabled; ending read-only inspection")
            if sample["gripper"] is not None and sample["gripper"]["value"]["enabled"]:
                raise RuntimeError("gripper was already enabled; ending read-only inspection")
            time.sleep(0.1)
        check_communication(arm)
        if not samples:
            raise RuntimeError("no feedback samples collected")
        # Require each feedback group to update in the latter half of this
        # diagnostic window. This is not a frozen motion-safety timeout.
        validate_snapshot(samples[-1], samples[len(samples) // 2]["host_wall_s"])
        report["status"] = "feedback_received_motors_disabled"
    finally:
        arm.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error("run through piper-socketcan exec as the administrator's ordinary UID")
    if not os.path.samefile("/proc/self/ns/net", "/run/netns/piper-can"):
        parser.error("run inside the existing piper-can namespace")
    distribution = importlib.metadata.distribution("pyAgxArm")
    source = json.loads(distribution.read_text("direct_url.json"))
    if source["vcs_info"]["commit_id"] != SDK_COMMIT:
        parser.error("the active SDK does not match the project's pinned commit")
    report = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "interface": args.interface,
        "sdk_commit": SDK_COMMIT,
        "probe_script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "diagnostic_window_s": 3.0,
        "status": "started",
        "scope": "initial SDK inspection, not five-cycle plugin or motion acceptance",
        "motor_enable_executed": False,
        "motion_executed": False,
    }
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    def factory(interface, driver):
        arm = AgxArmFactory.create_arm(
            create_agx_arm_config(robot="piper", firmeware_version=driver, channel=interface)
        )
        if "firmware_query_frame" not in report:
            from pyAgxArm.protocols.can_protocol.msgs.piper.default import ArmMsgReqFirmware

            query = arm._parser.pack(ArmMsgReqFirmware())
            report["firmware_query_frame"] = {
                "can_id": query.arbitration_id,
                "data_hex": bytes(query.data).hex(),
            }
            replies = report["firmware_reply_frames"] = []

            def record_reply(packet):
                if packet.arbitration_id == query.arbitration_id:
                    replies.append(
                        {"timestamp_s": packet.timestamp, "data_hex": bytes(packet.data).hex()}
                    )

            arm.get_context().register_parser_packet_fun(record_reply)
        return arm

    # Create output exclusively so a later inspection cannot overwrite this result.
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            probe(factory, args.interface, report)
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    return 0 if report["status"] == "feedback_received_motors_disabled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
