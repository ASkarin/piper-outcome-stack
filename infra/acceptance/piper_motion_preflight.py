"""Read present pose, enable state and controller limits without issuing motion commands."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from piper_read_only_probe import snapshot


def inspect_limits(robot, report):
    try:
        robot.connect()
        report["firmware_identity"] = robot.firmware_identity
        report["initial_observation"] = robot.get_observation()
        report["initial_state"] = snapshot(robot._arm, robot._gripper)
        report["joint_limits"] = []
        for index in range(1, 7):
            reply = robot._arm.get_joint_angle_vel_limits(index, timeout=1.0, min_interval=0.0)
            if robot._arm.has_comm_error():
                raise RuntimeError(
                    f"CAN error querying joint {index}: {robot._arm.get_comm_error()}"
                )
            if reply is None:
                raise RuntimeError(f"joint {index} limit query returned no response; no retry")
            values = {
                key: getattr(reply.msg, key)
                for key in ("min_angle_limit", "max_angle_limit", "max_joint_spd")
            }
            report["joint_limits"].append(
                {"joint": index, "controller_values": values, "sdk_timestamp_s": reply.timestamp}
            )
            if any(value is None or not math.isfinite(float(value)) for value in values.values()):
                raise RuntimeError(f"joint {index} limits are incomplete or invalid")
            if (
                values["min_angle_limit"] >= values["max_angle_limit"]
                or values["max_joint_spd"] <= 0
            ):
                raise RuntimeError(f"joint {index} controller limits cannot define a motion range")
        report["final_observation"] = robot.get_observation()
        report["feedback"] = asdict(robot.last_feedback_telemetry)
        report["final_state"] = snapshot(robot._arm, robot._gripper)
        report["status"] = "read_complete"
    finally:
        robot.disconnect()
        report["connected_after_disconnect"] = robot.is_connected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--firmware", required=True, choices=("default", "v183", "v188", "v189"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.geteuid() == 0 or not os.path.samefile("/proc/self/ns/net", "/run/netns/piper-can"):
        parser.error("run as the ordinary administrator through piper-socketcan exec")
    report = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "started",
        "scope": "post-GUI pose and controller-limit readout; not motion acceptance",
        "interface": args.interface,
        "python": sys.executable,
        "script": str(Path(__file__).resolve()),
        "script_source": Path(__file__).read_text(encoding="utf-8"),
        "firmware_driver": args.firmware,
        "motor_enable_executed": False,
        "motion_executed": False,
        "zero_reference_written": False,
        "controller_limits_written": False,
    }
    with args.output.open("x", encoding="utf-8") as out:
        try:
            from lerobot_robot_outcome_piper.config import OutcomePiperConfig
            from lerobot_robot_outcome_piper.robot import OutcomePiper

            config = OutcomePiperConfig(
                can_interface=args.interface,
                firmware=args.firmware,
                feedback_timeout_s=1.0,
                execution_mode="read_only",
                cameras={},
                calibration_dir=args.output.parent / "plugin-calibration",
            )
            inspect_limits(OutcomePiper(config), report)
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "read_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
