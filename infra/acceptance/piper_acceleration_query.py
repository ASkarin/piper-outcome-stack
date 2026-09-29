"""Operator-run acceleration query; no motion, enable, recovery or parameter writes."""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from lerobot_robot_outcome_piper.sdk import create_piper


def query(arm, report):
    try:
        arm.connect()
        for joint in range(1, 7):
            started = time.monotonic()
            value = arm.get_joint_acc_limits(joint, timeout=1.0, min_interval=0.0)
            if arm.has_comm_error():
                raise RuntimeError(f"CAN error while querying J{joint}: {arm.get_comm_error()}")
            if value is None:
                raise RuntimeError(f"J{joint} acceleration query returned no response")
            acceleration = float(value.msg.max_joint_acc)
            if not math.isfinite(acceleration) or acceleration < 0:
                raise ValueError(f"J{joint} invalid reported acceleration: {acceleration}")
            report["joints"].append(
                {
                    "joint": joint,
                    "max_joint_acc_rad_s2": acceleration,
                    "sdk_timestamp_s": value.timestamp,
                    "query_started_s": started,
                    "query_ended_s": time.monotonic(),
                }
            )
        report["status"] = "read_complete"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        arm.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--firmware", required=True, choices=["default", "v183", "v188", "v189"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "started",
        "interface": args.interface,
        "firmware_driver": args.firmware,
        "scope": "one acceleration-limit query per joint; no parameter writes or motion",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "joints": [],
    }
    with args.output.open("x") as f:
        try:
            query(create_piper(args.interface, args.firmware), report)
        finally:
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, f, indent=2)
            f.write("\n")
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
