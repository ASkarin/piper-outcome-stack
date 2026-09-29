"""One explicit operator-requested Follower role setup with before/after feedback.

Uses the pinned official SDK. This changes controller configuration; it is not a
read-only query and must not be invoked through piper-query or automatic reconnect.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone


def feedback_snapshot(receiver):
    with receiver.condition:
        result = {"received_ids": [hex(i) for i in sorted(receiver.received)]}
        if all(i in receiver.received for i in receiver.IDS):
            frame = receiver.snapshot()
            result.update(
                joint_deg=[math.degrees(float(v)) for v in frame.joints.msg],
                controller=str(frame.status.msg),
                gripper=str(frame.gripper.msg),
                received_monotonic_s=list(frame.received_s),
            )
            result["joint_enabled"] = [
                None if state is None else bool(state.msg.foc_status.driver_enable_status)
                for state, stamp in receiver.driver_states()
            ]
        return result


def configure_once(arm, receiver, report, *, confirm=input, snapshot=feedback_snapshot):
    report["follower_command_sent"] = False
    ready = receiver.wait_ready(3.0)
    report["before"] = snapshot(receiver)
    if ready:
        report["status"] = "feedback_already_present_no_change"
        return
    print(json.dumps(report["before"], ensure_ascii=False, indent=2), flush=True)
    if confirm("保持空载低位停放。Enter仅设置一次Follower角色；其他输入取消：").strip():
        report["status"] = "cancelled"
        return
    if arm.has_comm_error():
        raise RuntimeError("CAN error before role setup")
    # Feedback may have started while the operator was reading the prompt.
    if receiver.wait_ready(0.0):
        report["status"] = "feedback_started_before_command_no_change"
        report["after"] = snapshot(receiver)
        return
    report["requested_monotonic_s"] = time.monotonic()
    report["follower_command_attempted"] = True
    arm.set_follower_mode()
    report["follower_command_sent"] = True
    report["returned_monotonic_s"] = time.monotonic()
    if arm.has_comm_error():
        raise RuntimeError("CAN error after single Follower request; no retry")
    ready = receiver.wait_ready(3.0)
    report["after"] = snapshot(receiver)
    report["status"] = "feedback_restored" if ready else "feedback_incomplete_after_single_request"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--firmware", required=True, choices=["v189"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.geteuid() == 0 or not os.path.samefile("/proc/self/ns/net", "/run/netns/piper-can"):
        parser.error("use the administrator piper-socketcan exec launcher")
    if not sys.stdin.isatty():
        parser.error("interactive on-site terminal required")
    report = dict(
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        status="started",
        scope="single controller role setup; no enable, disable, mode-speed, reset or position commands",
        interface=args.interface,
        firmware_driver=args.firmware,
        script_source=Path(__file__).read_text(),
        python=sys.executable,
    )
    arm = None
    with args.output.open("x") as output:
        try:
            import importlib.metadata
            from lerobot_robot_outcome_piper.sdk import create_piper
            from lerobot_robot_outcome_piper.timing import FeedbackReceiver
            from piper_joint_commission import trace_transmissions

            origin = json.loads(
                importlib.metadata.distribution("pyAgxArm").read_text("direct_url.json")
            )
            report["sdk_commit"] = origin["vcs_info"]["commit_id"]
            if report["sdk_commit"] != "799b8412fbe8b9156bc9892d3dbeb2df7e98be71":
                raise RuntimeError("SDK differs from inspected version")
            arm = create_piper(args.interface, args.firmware)
            arm.connect()
            trace_transmissions(arm.get_context().get_comm(), report)
            gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            receiver = FeedbackReceiver(arm, gripper, time.monotonic)
            configure_once(arm, receiver, report)
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if arm is not None:
                try:
                    arm.disconnect()
                except Exception as exc:
                    report.update(status="failed", disconnect_error=str(exc))
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "script_source"}, indent=2, ensure_ascii=False
        )
    )
    return (
        0
        if report["status"]
        in (
            "feedback_restored",
            "feedback_already_present_no_change",
            "feedback_started_before_command_no_change",
            "cancelled",
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
