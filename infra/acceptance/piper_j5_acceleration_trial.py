"""Explicit operator application/restoration of the measured J5 acceleration trial."""

import argparse
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from lerobot_robot_outcome_piper.sdk import create_piper


def read(arm):
    value = arm.get_joint_acc_limits(5, timeout=1.0, min_interval=0.0)
    if arm.has_comm_error():
        raise RuntimeError(f"CAN error: {arm.get_comm_error()}")
    if value is None:
        raise RuntimeError("J5 acceleration query returned no response")
    return float(value.msg.max_joint_acc)


def apply(arm, expected, target, report):
    try:
        arm.connect()
        before = read(arm)
        report["before_rad_s2"] = before
        if not math.isfinite(before) or abs(before - expected) > 1e-6:
            raise ValueError(f"J5实际值为{before}，与预期{expected}不一致；未写入")
        report["write_api_called"] = True
        ok = arm.set_joint_acc_limits(5, max_joint_acc=target, timeout=1.0)
        report["sdk_confirmed"] = bool(ok)
        after = read(arm)
        report["after_rad_s2"] = after
        if not ok or not math.isfinite(after) or abs(after - target) > 1e-6:
            raise RuntimeError(f"设置未完成确认：SDK={ok}，读回={after}；不重复写入或自动恢复")
        report["status"] = "readback_confirmed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        arm.disconnect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=["apply", "restore"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    expected, target = (5.0, 2.5) if a.operation == "apply" else (2.5, 5.0)
    print(
        f"仅修改J5最大加速度：{expected} → {target} rad/s²；不使能、不运动、不修改速度。",
        flush=True,
    )
    if input("请保持机械臂静止。确认上述数值后按Enter；其他文字取消："):
        return
    report = {
        "status": "started",
        "operation": a.operation,
        "joint": 5,
        "expected_rad_s2": expected,
        "requested_rad_s2": target,
        "write_api_called": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with a.output.open("x") as f:
        try:
            apply(create_piper("can0", "v189"), expected, target, report)
        finally:
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, f, indent=2)
            f.write("\n")
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
