from pathlib import Path
import os
import json
import time
import math
import copy
import sys
from datetime import UTC, datetime
from lerobot_robot_outcome_piper.sdk import create_piper
from lerobot_robot_outcome_piper.timing import FeedbackReceiver

config = json.loads(Path(sys.argv[1]).read_text())["robot"]
root = Path(sys.argv[2])
root.mkdir()
if os.geteuid() == 0 or not os.path.samefile("/proc/self/ns/net", "/run/netns/piper-can"):
    raise SystemExit("Use piper-socketcan exec as the ordinary administrator")
if not os.isatty(0):
    raise SystemExit("Interactive operator terminal required")
r = {
    "scope": "one operator-confirmed electronic-stop recovery; other controller/driver faults rejected",
    "started_at_utc": datetime.now(UTC).isoformat(),
    "status": "started",
    "enable_command_sent": False,
    "position_target_sent": False,
    "resume_command_sent": False,
    "script_source": Path(__file__).read_text(),
}
arm = None
with (root / "result.json").open("x") as out:
    try:
        arm = create_piper(config["can_interface"], config["firmware"])
        arm.connect()
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        rx = FeedbackReceiver(arm, gripper, time.monotonic)
        if not rx.wait_ready(1.0):
            raise RuntimeError("Initial feedback is incomplete")

        def read():
            if arm.has_comm_error():
                raise RuntimeError(arm.get_comm_error())
            with rx.condition:
                f = rx.snapshot()
                states = [copy.deepcopy(arm.get_driver_states(i)) for i in range(1, 7)]
                stamps = [rx.received.get(i) for i in rx.DRIVER_IDS]
            now = time.monotonic()
            if any(s is None for s in states) or any(
                t is None or not 0 <= now - t <= 0.2 for t in (*f.received_s, *stamps)
            ):
                raise RuntimeError("Driver feedback missing or stale")
            if f.status.msg.err_code != 0 or any(
                s.msg.foc_status.driver_error_status for s in states
            ):
                raise RuntimeError(
                    "A fault other than the electronic emergency stop requires inspection"
                )
            return {
                "arm_status": int(f.status.msg.arm_status),
                "ctrl_mode": int(f.status.msg.ctrl_mode),
                "joint_deg": [math.degrees(v) for v in f.joints.msg],
                "enabled": [bool(s.msg.foc_status.driver_enable_status) for s in states],
                "status_received_s": f.received_s[3],
            }

        r["before"] = read()
        print(json.dumps(r["before"], indent=2), flush=True)
        if r["before"]["arm_status"] == 1:
            if input(
                "Press Enter to resume this electronic emergency stop once; other text cancels: "
            ).strip():
                raise RuntimeError("Operator cancelled")
            check = read()
            if check["arm_status"] != 1:
                raise RuntimeError("State changed while awaiting operator")
            sent = time.monotonic()
            r["resume_command_sent"] = True
            arm.reset()
            deadline = sent + 3.0
            while time.monotonic() < deadline:
                r["after"] = read()
                if r["after"]["status_received_s"] >= sent and r["after"]["arm_status"] == 0:
                    break
                if r["after"]["arm_status"] not in (0, 1):
                    raise RuntimeError("Unexpected state after resume")
                time.sleep(0.02)
            else:
                raise RuntimeError("Resume not confirmed; no resend")
            time.sleep(0.3)
            r["after"] = read()
            if r["after"]["arm_status"] != 0:
                raise RuntimeError("Normal state did not persist")
        elif r["before"]["arm_status"] == 0:
            r["after"] = r["before"]
        else:
            raise RuntimeError("Expected electronic emergency stop or already-normal state")
        r["status"] = "normal_feedback_confirmed"
    except (Exception, KeyboardInterrupt) as exc:
        r["status"] = "failed"
        r["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception as exc:
                r.update(status="failed", disconnect_error=str(exc))
        r["finished_at_utc"] = datetime.now(UTC).isoformat()
        json.dump(r, out, indent=2)
print(json.dumps({k: v for k, v in r.items() if k != "script_source"}, indent=2))
raise SystemExit(0 if r["status"] == "normal_feedback_confirmed" else 1)
