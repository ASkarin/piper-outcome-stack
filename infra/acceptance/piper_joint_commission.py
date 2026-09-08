"""One operator-started joint target; leave motors enabled after the requested move."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

STEP = math.radians(5)
TOLERANCE = math.radians(0.1)
FEEDBACK_TIMEOUT = 0.2
WAYPOINT_TIMEOUT = 10.0
SPEED_PERCENT = 1


def read_flange_result(arm, receiver):
    """Read controller flange feedback for endpoint reporting, not independent metrology."""
    with receiver.condition:
        message = copy.deepcopy(arm.get_flange_pose())
    if message is None:
        raise ValueError("controller flange feedback is unavailable")
    pose = list(map(float, message.msg))
    if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise ValueError("controller flange feedback is invalid")
    return {
        "reference_point": "flange",
        "xyz_mm": [v * 1000 for v in pose[:3]],
        "rpy_deg": [math.degrees(v) for v in pose[3:]],
        "sdk_timestamp_s": float(message.timestamp),
        "host_read_monotonic_s": time.monotonic(),
    }


def trace_transmissions(comm, report, clock=time.monotonic):
    """Record this session's actual send calls without changing or resending frames."""
    original_send = comm.send
    report["tx_frames"] = []

    def send(frame):
        event = {
            "can_id": frame.arbitration_id,
            "data_hex": bytes(frame.data).hex(),
            "started_monotonic_s": clock(),
        }
        report["tx_frames"].append(event)
        result = original_send(frame)
        event["returned_monotonic_s"] = clock()
        return result

    comm.send = send


def joint_waypoints(start, goal, limits, max_step=STEP):
    """Move toward one explicit target, checking every waypoint before sending."""
    if (
        len(start) != 6
        or len(goal) != 6
        or len(limits) != 6
        or not all(math.isfinite(v) for v in (*start, *goal))
    ):
        raise ValueError("six finite start/target angles and six limits are required")
    if any(not math.isfinite(v) for pair in limits for v in pair):
        raise ValueError("invalid limits")
    if any(not lo <= v <= hi or lo >= hi for v, (lo, hi) in zip(goal, limits)):
        raise ValueError("target is outside controller limits")
    if not math.isfinite(max_step) or not 2 * TOLERANCE < max_step <= math.radians(15):
        raise ValueError("step must exceed both endpoint tolerances and be at most 15 degrees")
    points, q = [], list(start)
    while q != list(goal):
        # Reserve arrival tolerance at both ends: the previous position may
        # undershoot and the new position may overshoot its nominal target.
        stride = max_step - 2 * TOLERANCE
        q = [
            g if abs(g - v) <= stride else v + math.copysign(stride, g - v) for v, g in zip(q, goal)
        ]
        if any(not lo <= v <= hi for v, (lo, hi) in zip(q, limits)):
            raise ValueError("cannot enter legal target range within the approved step")
        points.append(q)
    return points or [list(goal)]


class JointRun:
    def __init__(
        self,
        arm,
        receiver,
        limits,
        report,
        goal,
        clock=time.monotonic,
        sleep=time.sleep,
        max_step=STEP,
        speed_percent=SPEED_PERCENT,
        via_zero=False,
        return_zero=False,
        gripper=None,
        gripper_target=None,
    ):
        self.arm, self.rx, self.limits, self.report = arm, receiver, limits, report
        self.goal = list(goal)
        if not isinstance(speed_percent, int) or not 1 <= speed_percent <= 5:
            raise ValueError("commissioning speed must be an integer from 1 to 5 percent")
        self.step, self.speed, self.via_zero = max_step, speed_percent, via_zero
        self.return_zero = return_zero
        self.gripper, self.gripper_target = gripper, gripper_target
        self.gripper_hold_width = None
        self.phase = "initial"
        self.clock, self.sleep = clock, sleep
        self.mode_required = False
        self.active = False
        self.report["samples"], self.report["commands"] = [], []

    def read(self, require_enabled=False, for_hold=False):
        if self.arm.has_comm_error():
            raise RuntimeError(f"CAN error: {self.arm.get_comm_error()}")
        with self.rx.condition:
            f = self.rx.snapshot()
            states = [copy.deepcopy(self.arm.get_driver_states(i)) for i in range(1, 7)]
            stamps = [self.rx.received.get(i) for i in self.rx.DRIVER_IDS]
        now = self.clock()
        if any(s is None for s in states) or any(t is None for t in stamps):
            raise RuntimeError("missing driver feedback")
        if any(
            not math.isfinite(t) or not 0 <= now - t <= FEEDBACK_TIMEOUT
            for t in (*f.received_s, *stamps)
        ):
            raise RuntimeError("stale or invalid feedback")
        status = f.status.msg
        self.report["last_controller_status"] = str(status)
        if status.arm_status != 0 or status.err_code != 0:
            raise RuntimeError(f"controller fault: {status}")
        if status.ctrl_mode not in (0, 1):
            raise RuntimeError("controller is not in standby or CAN mode")
        if self.mode_required and (status.ctrl_mode != 1 or status.mode_feedback != 1):
            raise RuntimeError("controller left CAN/J mode")
        flags = [bool(s.msg.foc_status.driver_enable_status) for s in states]
        if any(s.msg.foc_status.driver_error_status for s in states):
            raise RuntimeError("motor driver fault")
        if require_enabled and not all(flags):
            raise RuntimeError("a joint lost enable")
        g = f.gripper.msg
        if not for_hold and (g.mode != "width" or g.status_code & 0x3F):
            raise RuntimeError("gripper width feedback unavailable or gripper reports a fault")
        if not for_hold and g.foc_status.driver_enable_status and self.gripper_target is None:
            raise RuntimeError("enabled gripper requires an explicit opening target")
        if (
            require_enabled
            and self.gripper_hold_width is not None
            and (
                not g.foc_status.driver_enable_status
                or abs(g.value - self.gripper_hold_width) > 0.0005
            )
        ):
            raise RuntimeError("gripper did not retain the requested opening")
        q = list(map(float, f.joints.msg))
        if len(q) != 6 or not all(math.isfinite(v) for v in q):
            raise RuntimeError("invalid joint angles")
        self.report["samples"].append(
            {
                "monotonic_s": now,
                "phase": self.phase,
                "gripper_width_m": float(g.value) if g.mode == "width" else None,
                "gripper_mode": g.mode,
                "gripper_enabled": bool(g.foc_status.driver_enable_status),
                "joint_rad": q,
                "enabled": flags,
                "status": {
                    key: int(getattr(status, key))
                    for key in (
                        "ctrl_mode",
                        "arm_status",
                        "mode_feedback",
                        "teach_status",
                        "motion_status",
                        "trajectory_num",
                        "err_code",
                    )
                },
            }
        )
        return q, flags, f, stamps

    def send(self, name, fn, *args):
        event = {"name": name, "args": args, "started_monotonic_s": self.clock()}
        self.report["commands"].append(event)
        fn(*args)
        event["returned_monotonic_s"] = self.clock()
        if self.arm.has_comm_error():
            raise RuntimeError(f"CAN error after {name}")

    def wait(self, predicate, timeout=WAYPOINT_TIMEOUT):
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            sample = self.read(require_enabled=self.active)
            if predicate(*sample):
                return sample
            self.sleep(0.01)
        raise RuntimeError("feedback confirmation timed out; no resend or next waypoint")

    def wait_stable(self, requested, segment_start, goal=None):
        """Allow settling within the original deadline, without resending the target."""
        goal = self.goal if goal is None else goal
        stable_since = None
        while self.clock() < requested + WAYPOINT_TIMEOUT:
            q, _, frame, _ = self.read(True)
            if any(abs(v - a) > self.step for v, a in zip(q, segment_start)):
                raise RuntimeError(
                    f"measured joint excursion exceeded {math.degrees(self.step):g} degrees"
                )
            within = all(abs(v - g) <= TOLERANCE for v, g in zip(q, goal))
            if within and frame.status.msg.motion_status == 0:
                if stable_since is None:
                    stable_since = self.clock()
                if self.clock() - stable_since >= 0.3:
                    return
            else:
                stable_since = None
            self.sleep(0.01)
        raise RuntimeError("target did not stabilize within the original waypoint deadline")

    def position_gripper(self, width, arm_goal):
        if self.gripper is None:
            raise ValueError("gripper instance is required")
        self.phase = "gripper_close" if width == 0 else "gripper_open"
        self.gripper_hold_width = None
        requested = self.clock()
        self.send("set_gripper_width", self.gripper.move_gripper_m, width, 1.0)
        stable_since = None

        def opened(q, flags, frame, stamps):
            nonlocal stable_since
            g = frame.gripper.msg
            ready = (
                frame.received_s[4] >= requested
                and g.foc_status.driver_enable_status
                and abs(g.value - width) <= 0.0005
                and all(abs(v - target) <= TOLERANCE for v, target in zip(q, arm_goal))
                and frame.status.msg.motion_status == 0
            )
            if not ready:
                stable_since = None
                return False
            if stable_since is None:
                stable_since = self.clock()
            return self.clock() - stable_since >= 0.3

        self.wait(opened)
        self.gripper_hold_width = width
        key = "gripper_closed_confirmed" if width == 0 else "gripper_open_confirmed"
        self.report[key] = True
        self.report["gripper_final_target_mm"] = width * 1000

    def run(self, confirm=input):
        start, flags, frame, _ = self.read()
        initially_enabled = list(flags)
        warm = all(flags) and (
            frame.status.msg.ctrl_mode == 0
            or (frame.status.msg.ctrl_mode == 1 and frame.status.msg.mode_feedback == 1)
        )
        cold = not any(flags) and frame.status.msg.ctrl_mode == 0
        if not (warm or cold):
            raise RuntimeError("expected standby or enabled CAN/J; partial enable is not supported")
        self.active = warm
        self.mode_required = warm and frame.status.msg.ctrl_mode == 1
        goals = [[0.0] * 6, self.goal] if self.via_zero else [self.goal]
        if self.return_zero:
            goals.append([0.0] * 6)
        stage_points = []
        previous = start
        for goal in goals:
            stage_points.append(joint_waypoints(previous, goal, self.limits, max_step=self.step))
            previous = goal
        if self.gripper_target is not None:
            if (
                self.gripper is None
                or not math.isfinite(self.gripper_target)
                or self.gripper_target < 0
            ):
                raise ValueError("invalid gripper target")
            param = self.gripper.get_gripper_teaching_pendant_param(timeout=1.0, min_interval=0.0)
            if self.arm.has_comm_error() or param is None:
                raise RuntimeError("gripper range query failed; no motion started")
            maximum = float(param.msg.max_range_config)
            self.report["gripper_max_range_m"] = maximum
            if not math.isfinite(maximum) or not 0 <= self.gripper_target <= maximum:
                raise ValueError("requested gripper opening exceeds reported range")
        self.report["plan"] = {
            "start_deg": list(map(math.degrees, start)),
            "target_deg": list(map(math.degrees, self.goal)),
            "stage_targets_deg": [list(map(math.degrees, q)) for q in goals],
            "stage_gripper_mm": [
                None
                if self.gripper_target is None
                else (0.0 if all(v == 0 for v in q) else self.gripper_target * 1000)
                for q in goals
            ],
            "stage_waypoints_deg": [
                [list(map(math.degrees, q)) for q in points] for points in stage_points
            ],
            "max_step_deg": math.degrees(self.step),
            "initially_enabled": warm,
            "speed_percent": self.speed,
            "gripper_open_mm": None if self.gripper_target is None else self.gripper_target * 1000,
            "gripper_force_parameter_n": None if self.gripper_target is None else 1.0,
            "gripper_width_tolerance_mm": None if self.gripper_target is None else 0.5,
            "after_target": "return to zero and remain enabled"
            if self.return_zero
            else "remain enabled at target",
            "feedback_timeout_s": FEEDBACK_TIMEOUT,
            "waypoint_timeout_s": WAYPOINT_TIMEOUT,
        }
        print(
            json.dumps(
                {k: v for k, v in self.report["plan"].items() if k != "stage_waypoints_deg"},
                indent=2,
            ),
            flush=True,
        )
        if confirm("确认目标、现场空间及夹爪空载，按回车开始本轮运动；输入其他内容取消：").strip():
            raise RuntimeError("operator cancelled")
        current, flags, _, _ = self.read()
        if flags != initially_enabled or any(
            abs(a - b) > TOLERANCE for a, b in zip(start, current)
        ):
            raise RuntimeError("pose or enable state changed while awaiting operator")
        self.report["operator_approved"] = True
        self.send("disable_auto_mode", self.arm.set_auto_set_motion_mode_enabled, False)
        self.send("disable_sdk_clipping", self.arm.set_joint_limits_enabled, False)
        # Keep an existing holding session enabled; otherwise enable once.
        if not warm:
            requested = self.clock()
            self.send("enable_all_joints", self.arm.enable)
            q, _, _, _ = self.wait(
                lambda q, flags, f, stamps: all(flags) and all(t >= requested for t in stamps), 3.0
            )
            self.active = True
            if any(abs(a - b) > TOLERANCE for a, b in zip(start, q)):
                raise RuntimeError("pose changed during enabling; no position target sent")
        self.send("speed_percent", self.arm.set_speed_percent, self.speed)
        requested = self.clock()
        self.send("CAN_J_mode", self.arm.set_motion_mode, self.arm.OPTIONS.MOTION_MODE.J)
        self.wait(
            lambda q, flags, f, stamps: (
                f.received_s[3] >= requested
                and f.status.msg.ctrl_mode == 1
                and f.status.msg.mode_feedback == 1
            ),
            3.0,
        )
        self.mode_required = True
        self.report["completed_stages"] = []
        for stage_index, (goal, points) in enumerate(zip(goals, stage_points)):
            self.phase = f"joint_stage_{stage_index + 1}"
            for index, target in enumerate(points):
                q, _, _, _ = self.read(True)
                if any(abs(a - b) > self.step + 1e-9 for a, b in zip(q, target)):
                    raise RuntimeError(
                        f"measured distance to next target exceeds {math.degrees(self.step):g} degrees"
                    )
                segment_start = q
                requested = self.clock()
                self.send("joint_waypoint", self.arm.move_j, target)

                def arrived(q, flags, f, stamps):
                    # Real joint feedback can briefly move away from its target.
                    # Enforce excursion from the starting pose, not path monotonicity.
                    if any(abs(v - a) > self.step for v, a in zip(q, segment_start)):
                        raise RuntimeError(
                            f"measured joint excursion exceeded {math.degrees(self.step):g} degrees"
                        )
                    return all(t >= requested for t in f.received_s[:3]) and all(
                        abs(a - b) <= TOLERANCE for a, b in zip(q, target)
                    )

                self.wait(arrived)
                print(f"阶段 {stage_index + 1}/{len(goals)}：{index + 1}/{len(points)}", flush=True)
            self.wait_stable(requested, segment_start, goal)
            if self.gripper_target is not None:
                self.position_gripper(
                    0.0 if all(v == 0 for v in goal) else self.gripper_target, goal
                )
            self.report["completed_stages"].append(list(map(math.degrees, goal)))
        self.report["status"] = "target_feedback_confirmed_enabled"

    def hold_on_failure(self):
        """One measured hold target when feedback permits; never reset or disable."""
        if not any(
            c["name"] in ("enable_all_joints", "joint_waypoint") for c in self.report["commands"]
        ):
            return
        # A gripper fault must not prevent holding otherwise healthy arm joints.
        q, flags, _, _ = self.read(for_hold=True)
        if (
            not self.mode_required
            or not all(flags)
            or any(not lo <= v <= hi for v, (lo, hi) in zip(q, self.limits))
        ):
            raise RuntimeError("cannot issue a valid measured hold; operator intervention required")
        self.send("hold_current_on_failure", self.arm.move_j, q)
        self.report["failure_hold"] = "command sent; physical stop not verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-deg", type=float, nargs=6, required=True)
    parser.add_argument("--max-step-deg", type=float, default=5.0)
    parser.add_argument("--speed-percent", type=int, default=1)
    parser.add_argument("--via-zero", action="store_true")
    parser.add_argument("--return-zero", action="store_true")
    parser.add_argument("--gripper-open-mm", type=float)
    args = parser.parse_args()
    if os.geteuid() == 0 or not os.path.samefile("/proc/self/ns/net", "/run/netns/piper-can"):
        parser.error("run using the administrator's piper-socketcan exec launcher")
    if not sys.stdin.isatty():
        parser.error("interactive operator terminal required")
    reference = json.loads(args.reference.read_text())
    if reference["status"] != "read_complete":
        parser.error("reference readout is incomplete")
    limits = [
        (e["controller_values"]["min_angle_limit"], e["controller_values"]["max_angle_limit"])
        for e in reference["joint_limits"]
    ]
    report = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "started",
        "scope": "operator-requested joint commissioning; not formal motion acceptance",
        "script_source": Path(__file__).read_text(),
        "python": sys.executable,
        "reference": str(args.reference),
        "zero_calibration_written": False,
        "automatic_disable": False,
        "requested_target_deg": args.target_deg,
    }
    arm, run, receiver = None, None, None
    with args.output.open("x") as output:
        try:
            from lerobot_robot_outcome_piper.sdk import create_piper
            from lerobot_robot_outcome_piper.timing import FeedbackReceiver

            origin = json.loads(
                importlib.metadata.distribution("pyAgxArm").read_text("direct_url.json")
            )
            report["sdk_commit"] = origin["vcs_info"]["commit_id"]
            if report["sdk_commit"] != "799b8412fbe8b9156bc9892d3dbeb2df7e98be71":
                raise RuntimeError("SDK differs from reviewed version")
            arm = create_piper(reference["interface"], reference["firmware_driver"])
            arm.connect()
            trace_transmissions(arm.get_context().get_comm(), report)
            gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            receiver = FeedbackReceiver(arm, gripper, time.monotonic)
            if not receiver.wait_ready(1.0):
                raise RuntimeError("initial feedback incomplete")
            if arm.get_firmware(timeout=1.0, min_interval=0.0) != reference["firmware_identity"]:
                raise RuntimeError("firmware identity changed")
            run = JointRun(
                arm,
                receiver,
                limits,
                report,
                goal=list(map(math.radians, args.target_deg)),
                max_step=math.radians(args.max_step_deg),
                speed_percent=args.speed_percent,
                via_zero=args.via_zero,
                return_zero=args.return_zero,
                gripper=gripper,
                gripper_target=None
                if args.gripper_open_mm is None
                else args.gripper_open_mm / 1000,
            )
            run.run()
        except (Exception, KeyboardInterrupt) as exc:
            report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
            if run is not None:
                try:
                    run.hold_on_failure()
                except Exception as hold_error:
                    report["hold_error"] = str(hold_error)
        finally:
            if arm is not None and receiver is not None:
                try:
                    report["flange_feedback"] = read_flange_result(arm, receiver)
                except (ValueError, TypeError, RuntimeError) as exc:
                    report["flange_feedback_error"] = str(exc)
            if arm is not None:
                try:
                    arm.disconnect()  # Official disconnect releases resources, not motor torque.
                except Exception as exc:
                    report["status"], report["disconnect_error"] = "failed", str(exc)
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            json.dump(report, output, indent=2, allow_nan=False)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("script_source", "samples")}, indent=2
        )
    )
    return 0 if report["status"] == "target_feedback_confirmed_enabled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
