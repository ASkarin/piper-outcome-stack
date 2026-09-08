"""Restricted Xbox release/resume test on one already-enabled return-to-zero path.

Uses the production TeleopControl and JointHold. This does not grant the formal
Robot gate or test free Cartesian teleoperation, loaded grasp or process death.
"""

from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace as NS

from piper_joint_commission import JointRun, TOLERANCE, FEEDBACK_TIMEOUT, trace_transmissions
from lerobot_robot_outcome_piper.teleop_control import (
    HoldSettings,
    JointHold,
    TeleopControl,
    TeleopState,
)


class InputLost(RuntimeError):
    pass


class OperatorStop(RuntimeError):
    pass


class XboxHoldRun(JointRun):
    def capture(self):
        q, _, _, _ = self.read(True)
        if any(not lo <= v <= hi for v, (lo, hi) in zip(q, self.limits)):
            raise RuntimeError("hold target outside controller limits")
        self.window = JointHold(q, self.settings, self.clock())
        self.send("capture_hold_once", self.arm.move_j, q)
        return q

    def confirm_hold(self, q, frame):
        confirmed = self.window.observe(q, frame.received_s[:3], self.clock())
        if self.control.hold_confirmed and not confirmed:
            self.control.request_hold()
        elif confirmed and not self.control.hold_confirmed:
            self.control.confirm_hold()
        return confirmed

    def hold_input_fault(self):
        if self.control.state is TeleopState.RUNNING:
            self.control.request_hold()
            self.capture()
        while True:
            q, _, frame, _ = self.read(True)
            if self.confirm_hold(q, frame):
                self.control.stop(False)
                self.report["stop_result"] = "hold_confirmed_then_fault"
                return
            self.sleep(0.01)

    def run_xbox(self, reader, confirm=input):
        self.settings = HoldSettings(TOLERANCE, 0.3, 10.0)  # Existing commissioning values only.
        self.control = TeleopControl()
        self.mode_required = True
        self.active = True
        initial, _, _, _ = self.read(True)
        if any(abs(a - b) > TOLERANCE for a, b in zip(initial, self.expected_start)):
            raise RuntimeError("pose differs from the previous hold report")
        if reader().emergency:
            raise RuntimeError("B is pressed before commissioning starts")
        print("仅测试从当前J5约+2.932°返回零位：速度1%，肩键启动，途中松键保持，回中重新按下继续。")
        print("保持候选参数沿用联调：0.1° / 稳定0.3秒 / 超时10秒，不写正式验收。")
        print("夹爪闭合0mm/1N。B或控制故障请求电子急停，可能阻尼下降；不自动失能。")
        if confirm(
            "确认在机旁、空载且路径清空；松开肩键并回中后，按Enter开始（其他内容取消）："
        ).strip():
            raise RuntimeError("operator cancelled")
        q, _, _, _ = self.read(True)
        if any(abs(a - b) > TOLERANCE for a, b in zip(q, initial)):
            raise RuntimeError("pose changed during confirmation")
        self.report["operator_approved"] = True
        self.send("disable_auto_mode", self.arm.set_auto_set_motion_mode_enabled, False)
        self.send("disable_sdk_clipping", self.arm.set_joint_limits_enabled, False)
        self.send("speed_percent", self.arm.set_speed_percent, 1)
        self.position_gripper(0.0, initial)
        self.window = None
        self.capture()
        self.report["input_samples"] = []
        self.phase = "initial_hold"
        self.started_motion = False
        pause_seen = False
        finishing = None
        run_count = 0
        sent_epoch = None
        arrival = None
        previous = None
        displayed_at = 0.0
        last_tick = self.clock()
        deadline = last_tick + 90  # Operator session budget, not a frozen motion threshold.
        while self.clock() < deadline:
            inp = reader()  # B is checked before axes, feedback or target dispatch.
            if inp.emergency:
                self.control.stop(True)
                raise OperatorStop("B pressed")
            if self.clock() - last_tick > FEEDBACK_TIMEOUT:
                raise InputLost("commissioning input loop stalled")
            last_tick = self.clock()
            q, _, frame, _ = self.read(True)
            if any(abs(a - b) > self.step for a, b in zip(q, initial)):
                raise RuntimeError("joint exceeded the commissioning excursion envelope")
            was_running = self.control.state is TeleopState.RUNNING
            intent, epoch = self.control.observe(inp.hold, inp.neutral)
            if was_running and not inp.hold:
                self.phase = "pause_hold"
                self.capture()
                if run_count == 1:
                    pause_seen = (
                        max(abs(v) for v in q) > TOLERANCE
                        and max(abs(a - b) for a, b in zip(q, initial)) > TOLERANCE
                    )
                    self.report["released_before_arrival"] = pause_seen
                    if not pause_seen:
                        finishing = "pause_not_demonstrated"
            if self.control.state is not TeleopState.RUNNING:
                confirmed = self.confirm_hold(q, frame)
                if confirmed and finishing:
                    self.report.update(
                        status=finishing,
                        final_hold_confirmed=True,
                        final_hold_target_rad=list(self.window.target),
                    )
                    return
            elif sent_epoch != epoch:
                if not self.window.confirmed or not self.window.within(q):
                    self.control.request_hold()
                    self.window.restart(self.clock())
                    continue
                if any(abs(v) > self.step for v in q) or any(
                    not lo <= 0 <= hi for lo, hi in self.limits
                ):
                    raise RuntimeError("zero target outside allowed step or limits")
                self.phase = "return_zero" if run_count == 0 else "resume_zero"
                arrival = JointHold([0.0] * 6, self.settings, self.clock())
                self.send("return_zero_once", self.arm.move_j, [0.0] * 6)
                self.started_motion = True
                run_count += 1
                sent_epoch = epoch
            elif arrival.observe(q, frame.received_s[:3], self.clock()):
                finishing = (
                    "release_resume_complete"
                    if pause_seen and run_count >= 2
                    else "pause_not_demonstrated"
                )
                self.control.request_hold()
                self.phase = "final_hold"
                self.capture()
            row = dict(
                monotonic_s=self.clock(),
                hold=inp.hold,
                neutral=inp.neutral,
                state=self.control.state.value,
                epoch=self.control.epoch,
                hold_target_rad=list(self.window.target),
                hold_confirmed=self.window.confirmed,
            )
            self.report["input_samples"].append(row)
            if row["state"] != previous:
                print("状态：", row["state"], flush=True)
                if row["state"] == "RUNNING":
                    print("正在返回零位；第一段请在运动途中松开肩键。", flush=True)
                if row["state"] == "PAUSED":
                    print("保持已确认；回中、松肩键，再重新按下可继续。", flush=True)
                previous = row["state"]
            if self.control.state is TeleopState.RUNNING and self.clock() - displayed_at >= 0.2:
                print(f"J5={math.degrees(q[4]):.3f}°", flush=True)
                displayed_at = self.clock()
            self.sleep(0.01)
        raise InputLost("operator session time budget elapsed")

    def stop_after_failure(self, exc):
        hold_fault = False
        if isinstance(exc, InputLost) and getattr(self, "window", None) is not None:
            try:
                self.hold_input_fault()
                hold_fault = True
            except Exception as hold_error:
                self.report["hold_error"] = str(hold_error)
        if not hold_fault and self.report.get("operator_approved"):
            try:
                self.send("electronic_stop_on_fault", self.arm.electronic_emergency_stop)
                self.report["stop_result"] = "sent_not_physically_confirmed"
            except Exception as stop_error:
                self.report.update(stop_result="unknown", stop_error=str(stop_error))


def xbox_reader(mapping):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
    import pygame

    pygame.display.init()
    pygame.joystick.init()
    devices = []

    def close():
        for j in devices:
            j.quit()
        pygame.joystick.quit()
        pygame.display.quit()

    try:
        for i in range(pygame.joystick.get_count()):
            j = pygame.joystick.Joystick(i)
            j.init()
            devices.append(j)
        matches = [j for j in devices if j.get_guid() == mapping["device"]["guid"]]
        if len(matches) != 1:
            raise InputLost("expected one measured Xbox GUID")
        j = matches[0]
    except BaseException:
        close()
        raise

    def read():
        pygame.event.pump()
        for e in pygame.event.get(pygame.JOYDEVICEREMOVED):
            if e.instance_id == j.get_instance_id():
                raise InputLost("Xbox disconnected")
        if not j.get_init():
            raise InputLost("Xbox disconnected")
        b = bool(j.get_button(mapping["buttons"]["emergency_stop_button"]))
        if b:
            return NS(emergency=True, hold=False, neutral=False)
        values = []
        for key in (
            "left_stick_horizontal",
            "left_stick_vertical",
            "right_stick_horizontal",
            "right_stick_vertical",
        ):
            values.append(abs(float(j.get_axis(mapping["physical_axes"][key]["index"]))))
        for key in ("left_trigger", "right_trigger"):
            axis = mapping["physical_axes"][key]
            value = (j.get_axis(axis["index"]) - axis["released"]) / (
                axis["pressed"] - axis["released"]
            )
            if not 0 <= value <= 1:
                raise ValueError("trigger outside measured range")
            values.append(value)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite Xbox input")
        return NS(
            emergency=False,
            hold=bool(j.get_button(mapping["buttons"]["hold_button"])),
            neutral=all(v <= 0.08 for v in values),
        )

    return read, close


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "start-report", "mapping", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if (
        os.geteuid() == 0
        or not os.path.samefile("/proc/self/ns/net", "/run/netns/piper-can")
        or not sys.stdin.isatty()
    ):
        parser.error(
            "ordinary administrator interactive terminal through piper-socketcan exec required"
        )
    ref = json.loads(args.reference.read_text())
    start = json.loads(args.start_report.read_text())
    mapping = json.loads(args.mapping.read_text())
    if (
        ref["status"] != "read_complete"
        or start["status"] != "hold_measurement_complete"
        or mapping["status"] != "physical_mapping_confirmed"
    ):
        parser.error("completed reference, initial hold and input reports required")
    report = dict(
        status="started",
        scope="bounded Xbox return-zero release/resume commissioning; not full Robot/IK/watchdog acceptance",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        script_source=Path(__file__).read_text(),
        implementation_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True
        ).strip(),
        implementation_diff=subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=Path(__file__).parents[2], text=True
        ),
        reference=str(args.reference),
        start_report=str(args.start_report),
        mapping=str(args.mapping),
        candidate_hold={"joint_tolerance_deg": 0.1, "stable_time_s": 0.3, "timeout_s": 10},
        deadzone_candidate=0.08,
        formal_hold_acceptance=False,
        automatic_disable=False,
    )
    arm = run = close = None
    with args.output.open("x") as output:
        try:
            reader, close = xbox_reader(mapping)
            if reader().emergency:
                raise RuntimeError("release B before starting")
            from lerobot_robot_outcome_piper.sdk import create_piper
            from lerobot_robot_outcome_piper.timing import FeedbackReceiver

            origin = json.loads(
                importlib.metadata.distribution("pyAgxArm").read_text("direct_url.json")
            )
            report["sdk_commit"] = origin["vcs_info"]["commit_id"]
            if report["sdk_commit"] != "799b8412fbe8b9156bc9892d3dbeb2df7e98be71":
                raise RuntimeError("SDK differs from reviewed version")
            arm = create_piper(ref["interface"], ref["firmware_driver"])
            arm.connect()
            trace_transmissions(arm.get_context().get_comm(), report)
            gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            rx = FeedbackReceiver(arm, gripper, time.monotonic)
            if not rx.wait_ready(1):
                raise RuntimeError("feedback unavailable")
            if arm.get_firmware(timeout=1, min_interval=0) != ref["firmware_identity"]:
                raise RuntimeError("firmware identity mismatch")
            limits = [
                (
                    r["controller_values"]["min_angle_limit"],
                    r["controller_values"]["max_angle_limit"],
                )
                for r in ref["joint_limits"]
            ]
            run = XboxHoldRun(
                arm,
                rx,
                limits,
                report,
                goal=[0] * 6,
                gripper=gripper,
                gripper_target=0,
                speed_percent=1,
            )
            run.expected_start = start["samples"][-1]["joint_rad"]
            run.run_xbox(reader)
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if run is not None:
                run.stop_after_failure(exc)
        finally:
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    report.update(status="failed", input_close_error=str(exc))
            if arm is not None:
                try:
                    arm.disconnect()
                except Exception as exc:
                    report.update(status="failed", disconnect_error=str(exc))
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, output, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "samples",
                    "input_samples",
                    "script_source",
                    "tx_frames",
                    "implementation_diff",
                )
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "release_resume_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
