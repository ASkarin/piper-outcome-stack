"""Confirmed joint-only preparation, between receive-only teaching attempts."""

from .raw_io import write_json
from dataclasses import replace, asdict
import math
import time
import uuid
from pathlib import Path

from .robot import OutcomePiper
from .safety import JOINT_KEYS, load_motion_safety, check_joint_feedback
from .joint_pose import JointPoseSequence
from .teleop_control import HoldSettings, JointHold
from .control_trace import ControlTrace
from .teach_data import read_json
from .errors import OutcomePiperIntentRejected, OutcomePiperValidationError


def normalize_preparation(config):
    if set(config) != {"work_joint_rad", "safety_path", "hold_settings"}:
        raise ValueError("preparation requires work_joint_rad, safety_path, hold_settings")
    joints = list(config["work_joint_rad"])
    if len(joints) != 6 or not all(math.isfinite(q) for q in joints):
        raise ValueError("work pose requires six finite radians")
    safety = load_motion_safety(Path(config["safety_path"]))
    if any(not lo <= q <= hi for q, lo, hi in zip(joints, safety.joint_lower, safety.joint_upper)):
        raise ValueError("work pose outside joint limits")
    settings = HoldSettings(**config["hold_settings"])
    return dict(
        work_joint_rad=joints,
        hold_settings=asdict(settings),
        safety=read_json(config["safety_path"]),
    )


def prompt_preparation(goal):
    print(
        "[准备确认] 回工作姿态（度）：" + str([round(math.degrees(q), 3) for q in goal]), flush=True
    )
    print("先结束机械臂示教、手离开活动范围；输入 prepare 确认后运动。", flush=True)
    print("不发送夹爪开合/力指令。cancel取消确认；quit退出。", flush=True)


class TeachPreparation:
    def __init__(
        self,
        config,
        preparation,
        root,
        *,
        robot_factory=OutcomePiper,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self.root = Path(root)
        self.goal = preparation["work_joint_rad"]
        self.settings = HoldSettings(**preparation["hold_settings"])
        path = self.root / "preparation-safety.json"
        if path.exists():
            if read_json(path) != preparation["safety"]:
                raise ValueError("preparation safety snapshot changed")
        else:
            write_json(path, preparation["safety"])
        self.safety = load_motion_safety(path)
        self.config = replace(
            config, execution_mode="motion", safety_path=path, cameras={}, capture_timing=None
        )
        self.robot_factory = robot_factory
        self.clock = clock
        self.sleep = sleep
        self.reports = []
        self.control_commands_attempted = False

    def hold(self, robot, observation):
        target = [observation[k] for k in JOINT_KEYS]
        robot.send_joint_target(target)
        hold = JointHold(target, self.settings, self.clock())
        while True:
            started = self.clock()
            obs = robot.observe_control()
            if hold.observe(
                [obs[k] for k in JOINT_KEYS],
                robot.last_feedback_telemetry.received_monotonic_s[:3],
                self.clock(),
            ):
                return obs
            self.sleep(max(0.0, 0.05 - (self.clock() - started)))

    def move(self, robot, sequence, terminal):
        while True:
            started = self.clock()
            obs = robot.observe_control()
            now = self.clock()
            line = terminal.poll()
            if line in ("stop", "cancel", "quit"):
                return dict(status="cancelled", quit=line == "quit", held=self.hold(robot, obs))
            if line:
                print("[准备运动中] stop取消并保持；quit取消并退出。其他输入不排队。", flush=True)
            if sequence.window is not None and now >= sequence.window.deadline:
                return dict(status="incomplete_held", held=self.hold(robot, obs))
            sequence.observe(
                [obs[k] for k in JOINT_KEYS],
                robot.last_feedback_telemetry.received_monotonic_s[:3],
                now,
                obs["gripper.pos"],
            )
            if sequence.complete:
                return dict(status="arrived", observed=obs)
            sequence.next_waypoint()
            robot.send_joint_target([sequence.values[k] for k in JOINT_KEYS])
            sequence.sent(self.clock())
            self.sleep(max(0.0, 0.05 - (self.clock() - started)))

    def perform(self, source, terminal):
        feedback = source.feedback(require_teach=False)
        if feedback["teach_status"] == 1 or feedback["arm_status"] != 0:
            return dict(status="rejected", reason="先结束示教并确认控制器正常；未发送控制指令")
        directory = self.root / f"preparation-{uuid.uuid4().hex}"
        directory.mkdir()
        trace = ControlTrace(directory / "control-trace.jsonl")
        report = dict(
            status="started",
            target_joint_rad=list(self.goal),
            gripper_commands="none",
            started_monotonic_s=self.clock(),
            control_commands_attempted=False,
        )
        self.reports.append(directory.name)
        robot = None
        source_closed = False
        quit_requested = False
        try:
            source.disconnect()
            source_closed = True
            robot = self.robot_factory(self.config)
            robot.control_trace = trace
            robot.connect()
            obs = robot.get_observation()
            # Recheck the SDK's already-received status after the receiver handover.
            status = robot._arm.get_arm_status()
            if status is None:
                raise RuntimeError("controller status unavailable after handover")
            if int(status.msg.teach_status) == 1:
                report.update(status="rejected", reason="示教录制尚未结束；未使能或运动")
                return report
            joints = [obs[k] for k in JOINT_KEYS]
            check_joint_feedback(joints, self.safety)
            sequence = JointPoseSequence(
                joints, obs["gripper.pos"], self.safety, self.settings, self.goal, "prepare_work"
            )
            # Match the dedicated Robot's dispatch admission before enabling.
            first = {**sequence.values, "gripper.pos": obs["gripper.pos"]}
            robot._validate_action(first, joints_only=True)
            report["control_commands_attempted"] = self.control_commands_attempted = True
            robot.enable()
            print("[返回工作姿态] 正在移动；stop取消并保持，Ctrl+C电子急停可能下沉。", flush=True)
            report.update(self.move(robot, sequence, terminal))
            quit_requested = report.get("quit", False)
            return report
        except BaseException as exc:
            if not report["control_commands_attempted"] and isinstance(
                exc, (OutcomePiperIntentRejected, OutcomePiperValidationError)
            ):
                report.update(status="rejected", reason=str(exc))
                return report
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if robot is not None and report["control_commands_attempted"]:
                try:
                    robot.request_emergency_stop(exc)
                except Exception as stop_error:
                    report["stop_error"] = str(stop_error)
            raise
        finally:
            try:
                if robot is not None:
                    robot.disconnect()
            except Exception as exc:
                report.update(status="failed", cleanup_error=str(exc))
                raise
            finally:
                report["finished_monotonic_s"] = self.clock()
                trace.save(stop_outcome=None if robot is None else robot.stop_outcome)
                write_json(directory / "result.json", report)
            if source_closed and report["status"] not in ("failed",) and not quit_requested:
                source.connect(require_teach=False)
