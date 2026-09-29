"""Terminal episode decisions while the official control loop stays alive."""

from concurrent.futures import ThreadPoolExecutor
import math
import os
import select
import sys
import time
from .teleop_control import TeleopState, TeleopMode
from .errors import OutcomePiperIntentRejected
from .console import operator_message


class TerminalCommands:
    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        if not self.stream.isatty():
            raise ValueError("interactive recording requires the operator terminal")
        self.buffer = b""

    def poll(self):
        if b"\n" not in self.buffer:
            if not select.select([self.stream], [], [], 0)[0]:
                return None
            block = os.read(self.stream.fileno(), 4096)
            if not block:
                raise EOFError("recording terminal disconnected")
            self.buffer += block
        if b"\n" not in self.buffer:
            return None
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode(self.stream.encoding or "utf-8").strip()


class EpisodeControls:
    def __init__(self, control, events, terminal, emit, *, work_available=False, duration_s=None):
        self.control, self.events, self.terminal, self.emit = control, events, terminal, emit
        self.phase = None
        self.request = None
        self.job = None
        self.preparation = None
        self.position = None
        self.start_epoch = None
        self.work_available = work_available
        self.duration_s = duration_s

    def enter(self, phase):
        self.phase = self.control.recording_phase = phase
        self.request = None
        self.events["exit_early"] = False
        if phase != "recording":
            self.control.request_hold()
        self.emit("recording_phase", phase=phase)
        mode = "平移" if self.control.mode is TeleopMode.TRANSLATION else "姿态"
        pose_keys = "A进入工作姿态，Y回零" if self.work_available else "Y回零"
        duration = (
            "" if self.duration_s is None else f"最长 {self.duration_s:g} 秒；暂停期间仍计时。"
        )
        messages = {
            "preparing": f"[准备] {mode}模式｜{pose_keys}。\n"
            "松开LB，让摇杆和扳机回中，等待保持确认。\n"
            "开始录制：start <起点编号>；退出并保留已保存回合：quit。",
            "recording": f"[录制中] 起点 {self.position}｜{duration}\n"
            "按住LB推杆操作，松LB保持；提前结束输入 end。A/Y本阶段不执行。",
            "review": "[录制结束] 遥操作已暂停，请等待保持确认后选择：\n"
            "save success：保存并标记任务成功；save failure [备注]：保存并标记任务失败。\n"
            "save cancelled [备注]：保存并标记任务取消；redo：放弃本次并重录。\n"
            "quit：放弃当前未保存回合并退出，已保存回合保留。",
            "saving": "[保存中] 不接受遥操作，请等待。B急停仍有效；返回准备阶段才可继续。",
            "finalizing": "[正在退出] 正在完成数据收尾，请等待；B急停仍有效。",
        }
        operator_message(messages[phase])
        if phase == "preparing" and self.control.gripper_target is None:
            operator_message(
                "首次 start 会按当前合法宽度接管夹爪，并施加配置的保持力；无需先按扳机。",
            )

    def poll(self):
        if self.preparation is not None and self.preparation.done():
            self.preparation.result()
            self.preparation = None
            if self.phase == "preparing":
                operator_message("[文件准备完成] 保持确认并回中后可输入 start。")
        if self.job is not None:
            if self.terminal.poll() is not None:
                operator_message("[请等待] 正在保存或收尾，不接受其他命令；B急停仍有效。")
            if self.job.done():
                self.job.result()
                if self.control.hold_confirmed:
                    self.events["exit_early"] = True
            return
        line = self.terminal.poll()
        if line is None:
            return
        parts = line.split(maxsplit=2)
        if not parts:
            return
        ready = self.control.hold_confirmed and self.control.state in (
            TeleopState.WAITING,
            TeleopState.PAUSED,
            TeleopState.CENTERED,
        )
        request = None
        if self.phase == "preparing" and parts[0] == "start" and len(parts) == 2 and ready:
            if self.preparation is not None and not self.preparation.done():
                operator_message("[准备中] 正在准备本回合文件，请稍后重新输入 start；B急停仍有效。")
                return
            self.position = parts[1]
            request = ("start",)
        elif self.phase == "recording" and line == "end":
            request = ("end",)
        elif (
            self.phase == "review"
            and parts[0] == "save"
            and len(parts) >= 2
            and parts[1] in ("success", "failure", "cancelled")
            and ready
        ):
            request = ("save", parts[1], parts[2] if len(parts) == 3 else "")
        elif self.phase == "review" and line == "redo" and ready:
            request = ("redo",)
        elif (
            self.phase in ("preparing", "review")
            and line == "quit"
            and self.control.hold_confirmed
            and self.control.state not in (TeleopState.RUNNING, TeleopState.POSE_MOVING)
        ):
            request = ("quit",)
        if request is None:
            allowed = {
                "preparing": {"start", "quit"},
                "recording": {"end"},
                "review": {"save", "redo", "quit"},
            }
            if parts[0] not in allowed.get(self.phase, set()):
                hint = (
                    "录制中请先输入 end 结束本回合。"
                    if self.phase == "recording"
                    else "请使用上方列出的当前阶段命令。"
                )
            elif self.phase != "recording" and not self.control.hold_confirmed:
                hint = "请先松开LB并回中，等待‘保持已确认’，再输入命令。"
            elif self.phase != "recording" and not ready and parts[0] != "quit":
                hint = "请先取消或完成当前姿态操作，进入已暂停状态后再试。"
            else:
                hint = (
                    "格式：start <起点编号>，例如 start debug1。"
                    if parts[0] == "start"
                    else "格式：save success、save failure [备注] 或 save cancelled [备注]。"
                    if parts[0] == "save"
                    else "该命令无需附加参数。"
                )
            operator_message(f"[未执行] {hint}")
            return
        self.request = request
        self.events["exit_early"] = True


def run_interactive_episodes(official, loop_args, audit, ds, control, terminal=None):
    """SDK traffic remains in the ordinary loop; workers only save/discard/finalize."""
    terminal = TerminalCommands() if terminal is None else terminal
    teleop_config = getattr(loop_args.get("teleop"), "config", None)
    ui = EpisodeControls(
        control,
        loop_args["events"],
        terminal,
        audit.emit,
        work_available=getattr(teleop_config, "work_pose_button", None) is not None,
        duration_s=getattr(ds, "episode_time_s", None),
    )
    processor = loop_args["teleop_action_processor"]

    def process(value):
        # Read raw B first when available; it must not wait behind terminal input.
        if not value[0].get("emergency_stop", False):
            from .stage_timing import span

            with span("terminal_and_writer_check"):
                audit.check_writer()
                ui.poll()
        result = processor(value)
        if ui.request == ("start",):
            raw = value[0]
            if (
                raw.get("hold", True)
                or not raw.get("neutral", False)
                or any(
                    raw.get(k, False) for k in ("mode_switch", "translation_switch", "home", "work")
                )
            ):
                ui.request = None
                ui.events["exit_early"] = False
                operator_message(
                    "[未开始] 请松开LB和模式/姿态按键，摇杆、扳机回中后重新输入 start。"
                )
            else:
                ui.start_epoch = control.epoch
        return result

    args = {**loop_args, "teleop_action_processor": process}

    def loop(phase, *, recording=False):
        ui.enter(phase)
        robot = loop_args.get("robot")
        previous_trace = None if robot is None else getattr(robot, "control_trace", None)
        if phase == "preparing" and robot is not None:
            from .control_trace import PreparationTimingTrace

            robot.control_trace = PreparationTimingTrace(audit.emit)
        try:
            official.record_loop(
                **args,
                dataset=audit if recording else None,
                control_time_s=ds.episode_time_s if recording else math.inf,
            )
        finally:
            if robot is not None:
                robot.control_trace = previous_trace

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper-dataset-save") as writer:

        def job(phase, fn):
            ui.enter(phase)
            ui.job = writer.submit(fn)
            try:
                official.record_loop(**args, dataset=None, control_time_s=math.inf)
                ui.job.result()
            finally:
                ui.job = None

        count = 0
        try:
            while count < ds.num_episodes:
                operator_message(f"[本次进度] 已保存 {count}/{ds.num_episodes} 回合，可提前退出。")
                ui.preparation = audit.prepare_episode()
                loop("preparing")
                if ui.request and ui.request[0] == "quit":
                    break
                try:
                    reference = loop_args["robot"].prepare_recording_gripper(ui.start_epoch)
                except OutcomePiperIntentRejected as exc:
                    operator_message(f"[未开始] {exc}；请调整后重新输入 start。")
                    continue
                audit.emit("recording_gripper_reference", **reference)
                audit.emit(
                    "episode_started",
                    episode_index=audit.dataset.num_episodes,
                    attempt=audit.attempt,
                    position_id=ui.position,
                )
                loop("recording", recording=True)
                loop("review")
                request = ui.request
                if request is None:
                    raise RuntimeError("recording review exited without an operator decision")
                if request[0] in ("redo", "quit"):
                    job("saving", audit.discard_episode)
                    operator_message("[已放弃本次回合] 已保存的数据不受影响。")
                    if request[0] == "quit":
                        break
                    continue
                index, attempt = audit.dataset.num_episodes, audit.attempt
                frames = len(audit.pending)
                audit.emit(
                    "episode_save_requested",
                    episode_index=index,
                    attempt=attempt,
                    position_id=ui.position,
                    task_outcome=request[1],
                    note=request[2],
                )
                save_started = time.monotonic()

                def seal():
                    audit.save_episode()
                    if frames:
                        audit.emit(
                            "episode_outcome",
                            episode_index=index,
                            attempt=attempt,
                            position_id=ui.position,
                            task_outcome=request[1],
                            note=request[2],
                            data_valid=True,
                            save_duration_s=time.monotonic() - save_started,
                        )
                    audit.flush()

                job("saving", seal)
                save_elapsed = time.monotonic() - save_started
                if frames:
                    count += 1
                    operator_message(
                        f"[原始回合已保存] 本次第 {count} 回合｜{frames} 帧｜封存 {save_elapsed:.1f} 秒。可开始下一起点。",
                    )
                else:
                    operator_message(
                        "[未保存] 本回合没有可用数据；返回准备阶段后重新输入 start。",
                    )

            job("finalizing", audit.dataset.finalize)
        finally:
            control.recording_phase = None
    return True
