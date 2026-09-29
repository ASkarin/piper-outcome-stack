"""Operator-driven raw teaching; never dispatches robot commands or creates actions."""

from .raw_io import write_json
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from .teach_data import (
    SCHEMA,
    load_config,
    read_json,
    validate_sample,
    save_pixels,
    timing_summary,
)
from .teach_source import TeachSource
from .raw_io import check_queue, drain_queue


def utc():
    return datetime.now(timezone.utc).isoformat()


class Attempt:
    def __init__(self, path, position, config, connection_id=None):
        self.path, self.config = Path(path), config
        self.path.mkdir()
        self.log = (self.path / "samples.jsonl").open("x")
        self.rows = []
        self.pending = deque()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="teach-raw-writer")
        self.result = dict(
            status="recording",
            data_valid=False,
            position_id=position,
            started_at_utc=utc(),
            source_connection=connection_id,
            frames=0,
        )
        write_json(self.path / "result.json", self.result)

    def append(self, frames, row):
        row["frame_index"] = len(self.rows)
        validate_sample(row, self.rows[-1] if self.rows else None, self.config)
        check_queue(self.pending, self.config["writer_queue_size"])
        row["files"] = {
            key: f"{len(self.rows):06d}-{i}.{'npz' if row['camera'][key]['stream'] == 'depth' else 'png'}"
            for i, key in enumerate(frames)
        }
        self.pending.append(self.pool.submit(save_pixels, self.path, row, frames, self.config))
        self.log.write(json.dumps(row, allow_nan=False) + "\n")
        self.log.flush()
        self.rows.append(row)

    def finish(self, reason):
        self.result.update(
            status="review",
            end_reason=reason,
            ended_at_utc=utc(),
            frames=len(self.rows),
            timing=timing_summary(self.rows),
        )
        self.log.close()
        write_json(self.path / "result.json", self.result)

    def drain(self):
        try:
            drain_queue(self.pending)
        finally:
            self.pool.shutdown(wait=True)

    def save(self, outcome, note):
        from .episode_save import compress_depth_file

        self.drain()
        if len(self.rows) < 2:
            raise ValueError("fewer than two samples; no next-state action can be constructed")
        stats = []
        for row in self.rows:
            for key, file in row["files"].items():
                if row["camera"][key]["stream"] == "depth":
                    stats.append(compress_depth_file(self.path / file, "npz"))
        write_json(self.path / "compression.json", stats)
        self.result.update(status="saved", data_valid=True, task_outcome=outcome, note=note)
        write_json(self.path / "result.json", self.result)

    def discard(self, reason):
        self.drain()
        self.result.update(status="discarded", data_valid=False, discard_reason=reason)
        write_json(self.path / "result.json", self.result)

    def fail(self, error):
        self.log.close()
        self.result.update(
            status="failed", data_valid=False, frames=len(self.rows), error=str(error)
        )
        try:
            self.drain()
        except Exception as exc:
            self.result["writer_error"] = str(exc)
        write_json(self.path / "result.json", self.result)


def run_session(
    source,
    config,
    root,
    terminal,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    connection_id=None,
    preparation=None,
):
    """Poll terminal and receiver in every phase; file work never holds SDK locks."""
    from .capture_gc import CaptureGC

    capture_gc = CaptureGC()
    root = Path(root)
    phase = "prepare_confirm" if preparation is not None else "preparing"
    attempt = None
    job = None
    previous_tick = None
    start = None
    saved = 0
    allow_direct_start = False
    requested_at = None
    if preparation is not None:
        from .teach_prepare import prompt_preparation

        prompt_preparation(preparation.goal)
    else:
        print("[准备] start <起点编号> 开始；quit 退出。机械臂示教按钮由你操作。", flush=True)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="teach-save") as saver:
        try:
            while True:
                tick_start = clock()
                feedback = (
                    source.feedback(require_teach=phase == "recording")
                    if preparation is not None
                    else source.feedback()
                )
                line = terminal.poll()
                if (
                    phase == "prepare_confirm"
                    and allow_direct_start
                    and line
                    and len(line.split()) == 2
                    and line.split()[0] == "start"
                ):
                    phase = "preparing"
                if job is not None:
                    if line is not None:
                        print("[请等待] 正在保存，不排队执行输入。", flush=True)
                    if job.done():
                        job.result()
                        saved += 1
                        print(
                            f"[已保存] {len(attempt.rows)} 帧；任务结果 {attempt.result['task_outcome']}。",
                            flush=True,
                        )
                        attempt, job = None, None
                        phase = "prepare_confirm" if preparation is not None else "preparing"
                        if preparation is not None:
                            allow_direct_start = True
                            print(
                                "[下一回合] start <起点编号> 直接录制；prepare 回工作姿态；quit 退出。",
                                flush=True,
                            )
                        else:
                            print("[准备] start <起点编号> 开始下一回合；quit 退出。", flush=True)
                elif phase == "prepare_confirm":
                    if line == "quit":
                        return saved
                    if line == "prepare":
                        allow_direct_start = False
                        if feedback["teach_status"] == 1:
                            print(
                                "[未执行] 请先单击结束机械臂示教录制，再输入 prepare。", flush=True
                            )
                            continue
                        outcome = preparation.perform(source, terminal)
                        if outcome.get("quit"):
                            return saved
                        if outcome["status"] == "arrived":
                            phase = "preparing"
                            print(
                                "[工作姿态已到位] 请切入示教录制模式；准备好后输入 start P1（或下一起点编号）。",
                                flush=True,
                            )
                        else:
                            print(
                                "[未开始下一回合] " + outcome.get("reason", outcome["status"]),
                                flush=True,
                            )
                            prompt_preparation(preparation.goal)
                    elif line:
                        print(
                            "[未执行运动] prepare 确认回工作姿态；cancel 保持等待；quit 退出。",
                            flush=True,
                        )
                elif phase == "preparing":
                    if line == "quit":
                        return saved
                    if line:
                        parts = line.split()
                        if len(parts) == 2 and parts[0] == "start":
                            capture_gc.prepare()
                            attempt = Attempt(
                                root / f"attempt-{uuid.uuid4().hex}",
                                parts[1],
                                config,
                                connection_id,
                            )
                            phase = "waiting"
                            requested_at = clock()
                            print(
                                "[等待示教] 已在示教录制则直接采样；否则单击示教按钮，等待控制器反馈。start 不切换电机状态。end 取消等待。",
                                flush=True,
                            )
                        else:
                            print("[未执行] 使用 start <起点编号> 或 quit。", flush=True)
                elif phase in ("waiting", "recording"):
                    reason = None
                    if line in ("end", "quit"):
                        reason = "operator_end" if line == "end" else "operator_quit"
                    elif phase == "recording" and feedback["teach_status"] == 2:
                        reason = "teach_stopped"
                    elif phase == "recording" and clock() - start >= config["episode_time_s"]:
                        reason = "duration_limit"
                    if reason:
                        capture_gc.stop()
                        attempt.result["capture_runtime"] = capture_gc.summary()
                        attempt.finish(reason)
                        print(
                            f"[录制结束] {len(attempt.rows)} 帧｜{reason}。停止采样不改变机械臂；示教灯仍亮时请手动结束。",
                            flush=True,
                        )
                        if line == "quit":
                            attempt.discard(reason)
                            attempt = None
                            return saved
                        phase = "review"
                        print(
                            "[回合处理] save success / save failure [备注] / save cancelled [备注]；redo；quit。",
                            flush=True,
                        )
                    elif feedback["teach_status"] == 1 and feedback["ctrl_mode"] == 2:
                        if phase == "waiting":
                            capture_gc.start()
                            wait_s = clock() - requested_at
                            attempt.result["start_to_teach_feedback_s"] = wait_s
                            print(
                                f"[示教反馈已确认] start 后 {wait_s:.2f} 秒；含人工按键等待，不代表驱动释放时间。",
                                flush=True,
                            )
                            start, previous_tick, phase = clock(), None, "recording"
                            print(
                                f"[录制中] 最长 {config['episode_time_s']:g} 秒；停止示教或输入 end 结束。",
                                flush=True,
                            )
                        slot = int((clock() - start) * config["fps"] + 1e-6)
                        if previous_tick is not None and slot != previous_tick + 1:
                            raise RuntimeError(
                                "missed sampling tick; no image or state was repeated"
                            )
                        frames, row = source.read()
                        if row["teach_status"] == 2:
                            capture_gc.stop()
                            attempt.result["capture_runtime"] = capture_gc.summary()
                            attempt.finish("teach_stopped")
                            phase = "review"
                            print(
                                "[录制结束] 示教已停止；save success / save failure / save cancelled；redo；quit。",
                                flush=True,
                            )
                        else:
                            row["tick_index"] = slot
                            attempt.append(frames, row)
                            previous_tick = slot
                        if line:
                            print("[录制中] 结束请输入 end，其他命令不执行。", flush=True)
                elif phase == "review" and line:
                    parts = line.split(maxsplit=2)
                    if (
                        parts[0] == "save"
                        and len(parts) >= 2
                        and parts[1] in ("success", "failure", "cancelled")
                    ):
                        print(
                            "[保存中] 正在写盘和无损压缩；程序不控制机械臂，Xbox B 不生效。",
                            flush=True,
                        )
                        phase = "saving"
                        job = saver.submit(
                            attempt.save, parts[1], parts[2] if len(parts) > 2 else ""
                        )
                    elif line in ("redo", "quit"):
                        attempt.discard(line)
                        attempt = None
                        if line == "quit":
                            return saved
                        phase = "prepare_confirm" if preparation is not None else "preparing"
                        print("[已放弃] 原始记录保留。", flush=True)
                        if preparation is not None:
                            allow_direct_start = True
                            print(
                                "[下一回合] start <起点编号> 直接重录；prepare 回工作姿态；quit 退出。",
                                flush=True,
                            )
                        else:
                            print("start <起点编号> 重新开始。", flush=True)
                    else:
                        print(
                            "[未执行] 使用 save success/failure/cancelled [备注]、redo 或 quit。",
                            flush=True,
                        )
                # Anchor capture to its fixed cadence. Waiting/review remains responsive.
                deadline = (
                    start + (previous_tick + 1) / config["fps"]
                    if phase == "recording" and previous_tick is not None
                    else tick_start + 1 / config["fps"]
                )
                sleep(max(0.0, deadline - clock()))
        except BaseException as exc:
            capture_gc.stop()
            if attempt is not None:
                attempt.result["capture_runtime"] = capture_gc.summary()
            if job is not None:
                # Never leave a worker that can mark an interrupted attempt saved later.
                try:
                    job.result()
                except BaseException:
                    pass
            if attempt is not None:
                attempt.fail(exc)
            raise
        finally:
            capture_gc.stop()


def main(argv=None, *, preparation_factory=None):
    parser = argparse.ArgumentParser(
        description=__doc__
        if preparation_factory is None
        else "Explicitly confirmed work-pose motion between receive-only teaching episodes."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append new attempts; never continue an interrupted attempt",
    )
    args = parser.parse_args(argv)
    from .record_control import TerminalCommands
    from .console import operator_console

    terminal = TerminalCommands()
    cfg, values = load_config(args.config)
    if preparation_factory is not None:
        from .teach_prepare import normalize_preparation

        if "preparation" not in values:
            raise ValueError("teach-collect requires preparation configuration")
        values["preparation"] = normalize_preparation(values["preparation"])
    elif "preparation" in values:
        raise ValueError(
            "use teach-collect for confirmed preparation; teach-record is receive-only"
        )
    root = args.output
    if args.resume:
        old = read_json(root / "session.json")
        if old["schema"] != SCHEMA or old["source"] != "manual_teach" or old["config"] != values:
            raise ValueError("resume requires identical teach source, scene and configuration")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(
            root / "session.json",
            dict(schema=SCHEMA, source="manual_teach", config=values, created_at_utc=utc()),
        )
    run = root / f"connection-{uuid.uuid4().hex}"
    run.mkdir()
    package = Path(__file__).parent
    repo = package.parents[3]
    if (repo / ".git").exists():
        (run / "source-commit.txt").write_bytes(
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
        )
        (run / "source.patch").write_bytes(
            subprocess.check_output(["git", "diff", "--binary"], cwd=repo)
        )
    shutil.copytree(package, run / "source", ignore=shutil.ignore_patterns("__pycache__"))
    write_json(
        run / "invocation.json", dict(python=sys.executable, argv=sys.argv, started_at_utc=utc())
    )
    source = TeachSource(cfg)
    preparation = None
    result = dict(status="started", source="manual_teach")
    if preparation_factory is None:
        result["motor_commands_sent"] = False
    else:
        result.update(
            scope="confirmed preparation plus receive-only capture",
            sampling_motor_commands_sent=False,
        )
    try:
        if preparation_factory is not None:
            preparation = preparation_factory(cfg, values["preparation"], root)
        with operator_console(sys.argv[1:]):
            print(
                "[只接收示教] 不发送机械臂指令；Xbox 所有按键均不生效。停止采样不会停止机械臂。"
                if preparation is None
                else "[分阶段采集] prepare 确认后自动回工作姿态；录制/保存阶段只接收。Xbox按键不参与。",
                flush=True,
            )
            if preparation is None:
                source.connect()
            else:
                source.connect(require_teach=False)
            result["saved_attempts"] = run_session(
                source, values, root, terminal, connection_id=run.name, preparation=preparation
            )
            result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            source.disconnect()
        except Exception as exc:
            result.update(status="failed", cleanup_error=str(exc))
            raise
        finally:
            result.update(finished_at_utc=utc(), transmit_attempts=source.transmit_attempts)
            if preparation is not None:
                result.update(
                    preparation_reports=preparation.reports,
                    preparation_control_commands_attempted=preparation.control_commands_attempted,
                    transmit_attempts_scope="receive-only capture source",
                )
            write_json(run / "result.json", result)
            print(
                f"[会话结束] 原始记录：{root}；"
                + (
                    "未回零、失能或切换模式。"
                    if preparation is None
                    else "退出不自动回工作姿态或失能；准备操作见独立报告。"
                ),
                flush=True,
            )
    return 0


def collect_main(argv=None):
    from .teach_prepare import TeachPreparation

    return main(argv, preparation_factory=TeachPreparation)
