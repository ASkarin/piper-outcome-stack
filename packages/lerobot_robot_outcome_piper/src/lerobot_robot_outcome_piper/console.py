"""Keep operator prompts visible while preserving library INFO in a session log."""

from contextlib import contextmanager
import logging
import copy
import json
import queue
import threading
import time
import sys
from contextvars import ContextVar
from .errors import OutcomePiperLogError
import tempfile
from pathlib import Path
from .stage_timing import span


class OperatorInfoFilter(logging.Filter):
    def filter(self, record):
        if getattr(record, "piper_preparation_pose_slow", False):
            return False
        if record.levelno != logging.INFO:
            return True
        message = record.getMessage()
        return not (
            message.startswith(
                (
                    "Imported third-party plugin:",
                    "Using video codec:",
                    "Xbox 机械臂新输入 epoch=",
                    "Xbox 保持目标：",
                    "Xbox 保持状态：",
                    "Xbox 姿态到位详情：",
                    "Xbox 工作姿态参考点",
                    "Xbox 回零参考点",
                )
            )
            or message == "Created a socket"
            or (
                record.pathname.endswith("camera_realsense.py")
                and message.endswith((" connected.", " disconnected."))
            )
        )


class OperatorFormatter(logging.Formatter):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def format(self, record):
        if record.levelno == logging.INFO and (
            record.getMessage().startswith("[") or getattr(record, "operator_prompt", False)
        ):
            return record.getMessage()
        return self.original.format(record) if self.original is not None else super().format(record)


_active_output = ContextVar("piper_operator_output", default=None)


def check_operator_logging():
    output = _active_output.get()
    if output is not None:
        output.check_health()


def operator_message(message):
    if _active_output.get() is None:
        print(message, flush=True)
    else:
        logging.info("%s", message, extra={"operator_prompt": True})


class AsyncOperatorHandler(logging.Handler):
    """One bounded output worker. Producers never wait for terminal/file writes."""

    def __init__(self, handlers, capacity=1024):
        super().__init__()
        self.outputs = handlers
        self.records = queue.Queue(maxsize=capacity)
        self.failure = None
        self.finishing = False
        self.accepted = self.rejected = self.processed = self.high_water = 0
        self.sink_stats = {}
        self.slow = []
        self.failed_outputs = set()
        self.old_errors = [(h, h.handleError) for h in handlers]
        for handler in handlers:
            handler.handleError = self._sink_failure(handler)
        self.worker = threading.Thread(target=self._run, name="piper-operator-output", daemon=True)
        self.worker.start()

    def _sink_failure(self, handler):
        def failed(record):
            error = sys.exc_info()[1]
            self.failed_outputs.add(handler)
            if self.failure is None:
                self.failure = OutcomePiperLogError(f"operator log output failed: {error}")

        return failed

    def emit(self, record):
        # Never raise from logging: B/hold code may itself need to emit a message.
        with span("log_enqueue"):
            if self.failure is not None or self.finishing:
                self.rejected += 1
                if self.failure is None:
                    self.failure = OutcomePiperLogError(
                        "operator log submitted after output shutdown"
                    )
                return
            try:
                item = copy.copy(record)
                item.msg, item.args = record.getMessage(), ()
                from .stage_timing import is_preparation_pose

                item.piper_preparation_pose_slow = (
                    is_preparation_pose()
                    and record.levelno == logging.WARNING
                    and Path(record.pathname).name == "lerobot_record.py"
                    and item.msg.startswith("Record loop is running slower (")
                )
                self.records.put_nowait((time.monotonic(), item))
                self.accepted += 1
                self.high_water = max(self.high_water, self.records.qsize())
            except Exception as exc:
                self.rejected += 1
                self.failure = OutcomePiperLogError(
                    f"operator log queue failed: {type(exc).__name__}: {exc}"
                )

    def _run(self):
        while True:
            item = self.records.get()
            try:
                if item is None:
                    return
                enqueued, record = item
                for index, handler in enumerate(self.outputs):
                    if handler in self.failed_outputs or record.levelno < handler.level:
                        continue
                    kind = "file" if isinstance(handler, logging.FileHandler) else "console"
                    started, cpu = time.monotonic(), time.thread_time()
                    try:
                        handler.handle(record)
                    except Exception as exc:
                        self.failed_outputs.add(handler)
                        if self.failure is None:
                            self.failure = OutcomePiperLogError(
                                f"operator {kind} output failed: {exc}"
                            )
                    wall, thread_cpu = time.monotonic() - started, time.thread_time() - cpu
                    stats = self.sink_stats.setdefault(
                        f"{kind}_{index}", dict(calls=0, wall_s=0.0, cpu_s=0.0, max_wall_s=0.0)
                    )
                    stats.update(
                        calls=stats["calls"] + 1,
                        wall_s=stats["wall_s"] + wall,
                        cpu_s=stats["cpu_s"] + thread_cpu,
                        max_wall_s=max(stats["max_wall_s"], wall),
                    )
                    if wall > 0.01:
                        self.slow.append(
                            dict(
                                sink=kind,
                                started_monotonic_s=started,
                                wall_s=wall,
                                thread_cpu_s=thread_cpu,
                                queue_wait_s=started - enqueued,
                                message=record.getMessage(),
                            )
                        )
                        self.slow.sort(key=lambda x: x["wall_s"], reverse=True)
                        del self.slow[32:]
                self.processed += 1
            finally:
                self.records.task_done()

    def check_health(self):
        if self.failure is not None:
            raise self.failure

    def finish(self):
        # Called only after the workflow has released hardware; preserve pending output.
        self.acquire()
        try:
            self.finishing = True
        finally:
            self.release()
        self.records.put(None)
        self.worker.join()
        for handler, original in self.old_errors:
            handler.handleError = original
        return dict(
            accepted=self.accepted,
            processed=self.processed,
            rejected=self.rejected,
            max_queue_items=self.high_water,
            error=None if self.failure is None else str(self.failure),
            sink_timings=self.sink_stats,
            slowest_outputs=self.slow,
        )


def print_controls(config):
    operator_message("[手柄] LB按住操作/松开保持；LT闭合/RT张开；B急停。")
    operator_message("RB切换平移/姿态模式：松LB回中后即可单击；等待“模式已就绪”，再按LB操作。")
    operator_message(
        "默认平移：腕部优先保持，XYZ控制抓取中心；右摇杆左右无动作。X切换保持朝向（含yaw）。",
    )
    operator_message("X仅在平移模式、松LB回中且保持确认时切换；姿态模式绕抓取中心旋转。")
    keys = (
        "A进入工作姿态，Y回零" if getattr(config, "work_pose_button", None) is not None else "Y回零"
    )
    operator_message(f"{keys}：保持确认后按下并松开目标键，再按住LB执行；松LB或推杆取消。")


@contextmanager
def operator_console(arguments):
    from lerobot.utils.utils import init_logging

    root = logging.getLogger()
    if not root.handlers:
        init_logging()
    previous_level = root.level
    root.setLevel(min(previous_level, logging.INFO))
    # Configs may live in immutable releases; runtime logs belong to the operator.
    directory = Path.home() / "piper-runs" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix="operator-", suffix=".log", delete=False
    ) as file:
        log_path = Path(file.name)
    log = logging.FileHandler(log_path, encoding="utf-8")
    log.setLevel(logging.INFO)
    log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.stream.write(f"Arguments: {list(arguments)!r}\n")
    operator_message(f"运行日志：{log_path}")
    original_handlers = list(root.handlers)
    selected = [
        h
        for h in original_handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    filter_ = OperatorInfoFilter()
    formats = [(handler, handler.formatter) for handler in selected]
    for handler, original in formats:
        handler.addFilter(filter_)
        handler.setFormatter(OperatorFormatter(original))
    output = AsyncOperatorHandler([*original_handlers, log])
    root.handlers = [output]
    token = _active_output.set(output)
    primary = None
    try:
        yield log_path
    except BaseException as exc:
        primary = exc
        raise
    finally:
        cleanup_error = None
        try:
            summary = output.finish()
            log_path.with_suffix(".logging.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
        except Exception as exc:
            cleanup_error = exc
        finally:
            _active_output.reset(token)
            root.handlers = original_handlers
            for handler, original in formats:
                handler.removeFilter(filter_)
                handler.setFormatter(original)
            log.close()
            root.setLevel(previous_level)
        failure = output.failure or cleanup_error
        if failure is not None:
            if primary is not None:
                primary.add_note(f"Operator logging also failed: {failure}")
            else:
                raise failure
