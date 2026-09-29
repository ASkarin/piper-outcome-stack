"""Per-control-cycle wall/CPU spans; persisted with the existing frame telemetry."""

from contextlib import contextmanager
from functools import wraps
import threading
import time

_local = threading.local()


def reset():
    late = getattr(_local, "late_log", None)
    _local.values = {} if late is None else {"previous_cycle_post_enqueue_log": late}
    _local.late_log = None
    _local.published = False
    _local.preparation_pose = False


def snapshot():
    _local.published = True
    return {k: dict(v) for k, v in getattr(_local, "values", {}).items()}


@contextmanager
def span(name):
    start, cpu = time.perf_counter(), time.thread_time()
    try:
        yield
    finally:
        values = getattr(_local, "values", None)
        if values is not None:
            entry = values.setdefault(name, dict(wall_s=0.0, thread_cpu_s=0.0, calls=0))
            wall, thread_cpu = time.perf_counter() - start, time.thread_time() - cpu
            entry["wall_s"] += wall
            entry["thread_cpu_s"] += thread_cpu
            entry["calls"] += 1
            if name in ("log_output", "log_enqueue") and getattr(_local, "published", False):
                late = getattr(_local, "late_log", None)
                if late is None:
                    late = _local.late_log = dict(wall_s=0.0, thread_cpu_s=0.0, calls=0)
                late["wall_s"] += wall
                late["thread_cpu_s"] += thread_cpu
                late["calls"] += 1


def measured(name):
    def decorate(fn):
        @wraps(fn)
        def run(*args, **kwargs):
            with span(name):
                return fn(*args, **kwargs)

        return run

    return decorate


def mark_preparation_pose(value):
    _local.preparation_pose = value


def is_preparation_pose():
    return getattr(_local, "preparation_pose", False)
