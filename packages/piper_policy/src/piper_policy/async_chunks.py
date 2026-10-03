"""Timestamped chunk execution; late actions are skipped, never replayed as new."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import deque
from threading import RLock
import math
import time

import numpy as np


@dataclass
class ActionChunk:
    actions: np.ndarray
    model_actions: np.ndarray
    noise: np.ndarray | None = None
    execution_actions: np.ndarray | None = None
    committed_model_actions: np.ndarray | None = None


@dataclass
class ChunkResult:
    actions: np.ndarray
    observation_tick: int
    inference_seconds: float
    model_actions: np.ndarray | None = None
    noise: np.ndarray | None = None
    execution_actions: np.ndarray | None = None
    committed_model_actions: np.ndarray | None = None


class ChunkWorker:
    """One prediction in flight; observations are copied at submission."""

    def __init__(self, predict):
        self.predict = predict
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-inference")
        self.future = None

    def submit(self, image, state, tick, **kwargs):
        if self.future is not None:
            raise RuntimeError("a prediction is already in flight")
        started = time.perf_counter()
        image, state = image.copy(), state.copy()

        def infer():
            actions = self.predict(image, state, **kwargs)
            # predict returns a CPU array, so its CUDA-to-CPU copy is complete.
            elapsed = time.perf_counter() - started
            if isinstance(actions, ActionChunk):
                return ChunkResult(
                    actions.actions,
                    tick,
                    elapsed,
                    actions.model_actions,
                    actions.noise,
                    actions.execution_actions,
                    actions.committed_model_actions,
                )
            return ChunkResult(actions, tick, elapsed)

        self.future = self.executor.submit(infer)

    def collect(self, *, block=False):
        if self.future is None or (not block and not self.future.done()):
            return None
        try:
            return self.future.result()
        finally:
            self.future = None

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class ChunkTimeline:
    """Discrete control ticks refer to the observation's original time origin."""

    def __init__(self, *, period=0.02, request_stride=10):
        if not math.isfinite(period) or period <= 0 or not 1 <= request_stride < 50:
            raise ValueError("invalid chunk timing")
        self.period, self.request_stride = period, request_stride
        self.active = self.pending = None
        self.ready_tick = None
        self.last_request_tick = None
        self.events = []

    def needs_request(self, tick):
        return self.pending is None and (
            self.last_request_tick is None or tick - self.last_request_tick >= self.request_stride
        )

    def schedule(self, result):
        if self.pending is not None:
            raise RuntimeError("a chunk is already pending")
        actions = np.asarray(result.actions)
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError("invalid asynchronous action chunk")
        if not math.isfinite(result.inference_seconds) or result.inference_seconds < 0:
            raise ValueError("invalid inference duration")
        if self.last_request_tick is not None and result.observation_tick <= self.last_request_tick:
            raise ValueError("observation ticks must increase")
        self.pending = ChunkResult(
            actions.copy(),
            result.observation_tick,
            result.inference_seconds,
            None if result.model_actions is None else result.model_actions.copy(),
            None if result.noise is None else result.noise.copy(),
            None if result.execution_actions is None else result.execution_actions.copy(),
            None
            if result.committed_model_actions is None
            else result.committed_model_actions.copy(),
        )
        self.last_request_tick = result.observation_tick
        self.ready_tick = result.observation_tick + math.ceil(
            math.nextafter(result.inference_seconds / self.period, -math.inf)
        )
        self.events.append(
            {
                "observation_tick": result.observation_tick,
                "inference_seconds": result.inference_seconds,
                "ready_tick": self.ready_tick,
            }
        )

    def target(self, tick):
        if self.pending is not None and tick >= self.ready_tick:
            if tick - self.pending.observation_tick >= 50:
                raise TimeoutError("inference result is older than its action horizon")
            self.active, self.pending = self.pending, None
        if self.active is None:
            return None, {"phase": "startup_wait"}
        index = tick - self.active.observation_tick
        if not 0 <= index < 50:
            raise TimeoutError("action queue exhausted before a fresh chunk arrived")
        commands = (
            self.active.execution_actions
            if self.active.execution_actions is not None
            else self.active.actions
        )
        return commands[index].copy(), {
            "phase": "policy",
            "observation_tick": self.active.observation_tick,
            "chunk_index": index,
            "action_age_s": index * self.period,
        }


class RTCTimeline(ChunkTimeline):
    """Algorithm 1 snapshot and delay history; publication uses actual elapsed ticks."""

    def __init__(self, *, initial_delay, period=0.02, request_stride=25):
        super().__init__(period=period, request_stride=request_stride)
        if not 0 <= initial_delay <= 50 - request_stride:
            raise ValueError("RTC initial delay does not fit the prediction horizon")
        self.delays = deque([initial_delay], maxlen=10)
        self.lock = RLock()

    def capture_request(self, image, state, tick):
        with self.lock:
            if not self.needs_request(tick):
                raise RuntimeError("RTC inference is not due")
            prefix, overlap = None, None
            delay = max(self.delays)
            if self.active is not None:
                consumed = tick - self.active.observation_tick
                overlap = 50 - consumed
                if not delay <= consumed <= 50 - delay:
                    raise TimeoutError("RTC delay and execution horizon leave insufficient overlap")
                if self.active.model_actions is None:
                    raise ValueError(
                        "RTC requires original normalized actions, not decoded physical actions"
                    )
                model_prefix = (
                    self.active.committed_model_actions
                    if self.active.committed_model_actions is not None
                    else self.active.model_actions
                )
                prefix = model_prefix[consumed:].copy()
            return (
                image.copy(),
                state.copy(),
                {
                    "previous_model_actions": prefix,
                    "inference_delay": delay,
                    "overlap_steps": overlap,
                },
            )

    def schedule(self, result):
        with self.lock:
            if result.model_actions is None or result.model_actions.shape != (50, 7):
                raise ValueError("RTC result must preserve the normalized action chunk")
            super().schedule(result)

    def target(self, tick):
        with self.lock:
            if self.pending is not None and tick >= self.ready_tick:
                self.delays.append(self.ready_tick - self.pending.observation_tick)
            return super().target(tick)
