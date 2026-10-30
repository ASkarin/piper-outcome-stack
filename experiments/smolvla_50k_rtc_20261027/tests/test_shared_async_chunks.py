import threading

import numpy as np
import pytest

from piper_policy.async_chunks import ChunkResult, ChunkTimeline, ChunkWorker


def chunk(offset=0):
    return np.repeat(np.arange(50)[:, None], 7, axis=1).astype(float) + offset


def test_delayed_initial_result_skips_elapsed_actions_and_keeps_old_queue_until_replacement():
    queue = ChunkTimeline(request_stride=10)
    queue.schedule(ChunkResult(chunk(), 0, 0.305))
    assert queue.target(15)[0] is None
    target, meta = queue.target(16)
    assert target[0] == 16 and meta["chunk_index"] == 16
    assert queue.needs_request(16)
    queue.schedule(ChunkResult(chunk(100), 16, 0.305))
    assert queue.target(31)[0][0] == 31
    target, meta = queue.target(32)
    assert target[0] == 116 and meta["observation_tick"] == 16


def test_queue_exhaustion_and_expired_first_result_stop_instead_of_reusing_state():
    queue = ChunkTimeline()
    queue.schedule(ChunkResult(chunk(), 0, 0.02))
    queue.target(1)
    queue.schedule(ChunkResult(chunk(), 10, 1.2))
    with pytest.raises(TimeoutError, match="exhausted"):
        queue.target(50)
    with pytest.raises(TimeoutError, match="older"):
        queue.target(70)


def test_invalid_chunks_and_duplicate_submission_fail():
    queue = ChunkTimeline()
    with pytest.raises(ValueError, match="invalid asynchronous"):
        queue.schedule(ChunkResult(np.full((50, 7), np.nan), 0, 0.1))
    queue.schedule(ChunkResult(chunk(), 0, 0.1))
    with pytest.raises(RuntimeError, match="pending"):
        queue.schedule(ChunkResult(chunk(), 1, 0.1))


def test_worker_is_nonblocking_and_freezes_observation():
    ready = threading.Event()

    def predict(image, state):
        assert ready.wait(2)
        return np.ones((50, 7)) * (image[0] + state[0])

    worker = ChunkWorker(predict)
    image, state = np.array([1.0]), np.array([2.0])
    try:
        worker.submit(image, state, 7)
        image[0] = state[0] = 100
        assert worker.collect() is None
        with pytest.raises(RuntimeError, match="in flight"):
            worker.submit(image, state, 8)
        ready.set()
        result = worker.collect(block=True)
        assert result.observation_tick == 7 and np.all(result.actions == 3)
    finally:
        ready.set()
        worker.close()


def test_worker_surfaces_model_failure():
    def failed(image, state):
        raise ValueError("prediction failed")

    worker = ChunkWorker(failed)
    try:
        worker.submit(np.zeros(1), np.zeros(1), 0)
        with pytest.raises(ValueError, match="prediction failed"):
            worker.collect(block=True)
    finally:
        worker.close()
