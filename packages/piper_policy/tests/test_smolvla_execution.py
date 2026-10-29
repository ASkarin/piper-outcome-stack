from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_policy.async_chunks import ChunkResult
from piper_policy.smolvla_execution import AsyncSmolVLAExecutor, JointLowPassPlan
from piper_policy.execution_constraints import project_gripper


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Worker:
    def __init__(self, predict):
        self.future = None
        self.result = None
        self.requests = []
        self.closed = False

    def submit(self, image, state, tick, **kwargs):
        self.requests.append((image.copy(), state.copy(), tick, kwargs))
        self.future = object()

    def collect(self):
        if isinstance(self.result, Exception):
            raise self.result
        result, self.result = self.result, None
        if result is not None:
            self.future = None
        return result

    def close(self):
        self.closed = True


def rtc_result(actions, tick, seconds):
    return ChunkResult(
        actions,
        tick,
        seconds,
        actions + 10,
        execution_actions=project_gripper(actions, "zero"),
        committed_model_actions=project_gripper(actions, "zero") + 10,
    )


def executor(monkeypatch):
    monkeypatch.setattr("piper_policy.smolvla_execution.ChunkWorker", Worker)
    clock = Clock()
    model = NS(
        model=NS(reset=lambda: None),
        predict_chunk=lambda *a: None,
        configure_rtc=lambda *a, **kw: None,
        encode_execution_actions=lambda a: a + 10,
    )
    return AsyncSmolVLAExecutor(model, clock=clock), clock


def call(ex, clock, state=None, sequence=1):
    return ex.execution_chunk(
        np.zeros((2, 2, 3), np.uint8),
        np.zeros(7) if state is None else state,
        sequence,
        observed_monotonic_s=clock.now,
    )


def test_initial_hold_and_delayed_chunk_keep_original_anchor(monkeypatch):
    ex, clock = executor(monkeypatch)
    chunk, index, new = call(ex, clock)
    assert not new and index == 0 and not chunk.any()
    assert ex.execution_details["phase"] == "initial_hold"
    clock.now = 0.24
    actions = np.repeat(np.arange(50, dtype=np.float32)[:, None], 7, axis=1)
    ex.worker.result = rtc_result(actions, 0, 0.24)
    chunk, index, new = call(ex, clock, np.ones(7), 2)
    assert new and index == 12
    np.testing.assert_array_equal(chunk[index], actions[12])
    assert ex.generation_sequence == 1 and ex.generation_state == [0] * 7
    clock.now = 0.5
    chunk, index, new = call(ex, clock, np.ones(7), 3)
    assert index == 25 and not new and ex.worker.requests[-1][2] == 25
    clock.now = 0.74
    ex.worker.result = rtc_result(actions + 100, 25, 0.24)
    chunk, index, new = call(ex, clock, np.ones(7) * 2, 4)
    assert new and index == 12 and ex.generation_sequence == 3
    assert ex.generation_state == [1] * 7
    np.testing.assert_array_equal(chunk[index], actions[12] + 100)
    clock.now = 1.52
    with pytest.raises(TimeoutError, match="queue exhausted"):
        call(ex, clock)
    worker = ex.worker
    ex.reset_execution()
    assert worker.closed and ex.worker is None and ex.origin is None


def test_first_inference_timeout_and_exception_never_return_stale_action(monkeypatch):
    ex, clock = executor(monkeypatch)
    call(ex, clock)
    clock.now = 1.0
    with pytest.raises(TimeoutError, match="first inference"):
        call(ex, clock)
    ex.reset_execution()
    clock.now = 2
    call(ex, clock)
    ex.worker.result = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        call(ex, clock)


def test_expired_completed_chunk_is_rejected(monkeypatch):
    ex, clock = executor(monkeypatch)
    call(ex, clock)
    ex.worker.result = rtc_result(np.zeros((50, 7)), 0, 1.1)
    clock.now = 1.1
    with pytest.raises(TimeoutError, match="older than its action horizon"):
        call(ex, clock)


def test_negative_opening_is_zero_with_raw_prediction_preserved(monkeypatch):
    ex, clock = executor(monkeypatch)
    call(ex, clock)
    raw = np.arange(350, dtype=np.float32).reshape(50, 7)
    raw[:, 6] = -0.001
    original = raw.copy()
    ex.worker.result = rtc_result(raw, 0, 0.24)
    clock.now = 0.24
    chunk, index, new = call(ex, clock)
    target = ex.select_target(chunk, chunk_index=index, new_chunk=new)
    assert target[6] == 0
    assert np.any(target[:6] != original[index, :6])
    np.testing.assert_array_equal(target, ex.timeline.active.execution_actions[index])
    np.testing.assert_array_equal(
        target, ex.select_target(chunk, chunk_index=index, new_chunk=False)
    )
    np.testing.assert_array_equal(chunk, original)
    assert (
        ex.execution_details["target_transform"] == "butterworth2_5hz_joints_and_gripper_max_zero"
    )
    assert ex.execution_details["gripper_correction_m"] == pytest.approx(0.001)
    np.testing.assert_array_equal(ex.execution_details["raw_target_rad_m"], original[index])
    assert ex.execution_details["gripper_projection"] == "zero"
    assert "smoothing" not in ex.execution_details


def test_rtc_prefix_matches_projected_execution(monkeypatch):
    ex, clock = executor(monkeypatch)
    call(ex, clock)
    raw = np.ones((50, 7), np.float32) * 0.01
    raw[:, 6] = -0.001
    ex.worker.result = rtc_result(raw, 0, 0.24)
    clock.now = 0.24
    call(ex, clock)
    filtered = ex.timeline.active.execution_actions.copy()
    clock.now = 0.5
    call(ex, clock)
    options = ex.worker.requests[-1][3]
    np.testing.assert_array_equal(options["previous_model_actions"], (filtered + 10)[25:])
    ex.worker.result = rtc_result(raw, 25, 0.24)
    clock.now = 0.74
    call(ex, clock)
    record = ex.execution_details["rtc_request"]
    np.testing.assert_array_equal(record["prefix_execution_actions"], filtered[25:])
    assert record["guidance"] == "native_joint6"
    assert record["guided_action_indices"] == list(range(6))


def test_native_configuration_uses_upstream_processor():
    from piper_policy.smolvla import SmolVLAChunkPredictor
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor
    from lerobot.configs import RTCAttentionSchedule

    p = SmolVLAChunkPredictor.__new__(SmolVLAChunkPredictor)
    p.model = NS(config=NS(), model=NS())
    p.configure_rtc(True)
    assert type(p.model.rtc_processor) is RTCProcessor
    assert p.model.model.rtc_processor is p.model.rtc_processor
    assert p.model.config.rtc_config.max_guidance_weight == 10
    assert p.model.config.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR
    p.configure_rtc(False)
    assert p.model.rtc_processor is None


@pytest.mark.parametrize("guidance", ["native", "native_joint6"])
def test_native_predict_converts_prefix_to_tensor_without_modification(guidance):
    import torch
    from piper_policy.smolvla import SmolVLAChunkPredictor

    prefix = np.arange(175, dtype=np.float32).reshape(25, 7)

    def predict(batch, **kw):
        assert isinstance(kw["prev_chunk_left_over"], torch.Tensor)
        np.testing.assert_array_equal(kw["prev_chunk_left_over"].numpy(), prefix)
        return torch.zeros((1, 50, 7))

    p = SmolVLAChunkPredictor.__new__(SmolVLAChunkPredictor)
    p.model = NS(
        config=NS(
            device="cpu",
            use_amp=False,
            rtc_config=NS(enabled=True),
            max_action_dim=32,
            action_feature=NS(shape=(7,)),
        ),
        model=NS(denoise_step=None, sample_noise=lambda shape, device: torch.zeros(shape)),
        predict_action_chunk=predict,
    )
    p.rtc_guidance = guidance
    p.task = "test"
    p.preprocessor = lambda x: x
    p.postprocessor = lambda x: x
    value = p.predict_chunk(
        np.zeros((480, 640, 3), np.uint8),
        np.zeros(7),
        previous_model_actions=prefix,
        inference_delay=20,
        overlap_steps=25,
    )
    np.testing.assert_array_equal(value.actions, np.zeros((50, 7)))


def test_filter_cross_chunk_equals_one_causal_scipy_stream():
    from scipy.signal import butter, sosfilt, sosfilt_zi

    initial = np.arange(7, dtype=float) * 0.01
    raw1 = np.random.default_rng(7).normal(size=(50, 7)) * 0.02
    raw2 = np.random.default_rng(8).normal(size=(50, 7)) * 0.02
    plan = JointLowPassPlan(initial)
    first = plan.publish(raw1, 0, 12).copy()
    second = plan.publish(raw2, 25, 37).copy()
    source = np.concatenate([raw1[12:37, :6], raw2[12:, :6]])
    sos = butter(2, 5, fs=50, output="sos")
    expected, _ = sosfilt(
        sos, source, axis=0, zi=sosfilt_zi(sos)[:, :, None] * initial[None, None, :6]
    )
    np.testing.assert_allclose(
        np.concatenate([first[12:37, :6], second[12:, :6]]), expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(second[12:, 6], np.maximum(raw2[12:, 6], 0))
    with pytest.raises(TimeoutError, match="previous action horizon"):
        plan.publish(raw1, 75, 76)


def test_filter_holds_constant_and_does_not_silently_clip_joint_overshoot():
    q = np.array([0.1, 0.2, -0.3, 0.4, 0.1, 0.2, 0.02])
    plan = JointLowPassPlan(q)
    raw = np.repeat(q[None], 50, axis=0)
    result = plan.publish(raw, 0, 14)
    np.testing.assert_allclose(result[14:], raw[14:], atol=1e-12)
    step = raw.copy()
    step[:, 0] += 0.1
    result = plan.publish(step, 25, 39)
    assert result[14, 0] < step[14, 0]
    assert result[:, 0].max() > step[14, 0]
    # Existing downstream limits decide whether such a target can be dispatched.
    np.testing.assert_array_equal(result[:, 6], raw[:, 6])
