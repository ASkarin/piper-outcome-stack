import numpy as np
import pytest
import torch

from piper_policy.rtc import PaperRTCProcessor, prefix_weights
from piper_policy.async_chunks import ChunkResult, RTCTimeline


def test_full_vjp_matches_paper_instead_of_identity_jacobian():
    x = torch.ones(1, 2, 1)
    with torch.no_grad():
        actual = PaperRTCProcessor(action_dim=1).denoise_step(
            x, torch.zeros_like(x), 2, 0.5, lambda z: 0.5 * z, execution_horizon=2
        )
    torch.testing.assert_close(actual, torch.full_like(x, 1.625))
    assert not actual.requires_grad and x.grad is None


def test_nonlinear_vjp_matches_finite_difference_and_ignores_padding_targets():
    dtype = torch.float64
    x = torch.linspace(-0.4, 0.7, 12, dtype=dtype).reshape(1, 4, 3)
    matrix = torch.arange(144, dtype=dtype).reshape(12, 12) / 1000

    def velocity(z):
        return torch.sin(z.flatten() @ matrix).reshape_as(z)

    def clean(z):
        return z - 0.4 * velocity(z)

    target = torch.zeros_like(x)
    target[:, :3, :2] = 0.2
    weights = prefix_weights(1, 3, 4, device=x.device, dtype=dtype)
    mask = torch.zeros_like(x)
    mask[:, :, :2] = weights[None, :, None]
    err = (target - clean(x)) * mask
    expected_grad = torch.zeros_like(x)
    for i in range(x.numel()):
        dx = torch.zeros_like(x)
        dx.flatten()[i] = 1e-6
        expected_grad.flatten()[i] = (((clean(x + dx) - clean(x - dx)) / (2e-6)) * err).sum()
    strength = (0.4**2 + 0.6**2) / (0.4 * 0.6)
    expected = velocity(x) - strength * expected_grad
    actual = PaperRTCProcessor(action_dim=2).denoise_step(
        x, target[:, :3, :2], 1, 0.4, velocity, execution_horizon=3
    )
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)


def test_no_prefix_and_zero_mask_reduce_to_ordinary_velocity():
    x = torch.ones(1, 50, 32)
    p = PaperRTCProcessor()

    def f(z):
        return z * 0.3

    torch.testing.assert_close(p.denoise_step(x, None, 0, 1, f), f(x))
    torch.testing.assert_close(p.denoise_step(x, torch.empty(1, 0, 7), 0, 1, f, 0), f(x))


@pytest.mark.parametrize("delay,overlap", [(0, 25), (12, 30), (25, 25)])
def test_exponential_mask_matches_equation_five(delay, overlap):
    w = prefix_weights(delay, overlap, 50, device="cpu", dtype=torch.float64)
    assert torch.all(w[:delay] == 1) and torch.all(w[overlap:] == 0)
    for i in range(delay, overlap):
        c = (overlap - i) / (overlap - delay + 1)
        assert w[i] == pytest.approx(c * np.expm1(c) / np.expm1(1))
    assert torch.all(w[1:] <= w[:-1])


def test_guidance_does_not_update_parameters_or_accumulate_gradients():
    model = torch.nn.Linear(7, 7).requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.ones(1, 50, 7)
    for _ in range(3):
        x = PaperRTCProcessor().denoise_step(x, torch.zeros(1, 25, 7), 12, 0.5, model, 25)
        assert not x.requires_grad
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k])
    assert all(p.grad is None for p in model.parameters())


def result(tick, delay):
    model = np.arange(350).reshape(50, 7).astype(float)
    return ChunkResult(np.square(model), tick, delay * 0.02, model)


def test_rtc_snapshot_keeps_signed_model_prefix_and_delay_history_is_not_from_future():
    q = RTCTimeline(initial_delay=12)
    first = result(0, 14)
    first.model_actions[:, 6] *= -1
    q.schedule(first)
    assert list(q.delays) == [12]
    q.target(14)
    image, state = np.ones(2), np.ones(7)
    snap, observed, kw = q.capture_request(image, state, 25)
    assert kw["inference_delay"] == 14 and kw["overlap_steps"] == 25
    np.testing.assert_array_equal(kw["previous_model_actions"], first.model_actions[25:])
    image[:] = state[:] = 0
    assert snap[0] == observed[0] == 1
    q.schedule(result(25, 20))
    assert max(q.delays) == 14
    target, meta = q.target(45)
    assert meta["chunk_index"] == 20 and meta["observation_tick"] == 25
    assert max(q.delays) == 20


def test_rtc_latency_overrun_stops_on_queue_exhaustion():
    q = RTCTimeline(initial_delay=12)
    q.schedule(result(0, 12))
    q.target(12)
    q.schedule(result(25, 30))
    with pytest.raises(TimeoutError, match="exhausted"):
        q.target(50)
    with pytest.raises(ValueError, match="initial delay"):
        RTCTimeline(initial_delay=26)


def test_insufficient_overlap_and_missing_model_prefix_are_errors():
    q = RTCTimeline(initial_delay=20)
    q.schedule(result(0, 20))
    q.target(20)
    with pytest.raises(TimeoutError, match="insufficient"):
        q.capture_request(np.zeros(1), np.zeros(7), 31)
    with pytest.raises(ValueError, match="normalized"):
        q.schedule(ChunkResult(np.zeros((50, 7)), 31, 0.1))


def test_predictor_preserves_raw_model_actions_before_in_place_postprocessing():
    from types import SimpleNamespace as NS
    from piper_policy.smolvla import SmolVLAChunkPredictor, IMAGE_FEATURE

    feature = NS(shape=(7,))
    config = NS(
        type="smolvla",
        input_features={"observation.state": feature, IMAGE_FEATURE: NS(shape=(3, 480, 640))},
        output_features={"action": feature},
        action_feature=feature,
        adapt_to_pi_aloha=False,
        use_delta_joint_actions_aloha=False,
        empty_cameras=0,
        n_obs_steps=1,
        chunk_size=50,
        n_action_steps=25,
        use_amp=False,
    )
    output = torch.ones(1, 50, 7)
    output[:, :, 6] = -0.2

    def post(x):
        x[:, :, 6].square_()
        return x

    model = NS(config=config, reset=lambda: None, predict_action_chunk=lambda batch: output)
    p = SmolVLAChunkPredictor(model, lambda x: x, post, "task")
    result = p.predict_chunk(np.zeros((480, 640, 3), np.uint8), np.zeros(7))
    assert np.all(result.model_actions[:, 6] < 0) and np.all(result.actions[:, 6] > 0)
    assert torch.all(output[:, :, 6] < 0)
