"""Exercise the upstream Euler/RTC loop with coupled real and padded coordinates."""

from types import MethodType, SimpleNamespace as NS

import pytest
import torch
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

from piper_policy.rtc import PaperRTCProcessor
from piper_policy.smolvla import SmolVLAChunkPredictor


def coupled_predictor(noise, steps, guidance, dtype):
    flow = NS(
        config=NS(num_steps=steps, use_cache=True),
        rtc_processor=None,
        expected_noise=noise,
        seen=[],
    )
    if guidance == "native":
        flow.rtc_processor = RTCProcessor(RTCConfig(enabled=True, execution_horizon=25))
    elif guidance in ("identity", "full-vjp"):
        flow.rtc_processor = PaperRTCProcessor(guidance=guidance)
    flow._rtc_enabled = lambda: flow.rtc_processor is not None
    flow.embed_prefix = lambda *a, **kw: (
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1),
    )
    flow.vlm_with_expert = NS(forward=lambda **kw: ((None, None), None))

    def denoise(prefix_pad_masks, past_key_values, x_t, timestep):
        t = timestep[:, None, None]
        flow.seen.append((timestep.detach().clone(), x_t.detach().clone()))
        # A known zero real-action target is recovered exactly on the training
        # padding path. Off-path padding affects every real coordinate.
        off_path = (x_t[..., 7:] - t * flow.expected_noise[..., 7:]).mean(-1, keepdim=True)
        real_velocity = x_t[..., :7] / t + 8 * off_path
        return torch.cat((real_velocity, torch.zeros_like(x_t[..., 7:])), dim=-1).to(dtype)

    flow.denoise_step = denoise
    flow.sample_actions = MethodType(VLAFlowMatching.sample_actions, flow)

    def predict(batch, noise, **options):
        return flow.sample_actions(None, None, None, None, torch.zeros(1, 7), noise, **options)[
            ..., :7
        ]

    predictor = SmolVLAChunkPredictor.__new__(SmolVLAChunkPredictor)
    predictor.model = NS(
        model=flow, config=NS(action_feature=NS(shape=(7,))), predict_action_chunk=predict
    )
    return predictor, flow


@pytest.mark.parametrize("steps", [5, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_padding_correction_removes_coupled_error_without_changing_real_target(steps, dtype):
    noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(42))
    predictor, flow = coupled_predictor(noise, steps, None, dtype)
    original = flow.denoise_step
    with torch.no_grad():
        uncorrected = predictor.model.predict_action_chunk({}, noise=noise)
        flow.seen.clear()
        corrected = predictor._predict_with_action_padding({}, noise)
    assert uncorrected.abs().max() > 0.1
    torch.testing.assert_close(corrected, torch.zeros_like(corrected), atol=0.01, rtol=0)
    assert flow.denoise_step is original
    assert len(flow.seen) == steps
    for t, x in flow.seen:
        torch.testing.assert_close(x[..., 7:], t[:, None, None] * noise[..., 7:], atol=0, rtol=0)


@pytest.mark.parametrize("guidance", ["native", "identity", "full-vjp"])
def test_rtc_keeps_padding_path_and_seven_dimensional_prefix(guidance):
    noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(42))
    predictor, flow = coupled_predictor(noise, 10, guidance, torch.float32)
    prefix = torch.full((1, 25, 7), 0.25)
    before = prefix.clone()
    with torch.no_grad():
        result = predictor._predict_with_action_padding(
            {}, noise, prev_chunk_left_over=prefix, inference_delay=20, execution_horizon=25
        )
    assert result.shape == (1, 50, 7) and torch.isfinite(result).all()
    assert not result.requires_grad
    torch.testing.assert_close(prefix, before, atol=0, rtol=0)
    for t, x in flow.seen:
        torch.testing.assert_close(x[..., 7:], t[:, None, None] * noise[..., 7:], atol=0, rtol=0)
    if guidance != "full-vjp":
        assert (result[:, :20] - prefix[:, :20]).abs().mean() < prefix[:, :20].abs().mean()
        assert result[:, 25:].abs().max() < 1e-5


def test_each_request_uses_its_own_noise_and_restores_denoiser_after_failure():
    noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(42))
    predictor, flow = coupled_predictor(noise, 10, "native", torch.float32)
    original = flow.denoise_step
    predict = predictor.model.predict_action_chunk

    def fail(batch, **kw):
        predict(batch, **kw)
        raise RuntimeError("failed inference")

    predictor.model.predict_action_chunk = fail
    with pytest.raises(RuntimeError, match="failed inference"), torch.no_grad():
        predictor._predict_with_action_padding({}, noise)
    assert flow.denoise_step is original
    predictor.model.predict_action_chunk = predict
    next_noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(91))
    flow.expected_noise = next_noise
    flow.seen.clear()
    with torch.no_grad():
        result = predictor._predict_with_action_padding({}, next_noise)
    torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-5, rtol=0)
    for t, x in flow.seen:
        torch.testing.assert_close(
            x[..., 7:], t[:, None, None] * next_noise[..., 7:], atol=0, rtol=0
        )
    assert flow.denoise_step is original
