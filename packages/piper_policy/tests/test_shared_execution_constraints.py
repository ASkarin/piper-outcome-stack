import numpy as np
import pytest
import torch
from types import SimpleNamespace as NS

from piper_policy.execution_constraints import project_gripper, projection_stats
from piper_policy.async_chunks import ChunkResult, RTCTimeline
from piper_policy.rtc import PaperRTCProcessor


def test_projection_is_explicit_and_preserves_raw_and_other_dimensions():
    raw = np.zeros((3, 7))
    raw[:, 6] = [-0.0008, 0.05, 0.13]
    original = raw.copy()
    target = project_gripper(raw, "zero")
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(target[:, :6], raw[:, :6])
    np.testing.assert_array_equal(target[:, 6], [0, 0.05, 0.13])
    np.testing.assert_array_equal(project_gripper(raw, "strict"), raw)
    assert projection_stats(raw, target)["count"] == 1
    assert projection_stats(raw, target)["max_correction_mm"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="finite"):
        project_gripper(np.full(7, np.nan), "zero")


def test_rtc_prefix_is_committed_execution_representation_not_raw_prediction():
    raw = np.zeros((50, 7))
    raw[:, 6] = -0.001
    model = np.ones((50, 7)) * -2
    execute = project_gripper(raw, "zero")
    committed = np.ones((50, 7)) * 3
    q = RTCTimeline(initial_delay=10)
    q.schedule(
        ChunkResult(
            raw, 0, 0.2, model, execution_actions=execute, committed_model_actions=committed
        )
    )
    actual, _ = q.target(25)
    assert actual[6] == 0 and q.active.actions[0, 6] < 0
    _, _, options = q.capture_request(np.zeros(1), np.zeros(7), 25)
    np.testing.assert_array_equal(options["previous_model_actions"], committed[25:])
    np.testing.assert_array_equal(q.active.model_actions, model)


def test_saved_absolute_normalizer_encodes_zero_command():
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from lerobot.configs import FeatureType, PolicyFeature, NormalizationMode
    from piper_policy.smolvla import SmolVLAChunkPredictor

    features = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    stats = {"action": {"mean": torch.ones(7) * 0.05, "std": torch.ones(7) * 0.02}}
    options = dict(
        features=features, norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD}, stats=stats
    )
    normalizer = NormalizerProcessorStep(**options)
    inverse = UnnormalizerProcessorStep(**options)
    predictor = object.__new__(SmolVLAChunkPredictor)
    predictor.model = NS(config=NS(device="cpu"))
    predictor.preprocessor = NS(steps=[normalizer])
    physical = np.ones((50, 7), dtype=np.float32) * 0.06
    physical[:, 6] = 0
    encoded = predictor.encode_execution_actions(physical)
    restored = inverse({"action": torch.from_numpy(encoded)})["action"].numpy()
    np.testing.assert_allclose(restored, physical, atol=1e-7)
    assert np.all(encoded[:, 6] < 0)  # normalized zero is not a literal model-space zero


def test_identity_guidance_matches_installed_lerobot_on_real_dimensions():
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.configs import RTCAttentionSchedule

    cfg = RTCConfig(
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=5,
        execution_horizon=25,
    )
    upstream = RTCProcessor(cfg)
    adapted = PaperRTCProcessor(guidance="identity")
    x = torch.ones(1, 50, 7)
    prefix = torch.zeros(1, 25, 7)

    def denoise(z):
        return 0.5 * z

    torch.testing.assert_close(
        upstream.denoise_step(x, prefix, 12, 0.5, denoise, 25),
        adapted.denoise_step(x, prefix, 12, 0.5, denoise, 25),
    )
    assert (
        adapted.denoise_step(torch.ones(1, 2, 7), torch.zeros(1, 2, 7), 2, 0.5, denoise, 2)[0, 0, 0]
        == 2
    )
