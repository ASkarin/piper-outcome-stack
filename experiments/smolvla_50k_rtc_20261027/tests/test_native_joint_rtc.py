import pytest
import torch
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor

from piper_policy.native_joint_rtc import NativeJointRTCProcessor


@pytest.mark.parametrize("time", [1.0, 0.5, 0.1])
def test_native_joints_unchanged_and_gripper_prefix_has_no_direct_effect(time):
    config = RTCConfig(enabled=True, execution_horizon=25)
    native, joints = RTCProcessor(config), NativeJointRTCProcessor(config)
    x = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(42))
    prefix = torch.randn(1, 25, 7, generator=torch.Generator().manual_seed(91))

    def velocity(value):
        return torch.sin(value) * 0.3

    with torch.no_grad():
        baseline = native.denoise_step(x, prefix, 14, time, velocity, 25)
        actual = joints.denoise_step(x, prefix, 14, time, velocity, 25)
        other_prefix = prefix.clone()
        other_prefix[..., 6] += 2
        changed = joints.denoise_step(x, other_prefix, 14, time, velocity, 25)
    torch.testing.assert_close(actual[..., :6], baseline[..., :6], atol=0, rtol=0)
    torch.testing.assert_close(actual[..., 7:], baseline[..., 7:], atol=0, rtol=0)
    torch.testing.assert_close(actual[..., 6], velocity(x)[..., 6], atol=0, rtol=0)
    torch.testing.assert_close(actual, changed, atol=0, rtol=0)
    assert not torch.equal(actual[..., 6], baseline[..., 6])


def test_first_chunk_and_predictor_configuration():
    from types import SimpleNamespace as NS
    from piper_policy.smolvla import SmolVLAChunkPredictor

    p = SmolVLAChunkPredictor.__new__(SmolVLAChunkPredictor)
    p.model = NS(config=NS(), model=NS())
    p.configure_rtc(True, guidance="native_joint6")
    assert isinstance(p.model.rtc_processor, NativeJointRTCProcessor)
    assert p.model.rtc_processor is p.model.model.rtc_processor
    x = torch.randn(1, 50, 32)
    with torch.no_grad():
        actual = p.model.rtc_processor.denoise_step(x, None, 14, 1.0, torch.sin)
    torch.testing.assert_close(actual, torch.sin(x), atol=0, rtol=0)
