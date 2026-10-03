from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

from piper_policy.smolvla_gripper import SqrtGripperStep, make_processors
from piper_policy.statistics import numeric_stats


def test_sqrt_gripper_roundtrip_zero_and_full_opening_without_source_mutation():
    original = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7) / 100
    original[..., 6] = torch.tensor([[0, 0.001, 0.12], [0.01, 0.05, 0.1]])
    saved = original.clone()
    encoded = SqrtGripperStep()({"action": original})
    decoded = SqrtGripperStep(decode=True)(encoded)["action"]
    torch.testing.assert_close(decoded, original)
    torch.testing.assert_close(original, saved)
    assert encoded["action"][0, 0, 6] == 0
    torch.testing.assert_close(encoded["action"][..., :6], original[..., :6])
    # This is a learned representation, not clipping an old physical action.
    latent = torch.zeros(1, 7)
    latent[0, 6] = -0.5
    assert SqrtGripperStep(decode=True)({"action": latent})["action"][0, 6] == 0.25


def test_invalid_training_actions_fail_and_observation_only_input_is_valid():
    with pytest.raises(ValueError, match="negative source"):
        SqrtGripperStep()({"action": -torch.ones(1, 7)})
    with pytest.raises(ValueError, match="finite"):
        SqrtGripperStep(decode=True)({"action": torch.full((1, 7), torch.nan)})
    assert SqrtGripperStep()({"observation": {}}) == {"observation": {}}


def test_decoder_survives_serialization(tmp_path):
    pipeline = PolicyProcessorPipeline(
        steps=[SqrtGripperStep(decode=True)],
        name="policy_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    pipeline.save_pretrained(tmp_path)
    restored = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    actions = torch.ones(1, 50, 7) / 10
    torch.testing.assert_close(pipeline(actions), restored(actions))
    assert restored.steps[0].decode


def test_training_stats_are_encoded_without_changing_raw_labels(monkeypatch):
    import lerobot.policies.smolvla.processor_smolvla as upstream

    captures = []

    def factory(config, dataset_stats):
        captures.append(dataset_stats)
        return NS(steps=[]), NS(steps=[])

    monkeypatch.setattr(upstream, "make_smolvla_pre_post_processors", factory)
    raw = np.ones((3, 7), dtype=np.float32)
    raw[:, 6] = [0, 0.01, 0.09]
    training = NS(actions=raw, meta=NS(stats={"action": numeric_stats(raw)}))
    pre, post = make_processors(None, training, "sqrt_gripper")
    assert captures[0]["action"]["mean"][6] == pytest.approx((0 + 0.1 + 0.3) / 3)
    np.testing.assert_array_equal(training.actions[:, 6], raw[:, 6])
    assert not pre.steps[0].decode and post.steps[0].decode
