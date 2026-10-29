import pytest
import json
import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from piper_policy.reach_bc import ReachBCConfig, ReachBCPolicy, reach_processors


def config():
    return ReachBCConfig(
        device="cpu",
        push_to_hub=False,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        input_mean=[0.0] * 10,
        input_scale=[1.0] * 10,
        joint_mean=[0.1] * 6,
        joint_scale=[0.2] * 6,
    )


def batch():
    return {
        "observation.state": torch.zeros(3, 7),
        "observation.environment_state": torch.ones(3, 3) * 0.02,
        "action": torch.tensor([[0.1] * 6 + [0.001]] * 3),
    }


def test_loss_uses_real_labels_and_all_six_joints():
    policy = ReachBCPolicy(config())
    sample = batch()
    loss, _ = policy(sample)
    loss.backward()
    assert any(parameter.grad.abs().sum() > 0 for parameter in policy.parameters())
    assert policy(sample, reduction="none")[0].shape == (3,)
    sample["action"][:, 6] = 0.002
    with pytest.raises(ValueError, match="fixed gripper"):
        policy(sample)


def test_checkpoint_roundtrip_and_processor_physical_units(tmp_path):
    torch.manual_seed(42)
    policy = ReachBCPolicy(config())
    before = policy.select_action(batch())
    policy.save_pretrained(tmp_path)
    restored = ReachBCPolicy.from_pretrained(tmp_path, strict=True, local_files_only=True)
    torch.testing.assert_close(restored.select_action(batch()), before, rtol=0, atol=0)
    assert torch.all(before[:, 6] == torch.tensor(0.001))
    assert restored.predict_action_chunk(batch()).shape == (3, 1, 7)
    pre, post = reach_processors(policy_cfg=config(), pretrained_path=None)
    single = {key: value[0] for key, value in batch().items() if key != "action"}
    transformed = pre(single)
    torch.testing.assert_close(transformed["observation.state"], single["observation.state"][None])
    torch.testing.assert_close(post(before), before)


def test_rejects_wrong_goal_shape_nonfinite_and_bad_scales():
    policy = ReachBCPolicy(config())
    sample = batch()
    sample["observation.environment_state"] = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="goal XYZ3"):
        policy.select_action(sample)
    sample = batch()
    sample["observation.state"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        policy.select_action(sample)
    invalid = config()
    invalid.joint_scale[0] = 0
    with pytest.raises(ValueError, match="positive"):
        ReachBCPolicy(invalid)


def test_delta_loss_and_absolute_decoding_use_current_observation(tmp_path):
    recipe = config()
    recipe.joint_representation = "delta"
    recipe.joint_mean = [0.01] * 6
    recipe.joint_scale = [0.02] * 6
    policy = ReachBCPolicy(recipe)
    for parameter in policy.parameters():
        parameter.data.zero_()
    sample = batch()
    sample["observation.state"][:, :6] = 0.2
    sample["action"][:, :6] = 0.21
    original_state = sample["observation.state"].clone()
    loss, _ = policy(sample)
    assert loss.item() < 1e-10
    predicted = policy.select_action(sample)
    torch.testing.assert_close(predicted[:, :6], sample["action"][:, :6])
    torch.testing.assert_close(sample["observation.state"], original_state, rtol=0, atol=0)
    sample["observation.state"][:, :6] = 0.4
    torch.testing.assert_close(policy.select_action(sample)[:, :6], torch.full((3, 6), 0.41))
    policy.save_pretrained(tmp_path)
    restored = ReachBCPolicy.from_pretrained(tmp_path, strict=True, local_files_only=True)
    torch.testing.assert_close(
        restored.select_action(sample), policy.select_action(sample), rtol=0, atol=0
    )
    assert restored.config.joint_representation == "delta"
    policy.model[-1].bias.data.fill_(100.0)
    assert torch.all(policy.select_action(sample)[:, :6] > 2.0)


def test_existing_absolute_checkpoint_without_new_field_is_unchanged(tmp_path):
    policy = ReachBCPolicy(config())
    expected = policy.select_action(batch())
    policy.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    saved.pop("joint_representation")
    path.write_text(json.dumps(saved))
    restored = ReachBCPolicy.from_pretrained(tmp_path, strict=True, local_files_only=True)
    assert restored.config.joint_representation == "absolute"
    torch.testing.assert_close(restored.select_action(batch()), expected, rtol=0, atol=0)


def test_feedback_endpoint_labels_goal_only_and_fixed_state_gain(tmp_path):
    recipe = config()
    recipe.joint_representation = "feedback_endpoint"
    recipe.joint_mean = [0.5] * 6
    policy = ReachBCPolicy(recipe)
    for parameter in policy.parameters():
        parameter.data.zero_()
    sample = batch()
    sample["observation.state"][:, :6] = 0.2
    sample["action"][:, :6] = 0.26  # actual teacher label: .8 * .2 + .2 * .5
    assert policy(sample)[0].item() < 1e-10
    predicted = policy.select_action(sample)
    torch.testing.assert_close(predicted[:, :6], torch.full((3, 6), 0.26))
    assert policy.model[0].in_features == 3
    assert sum(p.numel() for p in policy.parameters()) == 17798
    endpoint_before = policy._joint_prediction(sample).detach().clone()
    sample["observation.state"] += 0.1
    torch.testing.assert_close(policy._joint_prediction(sample), endpoint_before, rtol=0, atol=0)
    torch.testing.assert_close(
        policy.select_action(sample)[:, :6] - predicted[:, :6], torch.full((3, 6), 0.08)
    )
    policy.save_pretrained(tmp_path)
    restored = ReachBCPolicy.from_pretrained(tmp_path, strict=True, local_files_only=True)
    torch.testing.assert_close(
        restored.select_action(sample), policy.select_action(sample), rtol=0, atol=0
    )
    # No clipping; existing execution checks remain responsible for rejecting large outputs.
    restored.model[-1].bias.data.fill_(100)
    assert (restored.select_action(sample)[:, :6] > 2).all()
