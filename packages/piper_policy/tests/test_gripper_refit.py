from types import SimpleNamespace as NS

import numpy as np
import torch

from piper_policy.gripper_refit import GripperHeadRefiner
from piper_policy.positive_act import parameterize


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.action_head = torch.nn.Linear(3, 7)
        self.config = NS(
            gripper_state_mean=0.06,
            gripper_state_std=0.02,
            gripper_action_mean=-0.002,
            gripper_action_std=0.009,
            positive_floor_m=0.000001,
            positive_scale_m=0.001,
        )

    def save_pretrained(self, output):
        torch.save(self.state_dict(), output / "weights.pt")


class ToyPredictor:
    def __init__(self):
        self.model = ToyPolicy()

    def predict_batch(self, images, states):
        assert not self.model.training
        features = torch.stack([states[:, 0], states[:, 6], images.mean((1, 2, 3))], -1)
        features = features[:, None].repeat(1, 3, 1)
        with torch.no_grad():
            actions = self.model.model.action_head(features)
            normalized_state = states.clone()
            normalized_state[:, 6] = (states[:, 6] - 0.06) / 0.02
            actions = parameterize(actions, normalized_state, self.model.config)
            actions[..., 6] = actions[..., 6] * 0.009 - 0.002 + states[:, None, 6]
        return actions.numpy()


def make_refiner(tmp_path):
    torch.manual_seed(5)
    predictor = ToyPredictor()
    source = tmp_path / "source"
    source.mkdir()
    (source / "policy_preprocessor.json").write_text("{}")
    (source / "representation.json").write_text("{}")
    refiner = GripperHeadRefiner(predictor, source)
    images = torch.full((2, 3, 2, 2), 0.3)
    states = torch.zeros(2, 7)
    states[:, 6] = torch.tensor([0.05, 0.085])
    return predictor, refiner, images, states


def test_cached_decode_matches_original_and_masks_padding(tmp_path):
    predictor, refiner, images, states = make_refiner(tmp_path)
    features, prediction = refiner.extract_features(images, states)
    actual = refiner.predict_gripper(features, states)
    np.testing.assert_allclose(actual.detach(), prediction[..., 6], atol=1e-8, rtol=0)
    valid = torch.ones_like(actual, dtype=torch.bool)
    valid[0, -1] = False
    labels = actual.detach() + 0.004
    expected = refiner.loss(features, states, labels, valid)
    labels[0, -1] = 999
    torch.testing.assert_close(refiner.loss(features, states, labels, valid), expected)
    assert not any(p.requires_grad for p in predictor.model.parameters())


def test_fitting_and_save_change_only_gripper_row_and_restore_parent(tmp_path):
    predictor, refiner, images, states = make_refiner(tmp_path)
    before = {k: v.clone() for k, v in predictor.model.state_dict().items()}
    features, prediction = refiner.extract_features(images, states)
    labels = torch.from_numpy(prediction[..., 6]).clone() + 0.004
    valid = torch.ones_like(labels, dtype=torch.bool)
    initial = float(refiner.loss(features, states, labels, valid).detach())
    history = refiner.fit(
        features, states, labels, valid, [0, 1], steps=10, batch_size=2, learning_rate=0.01
    )
    assert history[-1]["step"] == 10
    assert float(refiner.loss(features, states, labels, valid).detach()) < initial
    assert all(p.grad is None for p in predictor.model.parameters())
    out = tmp_path / "refit"
    refiner.save(out)
    saved = torch.load(out / "weights.pt", weights_only=True)
    for key, value in before.items():
        torch.testing.assert_close(predictor.model.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(saved[key][:6], value[:6], rtol=0, atol=0)
        assert not torch.equal(saved[key][6], value[6])
    assert (out / "policy_preprocessor.json").read_bytes() == b"{}"
    assert not (out / "training_state").exists()
    np.testing.assert_array_equal(predictor.predict_batch(images, states), prediction)


def test_ridge_recovers_linear_gripper_mapping_without_fitting_padding(tmp_path):
    predictor, refiner, _, _ = make_refiner(tmp_path)
    torch.manual_seed(13)
    features = torch.randn(12, 5, 3)
    states = torch.zeros(12, 7)
    states[:, 6] = 0.07
    valid = torch.ones(12, 5, dtype=torch.bool)
    valid[0, -1] = False
    original_weight, original_bias = refiner.weight.detach().clone(), refiner.bias.detach().clone()
    with torch.no_grad():
        refiner.weight.add_(torch.tensor([0.04, -0.02, 0.03]))
        refiner.bias.add_(0.08)
        truth = refiner.predict_gripper(features, states).clone()
        refiner.weight.copy_(original_weight)
        refiner.bias.copy_(original_bias)
    truth[0, -1] = -99  # padding is excluded before inverse positivity
    before = float(refiner.loss(features, states, truth, valid).detach())
    result = refiner.fit_ridge(
        features, states, truth, valid, list(range(12)), relative_regularization=1e-8
    )
    after = float(refiner.loss(features, states, truth, valid).detach())
    assert result["solver"] == "regularized_linear_solve"
    assert after < before * 0.001
    assert after < 1e-5
    torch.testing.assert_close(
        predictor.model.model.action_head.weight[6], original_weight, rtol=0, atol=0
    )


def test_bias_floor_uses_training_only_and_preserves_all_weights(tmp_path):
    predictor, refiner, _, _ = make_refiner(tmp_path)
    features = torch.zeros(3, 2, 3)
    states = torch.zeros(3, 7)
    states[:, 6] = torch.tensor([0.03, 0.035, 0.00001])
    truth = torch.full((3, 2), 0.04)
    valid = torch.ones_like(truth, dtype=torch.bool)
    valid[0, 1] = False
    original_weight = refiner.weight.detach().clone()
    original_policy = {key: value.clone() for key, value in predictor.model.state_dict().items()}
    before = refiner.predict_gripper(features, states)
    result = refiner.calibrate_bias_floor(features, states, truth, valid, [0, 1])
    after = refiner.predict_gripper(features, states)
    assert before[:2].min() < 0.035
    assert after[:2].min() >= 0.035 - 1e-8
    assert after[2].max() < 0.035  # the held-out low input did not set the calibration
    assert result["eligible_training_entries"] == 3
    assert result["minimum_headspace_margin"] >= 0
    assert result["normalized_bias_shift"] > 0
    torch.testing.assert_close(refiner.weight, original_weight, rtol=0, atol=0)
    for key, value in original_policy.items():
        torch.testing.assert_close(predictor.model.state_dict()[key], value, rtol=0, atol=0)


def test_bias_floor_leaves_satisfied_model_unchanged(tmp_path):
    _, refiner, _, _ = make_refiner(tmp_path)
    features = torch.zeros(1, 2, 3)
    states = torch.zeros(1, 7)
    states[:, 6] = 0.085
    before = refiner.bias.detach().clone()
    result = refiner.calibrate_bias_floor(
        features, states, torch.full((1, 2), 0.085), torch.ones(1, 2, dtype=torch.bool), [0]
    )
    assert result["normalized_bias_shift"] == 0
    torch.testing.assert_close(refiner.bias, before, rtol=0, atol=0)


def test_joint_refit_preserves_gripper_and_parent_checkpoint(tmp_path):
    from safetensors.torch import save_file
    from piper_policy.joint_refit import JointHeadRefiner

    predictor, _, _, _ = make_refiner(tmp_path)
    source = tmp_path / "source"
    mean = torch.tensor([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, -0.002])
    std = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.009])
    save_file(
        {"action.mean": mean, "action.std": std},
        str(source / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"),
    )
    refiner = JointHeadRefiner(predictor, source)
    torch.manual_seed(9)
    features = torch.randn(12, 5, 3)
    states = torch.randn(12, 7) * 0.1
    valid = torch.ones(12, 5, dtype=torch.bool)
    original = {k: v.clone() for k, v in predictor.model.state_dict().items()}
    expected = (
        torch.nn.functional.linear(
            features,
            original["model.action_head.weight"][:6],
            original["model.action_head.bias"][:6],
        )
        * std[:6]
        + mean[:6]
        + states[:, None, :6]
    )
    torch.testing.assert_close(refiner.predict_joints(features, states), expected)
    original_weight, original_bias = refiner.weight.detach().clone(), refiner.bias.detach().clone()
    with torch.no_grad():
        refiner.weight.add_(torch.randn_like(refiner.weight) * 0.03)
        refiner.bias.add_(0.02)
        truth = refiner.predict_joints(features, states).clone()
        refiner.weight.copy_(original_weight)
        refiner.bias.copy_(original_bias)
    refiner.fit_ridge(features, states, truth, valid, list(range(12)), relative_regularization=1e-8)
    torch.testing.assert_close(refiner.predict_joints(features, states), truth, atol=1e-6, rtol=0)
    output = tmp_path / "joint-refit"
    refiner.save(output)
    saved = torch.load(output / "weights.pt", weights_only=True)
    for key, value in original.items():
        torch.testing.assert_close(saved[key][6], value[6], atol=0, rtol=0)
        torch.testing.assert_close(predictor.model.state_dict()[key], value, atol=0, rtol=0)
        assert not torch.equal(saved[key][:6], value[:6])
    assert (output / "joint_refit.json").is_file()
    assert not (output / "gripper_refit.json").exists()


def test_local_floor_preserves_unaffected_group_and_excludes_validation(tmp_path):
    predictor, refiner, _, _ = make_refiner(tmp_path)
    features = torch.zeros(3, 2, 3)
    features[0, :, 0] = 1
    features[1, :, 0] = -1
    features[2, :, 0] = 100  # held-out input/label must not enter either objective
    states = torch.zeros(3, 7)
    states[:, 6] = torch.tensor([0.03, 0.08, 0.00001])
    truth = torch.full((3, 2), 0.04)
    valid = torch.ones(3, 2, dtype=torch.bool)
    with torch.no_grad():
        refiner.weight.zero_()
        refiner.bias.zero_()
    original = {k: v.clone() for k, v in predictor.model.state_dict().items()}
    before = refiner.predict_gripper(features, states).detach()
    result = refiner.calibrate_local_floor(
        features, states, truth, valid, [0, 1], prediction_floor_m=0.035001
    )
    after = refiner.predict_gripper(features, states).detach()
    assert after[0].min() >= 0.035
    assert abs(after[1] - before[1]).max() < 0.0001  # global bias would shift ~7mm
    assert result["active_constraints"] == 1
    assert result["eligible_training_entries"] == 4
    for k, v in original.items():
        torch.testing.assert_close(predictor.model.state_dict()[k], v, rtol=0, atol=0)


def test_local_floor_solves_two_active_constraints_and_fails_before_mutation(tmp_path):
    import pytest

    _, refiner, _, _ = make_refiner(tmp_path)
    features = torch.zeros(2, 1, 3)
    features[0, 0, 0], features[1, 0, 0] = 1, -1
    states = torch.zeros(2, 7)
    states[:, 6] = 0.03
    truth, valid = torch.full((2, 1), 0.04), torch.ones(2, 1, dtype=torch.bool)
    with torch.no_grad():
        refiner.weight.zero_()
        refiner.bias.zero_()
    w, b = refiner.weight.clone(), refiner.bias.clone()
    with pytest.raises(RuntimeError, match="budget"):
        refiner.calibrate_local_floor(features, states, truth, valid, [0, 1], max_constraints=1)
    torch.testing.assert_close(refiner.weight, w, rtol=0, atol=0)
    torch.testing.assert_close(refiner.bias, b, rtol=0, atol=0)
    r = refiner.calibrate_local_floor(
        features, states, truth, valid, [0, 1], prediction_floor_m=0.035001
    )
    assert r["active_constraints"] == 2
    assert refiner.predict_gripper(features, states).min() >= 0.035
    assert abs(float(refiner.weight[0].detach())) < 1e-6


def test_joint_temporal_design_matches_official_te_and_keeps_old_anchors(tmp_path):
    from safetensors.torch import save_file
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
    from piper_policy.joint_refit import JointHeadRefiner

    predictor, _, _, _ = make_refiner(tmp_path)
    source = tmp_path / "source"
    mean = torch.zeros(7)
    std = torch.ones(7) * 0.1
    save_file(
        {"action.mean": mean, "action.std": std},
        str(source / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"),
    )
    refiner = JointHeadRefiner(predictor, source)
    torch.manual_seed(21)
    features = torch.randn(70, 50, 3) * 0.1
    states = torch.randn(70, 7) * 0.05
    chunks = refiner.predict_joints(features, states).detach()
    ensemble = ACTTemporalEnsembler(0.01, 50)
    reference = torch.stack([ensemble.update(chunk[None])[0] for chunk in chunks])
    design, anchors = refiner.temporal_design(features, states, 0.01)
    predicted = refiner.predict_temporal_joints(design, anchors)
    torch.testing.assert_close(predicted, reference.double(), atol=1e-6, rtol=0)
    expected_anchor = (states[0, :6].double() * np.exp(0.01) + states[1, :6].double()) / (
        1 + np.exp(0.01)
    )
    torch.testing.assert_close(anchors[1], expected_anchor, atol=1e-12, rtol=0)
    original_weight = refiner.weight.detach().clone()
    original_bias = refiner.bias.detach().clone()
    with torch.no_grad():
        refiner.weight.add_(0.015)
        refiner.bias.add_(0.01)
        truth = refiner.predict_temporal_joints(design, anchors).clone()
        refiner.weight.copy_(original_weight)
        refiner.bias.copy_(original_bias)
    refiner.fit_temporal_ridge(
        design, anchors, truth, list(range(70)), relative_regularization=1e-8
    )
    torch.testing.assert_close(
        refiner.predict_temporal_joints(design, anchors), truth, atol=1e-6, rtol=0
    )
