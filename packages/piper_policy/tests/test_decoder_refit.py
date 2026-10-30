from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from lerobot.policies.act.modeling_act import ACTDecoder
from piper_policy.decoder_refit import ACTDecoderRefiner
from piper_policy.positive_act import PositiveACTConfig


def make_refiner(tmp_path):
    torch.manual_seed(7)
    cfg = PositiveACTConfig(
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_decoder_layers=1,
        chunk_size=3,
        n_action_steps=1,
        gripper_state_mean=0.06,
        gripper_state_std=0.02,
        gripper_action_mean=0.0,
        gripper_action_std=0.01,
    )
    policy = torch.nn.Module()
    policy.config = cfg
    policy.model = torch.nn.Module()
    policy.model.decoder = ACTDecoder(cfg)
    policy.model.action_head = torch.nn.Linear(16, 7)
    policy.model.decoder_pos_embed = torch.nn.Embedding(3, 16)
    policy.model.encoder = torch.nn.Linear(16, 16)
    save_file(
        {"action.mean": torch.zeros(7), "action.std": torch.tensor([0.1] * 6 + [0.01])},
        tmp_path / "policy_postprocessor_unnormalizer.safetensors",
    )
    ref = ACTDecoderRefiner(SimpleNamespace(model=policy), tmp_path)
    memory = torch.randn(2, 5, 16)
    pos = torch.randn(2, 5, 16)
    state = torch.zeros(2, 7)
    state[:, 6] = 0.06
    return ref, memory, pos, state


def test_padding_and_new_corrections_do_not_enter_consistency(tmp_path):
    ref, m, p, s = make_refiner(tmp_path)
    prediction = ref.predict(m, p, s).detach()
    truth = prediction + 0.001
    valid = torch.ones(2, 3, dtype=torch.bool)
    valid[0, 2] = False
    old = torch.tensor([True, False])
    expected = ref.loss(m, p, s, truth, valid, prediction, old)
    truth[0, 2] = 999
    parent = prediction.clone()
    parent[1] = 999
    parent[0, 2] = 999
    actual = ref.loss(m, p, s, truth, valid, parent, old)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert float(actual[2].detach()) < 1e-6


def test_decoder_updates_leave_encoder_and_query_positions_frozen(tmp_path):
    ref, m, p, s = make_refiner(tmp_path)
    snapshot = {k: v.clone() for k, v in ref.policy.state_dict().items()}
    target = ref.predict(m, p, s).detach() + 0.002
    optimizer = torch.optim.AdamW([v for v in ref.policy.parameters() if v.requires_grad], lr=1e-3)
    loss, _, _ = ref.loss(
        m,
        p,
        s,
        target,
        torch.ones(2, 3, dtype=torch.bool),
        target,
        torch.zeros(2, dtype=torch.bool),
    )
    loss.backward()
    optimizer.step()
    changed = {k for k, v in ref.policy.state_dict().items() if not torch.equal(v, snapshot[k])}
    assert changed and changed <= set(ref.trainable_names())
    assert all(v.grad is None for v in ref.policy.parameters() if not v.requires_grad)
    assert not ref.policy.training and not ref.model.decoder.training


def test_capture_expands_act_shared_positions_without_changing_values(tmp_path):
    ref, memory, position, states = make_refiner(tmp_path)
    shared = position[:1].transpose(0, 1)

    def inference(images, observed):
        with torch.inference_mode():
            output = ref.model.decoder(
                torch.zeros(3, 2, 16),
                memory.transpose(0, 1),
                encoder_pos_embed=shared,
                decoder_pos_embed=ref.model.decoder_pos_embed.weight.unsqueeze(1),
            )
        return output.transpose(0, 1).numpy()

    ref.predictor.predict_batch = inference
    cached, positions, _ = ref.extract_context(torch.zeros(2, 3, 2, 2), states)
    torch.testing.assert_close(cached, memory, rtol=0, atol=0)
    torch.testing.assert_close(positions, position[:1].expand_as(memory), rtol=0, atol=0)
    assert not cached.is_inference() and not positions.is_inference()


def test_execution_loss_masks_padding_and_keeps_parent_reference_old_only(tmp_path):
    ref, m, p, s = make_refiner(tmp_path)
    prediction = ref.predict(m, p, s).detach()
    truth = prediction.clone()
    truth[..., 6] = 0.04
    valid = torch.tensor([[True, True, False], [True, True, True]])
    old = torch.tensor([True, False])
    expected, parts = ref.execution_loss(m, p, s, truth, valid, prediction, old, front_steps=2)
    truth[0, 2] = float("nan")
    parent = prediction.clone()
    parent[1] = float("nan")
    parent[0, 2] = float("nan")
    actual, actual_parts = ref.execution_loss(m, p, s, truth, valid, parent, old, front_steps=2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key in parts:
        torch.testing.assert_close(actual_parts[key], parts[key], atol=0, rtol=0)
    actual.backward()
    assert all(torch.isfinite(v.grad).all() for v in ref.policy.parameters() if v.grad is not None)


def test_execution_loss_has_first_front_and_tail_gradient_and_unsaturates_gripper(tmp_path):
    ref, m, p, s = make_refiner(tmp_path)
    raw = torch.zeros(2, 3, 7, requires_grad=True)
    with torch.no_grad():
        raw[..., 6] = -10  # absolute unconstrained width = -40 mm
    ref._raw_prediction = lambda *_: raw
    truth = s[:, None].expand_as(raw).clone()
    truth[..., :6] += 0.01
    truth[..., 6] = 0.004
    valid = torch.ones(2, 3, dtype=torch.bool)
    parent = truth.clone()
    old = torch.zeros(2, dtype=torch.bool)
    ordinary, _, _ = ref.loss(m, p, s, truth, valid, parent, old)
    ordinary_grad = torch.autograd.grad(ordinary, raw)[0]
    loss, parts = ref.execution_loss(m, p, s, truth, valid, parent, old, front_steps=2)
    gradient = torch.autograd.grad(loss, raw)[0]
    assert gradient[0, 0, 6] < -0.01
    assert abs(ordinary_grad[0, 0, 6]) < 1e-10
    # The three separately averaged windows give weights 11:5:2 for H=3/front=2.
    torch.testing.assert_close(
        gradient[0, :, 0].abs() / gradient[0, 2, 0].abs(), torch.tensor([5.5, 2.5, 1.0])
    )
    assert float(parts["parent_consistency"]) == 0


def test_execution_loss_rejects_unrepresentable_valid_teacher(tmp_path):
    ref, m, p, s = make_refiner(tmp_path)
    truth = s[:, None].expand(2, 3, 7).clone()
    truth[0, 0, 6] = ref.config.positive_floor_m
    with pytest.raises(ValueError, match="positivity floor"):
        ref.execution_loss(
            m,
            p,
            s,
            truth,
            torch.ones(2, 3, dtype=torch.bool),
            truth,
            torch.zeros(2, dtype=torch.bool),
            front_steps=2,
        )


def test_inverse_positive_targets_roundtrip_across_widths_and_anchors(tmp_path):
    from piper_policy.positive_act import inverse_positive_width, parameterize

    ref, _, _, _ = make_refiner(tmp_path)
    width = torch.tensor([[0.00000101, 0.004, 0.08], [0.001, 0.03, 0.12]], dtype=torch.float64)
    state = torch.zeros(2, 7, dtype=torch.float64)
    state[:, 6] = torch.tensor([0.025, 0.09])
    raw = torch.zeros(2, 3, 7, dtype=torch.float64)
    raw[..., 6] = (inverse_positive_width(width, ref.config) - state[:, None, 6]) / 0.01
    normalized_state = state.clone()
    normalized_state[:, 6] = (state[:, 6] - 0.06) / 0.02
    normalized = parameterize(raw, normalized_state, ref.config)
    restored = normalized[..., 6] * 0.01 + state[:, None, 6]
    torch.testing.assert_close(restored, width, atol=1e-15, rtol=0)


def test_parent_windows_preserve_tail_but_prioritize_execution_and_detach_reference(tmp_path):
    ref, m, p, s = make_refiner(tmp_path)
    raw = torch.zeros(2, 3, 7, requires_grad=True)
    ref._raw_prediction = lambda *_: raw
    truth = ref.predict(m, p, s).detach()
    parent = truth.clone()
    parent[0, :, :6] += 0.01
    parent[1] = float("nan")  # new samples never enter parent consistency
    parent.requires_grad_(True)
    valid = torch.tensor([[True, True, True], [True, True, False]])
    _, parts = ref.execution_loss(
        m, p, s, truth, valid, parent, torch.tensor([True, False]), front_steps=2
    )
    gradient, parent_gradient = torch.autograd.grad(
        parts["parent_consistency"],
        (raw, parent),
        allow_unused=True,
    )
    torch.testing.assert_close(
        gradient[0, :, 0].abs() / gradient[0, 2, 0].abs(), torch.tensor([5.5, 2.5, 1.0])
    )
    torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]), atol=0, rtol=0)
    assert gradient[0, 2, 0] != 0
    assert parent_gradient is None
    torch.testing.assert_close(
        parts["parent_consistency"],
        (parts["parent_full"] + parts["parent_front"] + parts["parent_first"]) / 3,
    )


def test_physical_joint_references_ignore_gripper_and_padding(tmp_path):
    ref, m, p, s = make_refiner(tmp_path)
    truth = ref.predict(m, p, s).detach()
    truth[..., :6] += 0.1
    truth[..., 6] = 999
    valid = torch.tensor([[True, True, False], [True, True, True]])
    truth[0, 2, :6] = 999
    losses = ref.physical_joint_losses(m, p, s, truth, valid)
    for value in losses.values():
        torch.testing.assert_close(value, torch.tensor(0.1))
    sum(losses.values()).backward()
    assert torch.count_nonzero(ref.model.action_head.weight.grad[6]) == 0
    assert torch.count_nonzero(ref.model.action_head.weight.grad[:6]) > 0
