"""Experiment-only, label-relative temporal loss; no inference modification."""

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ORIGINAL_FORWARD = SmolVLAPolicy.forward


def temporal_loss(error, is_pad):
    # error is in the checkpoint's standardized action coordinates.
    # Each chunk belongs to a single retained segment. Require every point
    # between both endpoints to be valid, including at lag five (100 ms).
    joint_error = error[..., :6].float()
    terms = []
    for lag in (1, 5):
        valid = (~is_pad).unfold(1, lag + 1, 1).all(dim=-1)
        difference = joint_error[:, lag:] - joint_error[:, :-lag]
        terms.append(
            (difference.square() * valid[..., None]).sum() / (valid.sum() * 6).clamp_min(1)
        )
    return torch.stack(terms).mean()


def objective(
    policy, batch, noise=None, time=None, reduction="mean", weight=0.0, weighting="t_squared"
):
    assert reduction == "mean" and not policy.config.adapt_to_pi_aloha
    assert weighting in ("t_squared", "flat")
    actions = policy.prepare_action(batch)
    if noise is None:
        noise = policy.model.sample_noise(actions.shape, actions.device)
    if time is None:
        time = policy.model.sample_time(actions.shape[0], actions.device)
    captured = []
    handle = policy.model.action_out_proj.register_forward_hook(
        lambda module, inputs, output: captured.append(output)
    )
    try:
        base, detail = ORIGINAL_FORWARD(policy, batch, noise, time, reduction)
    finally:
        handle.remove()
    assert len(captured) == 1
    # x_t - t*v_hat - action = t*(noise - action - v_hat).
    residual = noise - actions - captured[0]
    clean_auxiliary = temporal_loss(time[:, None, None] * residual, batch["action_is_pad"])
    flat_auxiliary = temporal_loss(residual, batch["action_is_pad"])
    auxiliary = clean_auxiliary if weighting == "t_squared" else flat_auxiliary
    total = base + weight * auxiliary
    detail.update(
        flow_loss=base.item(),
        temporal_loss=auxiliary.item(),
        temporal_clean_loss=clean_auxiliary.item(),
        temporal_flat_loss=flat_auxiliary.item(),
        temporal_weight=weight,
        temporal_weighting=weighting,
        loss=total.item(),
    )
    return total, detail, base, clean_auxiliary, flat_auxiliary


def training_forward(weight, weighting):
    def forward(self, batch, noise=None, time=None, reduction="mean"):
        total, detail, _, _, _ = objective(self, batch, noise, time, reduction, weight, weighting)
        return total, detail

    return forward
