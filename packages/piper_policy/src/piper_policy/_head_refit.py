"""Shared frozen ACT feature capture, export and regularized linear algebra."""

import json
from pathlib import Path
import shutil

import torch
from .policy_execution import ACTChunkPredictor


@torch.no_grad()
def ridge_solution(features, desired, valid, train_indices, prior, relative_regularization):
    if relative_regularization <= 0 or not len(train_indices):
        raise ValueError("positive regularization and nonempty training indices required")
    dimension = features.shape[-1]
    vector = prior.ndim == 1
    if vector:
        prior = prior[:, None]
        desired = desired[..., None]
    prior = prior.double()
    normal = torch.zeros(dimension + 1, dimension + 1, dtype=torch.float64)
    rhs = torch.zeros_like(prior)
    counts = torch.bincount(torch.as_tensor(train_indices), minlength=len(features)).double()
    observations = torch.nonzero(counts).flatten()
    total_weight = 0.0
    for offset in range(0, len(observations), 32):
        ids = observations[offset : offset + 32]
        mask = valid[ids].flatten()
        x = features[ids].double().flatten(0, 1)[mask]
        x = torch.cat([x, torch.ones(len(x), 1, dtype=torch.float64)], dim=1)
        target = desired[ids].double().flatten(0, 1)[mask]
        weights = counts[ids, None].expand_as(valid[ids]).flatten()[mask]
        weighted_x = x * weights.sqrt()[:, None]
        residual = (target - x @ prior) * weights.sqrt()[:, None]
        normal += weighted_x.T @ weighted_x
        rhs += weighted_x.T @ residual
        total_weight += float(weights.sum())
    if total_weight == 0:
        raise ValueError("training observations have no valid targets")
    normal /= total_weight
    rhs /= total_weight
    regularization = relative_regularization * normal[:dimension, :dimension].diag().mean()
    if not torch.isfinite(regularization) or regularization <= 0:
        raise ValueError("invalid cached feature energy")
    delta = torch.linalg.solve(
        normal + regularization * torch.eye(dimension + 1, dtype=torch.float64), rhs
    )
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite output head solve")
    fitted = prior + delta
    result = dict(
        solver="regularized_linear_solve",
        relative_regularization=relative_regularization,
        regularization=float(regularization),
        unique_observations=len(observations),
        valid_weighted_entries=total_weight,
        parameter_update_l2=float(delta.norm()),
    )
    return (fitted[:, 0] if vector else fitted), result


class FrozenACTHead:
    """Common implementation for the two explicit component-refit tools."""

    def __init__(self, predictor, checkpoint):
        self.predictor = predictor
        self.checkpoint = Path(checkpoint)
        self.config = predictor.model.config
        self.head = predictor.model.model.action_head
        if not isinstance(self.head, torch.nn.Linear) or self.head.out_features != 7:
            raise ValueError("refit requires the seven-axis ACT linear output head")
        predictor.model.eval()
        predictor.model.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        return cls(ACTChunkPredictor.from_checkpoint(checkpoint), checkpoint)

    def extract_features(self, images, states):
        captured = []

        def capture(module, inputs):
            captured.append(inputs[0].detach().cpu().clone())

        hook = self.head.register_forward_pre_hook(capture)
        try:
            prediction = self.predictor.predict_batch(images, states)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("expected one ACT head call per inference batch")
        return captured[0], prediction

    def save(self, output):
        rows, weight, bias, metadata_name, parameters = self.replacement()
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("nonfinite output head")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        original_weight = self.head.weight[rows].detach().clone()
        original_bias = self.head.bias[rows].detach().clone()
        try:
            with torch.no_grad():
                self.head.weight[rows].copy_(weight.to(self.head.weight.device))
                self.head.bias[rows].copy_(bias.to(self.head.bias.device))
            self.predictor.model.save_pretrained(output)
        finally:
            with torch.no_grad():
                self.head.weight[rows].copy_(original_weight)
                self.head.bias[rows].copy_(original_bias)
        for source in self.checkpoint.iterdir():
            if (
                source.name.startswith("policy_preprocessor")
                or source.name.startswith("policy_postprocessor")
                or source.name == "representation.json"
            ):
                shutil.copy2(source, output / source.name)
        (output / metadata_name).write_text(
            json.dumps(
                dict(
                    source_checkpoint=str(self.checkpoint),
                    scope="inference-only checkpoint; full upstream optimizer state not copied",
                    changed_parameters=parameters,
                    feature_path="eval mode, zero VAE latent, RGB and state only",
                ),
                indent=2,
            )
        )
