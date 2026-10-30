"""Project a proposed optimizer parameter step against one or two reference gradients."""

import itertools
import torch


def project_displacement(delta, gradients):
    """Closest step in the intersection of up to two non-ascent halfspaces."""
    if gradients.ndim != 2 or len(gradients) not in (1, 2):
        raise ValueError("one or two reference gradients required")
    norms = gradients.norm(dim=1)
    active = norms > 0
    unit = gradients[active] / norms[active, None]
    dots = unit @ delta
    if not torch.any(dots > 0):
        return delta, False
    gram = (unit @ unit.T).cpu()
    rhs = dots.cpu()
    tolerance = 64 * torch.finfo(delta.dtype).eps * float(delta.norm())
    candidates = []
    for size in range(1, len(unit) + 1):
        for subset in itertools.combinations(range(len(unit)), size):
            selected = torch.tensor(subset)
            block = gram[selected][:, selected]
            multiplier = torch.linalg.lstsq(block, rhs[selected]).solution
            if torch.any(multiplier < -tolerance):
                continue
            full = torch.zeros(len(unit), dtype=delta.dtype)
            full[selected] = multiplier
            if torch.any(rhs - gram @ full > tolerance):
                continue
            cost = float(full @ gram @ full)
            candidates.append((cost, full))
    if not candidates:
        raise RuntimeError("no numerically feasible reference projection found")
    multiplier = min(candidates, key=lambda item: item[0])[1].to(delta.device)
    return delta - unit.T @ multiplier, True


def step_with_reference_projection(optimizer, loss, reference_loss=None):
    """Keep optimizer moments, but remove a first-order ascent component if present.

    This uses the actual optimizer displacement, including momentum and decay.
    It does not guarantee nonlinear loss, validation, or policy improvement.
    Floating-point assignment can leave a small first-order residual, reported
    explicitly rather than described as exact preservation.
    """
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if reference_loss is None:
        loss.backward()
        optimizer.step()
        return dict(projected=False, reference_present=False)
    losses = [reference_loss] if isinstance(reference_loss, torch.Tensor) else list(reference_loss)
    gradients = []
    for value in losses:
        reference = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        gradients.append(
            torch.cat(
                [
                    torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                    for p, g in zip(parameters, reference)
                ]
            ).double()
        )
    gradient = torch.stack(gradients)
    before = torch.cat([p.detach().flatten() for p in parameters])
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        proposed = torch.cat([p.flatten() for p in parameters])
        delta = proposed.double() - before.double()
        dots = gradient @ delta
        adjusted_delta, projected = project_displacement(delta, gradient)
        if projected:
            adjusted = before.double() + adjusted_delta
            offset = 0
            for p in parameters:
                p.copy_(adjusted[offset : offset + p.numel()].reshape_as(p).to(p.dtype))
                offset += p.numel()
        actual = torch.cat([p.flatten() for p in parameters]).double() - before.double()
        applied_dot = gradient @ actual
        norms = gradient.norm(dim=1)
        scale = norms * actual.norm()
        return dict(
            projected=projected,
            reference_present=True,
            proposed_reference_dots=dots.tolist(),
            applied_reference_dots=applied_dot.tolist(),
            applied_reference_cosines=[
                float(d / s) if s else None for d, s in zip(applied_dot, scale)
            ],
            reference_gradient_norms=norms.tolist(),
            proposed_step_norm=float(delta.norm()),
            applied_step_norm=float(actual.norm()),
        )
