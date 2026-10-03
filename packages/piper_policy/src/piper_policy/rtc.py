"""RTC Eq. 2-5 (Black et al., 2025), for SmolVLA's reverse-time flow."""

import math

import torch


def prefix_weights(delay, overlap, horizon, *, device, dtype):
    if not 0 <= delay <= overlap <= horizon:
        raise ValueError("RTC requires 0 <= delay <= overlap <= horizon")
    i = torch.arange(horizon, device=device, dtype=dtype)
    c = (overlap - i) / (overlap - delay + 1)
    # Evaluate the exponential only on its intended interval.
    weights = torch.zeros_like(i)
    middle = (i >= delay) & (i < overlap)
    weights[middle] = c[middle] * torch.expm1(c[middle]) / math.expm1(1)
    weights[i < delay] = 1
    return weights


class PaperRTCProcessor:
    """Full VJP through the denoiser; no parameter gradients or action clipping."""

    def __init__(self, beta=5.0, action_dim=7, guidance="full-vjp"):
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("RTC beta must be finite and positive")
        if guidance not in ("full-vjp", "identity"):
            raise ValueError("unknown RTC guidance")
        self.beta, self.action_dim, self.guidance = beta, action_dim, guidance

    def is_debug_enabled(self):
        return False

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ):
        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)
        if not 0 < time <= 1:
            raise ValueError("reverse flow time must be in (0, 1]")
        if x_t.ndim != 3 or x_t.shape[-1] < self.action_dim:
            raise ValueError("RTC expects a batched padded action chunk")
        prefix = torch.as_tensor(prev_chunk_left_over, device=x_t.device, dtype=x_t.dtype)
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        overlap = execution_horizon
        if prefix.shape != (x_t.shape[0], overlap, self.action_dim):
            raise ValueError(
                "RTC prefix must match the declared overlap and real action dimensions"
            )
        if not torch.isfinite(prefix).all():
            raise ValueError("RTC prefix must be finite")
        weights = prefix_weights(
            inference_delay, overlap, x_t.shape[1], device=x_t.device, dtype=x_t.dtype
        )[None, :, None]
        target = torch.zeros_like(x_t)
        target[:, :overlap, : self.action_dim] = prefix
        mask = torch.zeros_like(x_t)
        mask[:, :, : self.action_dim] = weights
        if self.guidance == "identity":
            # LeRobot's current g=e behavior, with the same real-dimension mask as full VJP.
            with torch.no_grad():
                velocity = original_denoise_step_partial(x_t)
                correction = (target - (x_t - time * velocity)) * mask
        else:
            with torch.enable_grad():
                x = x_t.detach().clone().requires_grad_(True)
                velocity = original_denoise_step_partial(x)
                clean = x - time * velocity
                error = ((target - clean) * mask).detach()
                correction = torch.autograd.grad(
                    clean, x, grad_outputs=error, create_graph=False, retain_graph=False
                )[0]
        tau = 1 - time
        strength = (
            self.beta if tau == 0 else min(self.beta, (time * time + tau * tau) / (tau * time))
        )
        guided = velocity.detach() - strength * correction.detach()
        if not torch.isfinite(guided).all():
            raise ValueError("RTC produced non-finite guided velocity")
        return guided
