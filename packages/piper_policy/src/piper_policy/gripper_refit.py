"""Inference-path ACT gripper-head refitting; no simulator or device dependencies."""

import torch
from torch.nn import functional as F
from .positive_act import inverse_positive_width, parameterize
from ._head_refit import FrozenACTHead, ridge_solution


def _polish_nonnegative_dual(gram, rhs, multipliers, *, tolerance, max_steps):
    """Solve remaining dual faces when correlated rows stall coordinate sweeps.

    The objective is unchanged: .5 * lambda.T @ gram @ lambda - rhs.T @ lambda,
    subject to nonnegative lambda. Negative free-face solutions are approached
    only until a multiplier reaches zero; inactive positive residuals add a face.
    """
    lam = multipliers.clone()
    free = lam > 0
    for step in range(max_steps):
        ids = torch.nonzero(free).flatten()
        proposed = torch.zeros_like(lam)
        if len(ids):
            proposed[ids] = torch.linalg.solve(gram[ids][:, ids], rhs[ids])
        negative = ids[proposed[ids] < 0]
        if len(negative):
            rates = lam[negative] / (lam[negative] - proposed[negative])
            first = int(rates.argmin())
            lam = (lam + rates[first] * (proposed - lam)).clamp_min(0)
            lam[negative[first]] = 0
            free = lam > 0
            continue
        lam = proposed
        residual = rhs - gram @ lam
        kkt = torch.where(lam > 0, residual.abs(), residual.clamp_min(0))
        if torch.isfinite(lam).all() and float(kkt.max()) <= tolerance:
            return lam, step + 1
        free = lam > 0
        inactive = torch.where(~free, residual, torch.full_like(residual, -float("inf")))
        if float(inactive.max()) <= tolerance:
            raise RuntimeError("local gripper active-set precision exhausted")
        free[int(inactive.argmax())] = True
    raise RuntimeError("local gripper active-set budget exhausted")


class GripperHeadRefiner(FrozenACTHead):
    """Fit only the final gripper row, using cached RGB/state zero-latent features."""

    def __init__(self, predictor, checkpoint):
        super().__init__(predictor, checkpoint)
        self.weight = torch.nn.Parameter(self.head.weight[6].detach().cpu().clone())
        self.bias = torch.nn.Parameter(self.head.bias[6].detach().cpu().clone())

    def predict_gripper(self, features, states):
        if features.ndim != 3 or states.shape != (len(features), 7):
            raise ValueError("expected N,H,D features and N,7 observed states")
        # Use the existing positivity transform rather than a second implementation.
        actions = torch.zeros((*features.shape[:2], 7), dtype=features.dtype)
        actions[..., 6] = F.linear(features, self.weight[None], self.bias[None]).squeeze(-1)
        normalized_state = torch.zeros_like(states)
        normalized_state[:, 6] = (
            states[:, 6] - self.config.gripper_state_mean
        ) / self.config.gripper_state_std
        relative = parameterize(actions, normalized_state, self.config)[..., 6]
        return (
            relative * self.config.gripper_action_std
            + self.config.gripper_action_mean
            + states[:, None, 6]
        )

    def loss(self, features, states, absolute_targets, valid):
        prediction = self.predict_gripper(features, states)
        if prediction.shape != absolute_targets.shape or valid.shape != prediction.shape:
            raise ValueError("target and padding mask must match N,H predictions")
        if valid.dtype != torch.bool or not valid.any():
            raise ValueError("at least one valid target is required")
        return (prediction - absolute_targets).abs()[valid].mean() / self.config.gripper_action_std

    def fit(
        self,
        features,
        states,
        absolute_targets,
        valid,
        train_indices,
        *,
        steps,
        batch_size=32,
        learning_rate=1e-5,
        seed=42,
    ):
        """Fixed-budget Adam fitting on cached inference features; no model forward here."""
        if steps <= 0 or batch_size <= 0 or learning_rate <= 0 or not len(train_indices):
            raise ValueError(
                "positive budget, learning rate and nonempty training indices required"
            )
        optimizer = torch.optim.Adam([self.weight, self.bias], lr=learning_rate)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.as_tensor(train_indices, dtype=torch.long)
        history = []
        for step in range(steps):
            selected = indices[torch.randint(len(indices), (batch_size,), generator=generator)]
            optimizer.zero_grad()
            loss = self.loss(
                features[selected], states[selected], absolute_targets[selected], valid[selected]
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite refit loss")
            loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 0 or step + 1 == steps:
                history.append(dict(step=step + 1, normalized_l1=float(loss.detach())))
        return history

    @torch.no_grad()
    def fit_ridge(
        self,
        features,
        states,
        absolute_targets,
        valid,
        train_indices,
        *,
        relative_regularization=0.01,
    ):
        desired = torch.zeros_like(absolute_targets, dtype=torch.float64)
        unconstrained = inverse_positive_width(absolute_targets[valid].double(), self.config)
        anchor = states[:, 6].double()[:, None].expand_as(valid)[valid]
        desired[valid] = (
            unconstrained - anchor - self.config.gripper_action_mean
        ) / self.config.gripper_action_std
        prior = torch.cat([self.weight.detach().double(), self.bias.detach().double()[None]])
        fitted, result = ridge_solution(
            features, desired, valid, train_indices, prior, relative_regularization
        )
        self.weight.copy_(fitted[:-1].float())
        self.bias.copy_(fitted[-1].float())
        return result

    @torch.no_grad()
    def calibrate_bias_floor(
        self,
        features,
        states,
        absolute_targets,
        valid,
        train_indices,
        *,
        eligible_target_m=0.04,
        prediction_floor_m=0.035,
    ):
        """Smallest upward bias satisfying train-only head-space lower bounds.

        This changes one learned parameter, not inference-time action clipping.
        Floating-point decoding and unseen inputs still require public evaluation.
        """
        if features.ndim != 3 or states.shape != (len(features), 7):
            raise ValueError("expected N,H,D features and N,7 states")
        if absolute_targets.shape != features.shape[:2] or valid.shape != absolute_targets.shape:
            raise ValueError("targets and padding mask must match N,H")
        if valid.dtype != torch.bool or not len(train_indices):
            raise ValueError("nonempty train indices and boolean valid mask required")
        if not 0 < prediction_floor_m <= eligible_target_m < float("inf"):
            raise ValueError("finite positive floor no greater than eligible target required")
        ids = torch.as_tensor(train_indices, dtype=torch.long)
        eligible = valid[ids] & (absolute_targets[ids] >= eligible_target_m)
        if not eligible.any():
            raise ValueError("no eligible training targets")
        raw = F.linear(features[ids].double(), self.weight.double()[None], self.bias.double()[None])
        raw = raw.squeeze(-1)
        floor = inverse_positive_width(
            torch.tensor(prediction_floor_m, dtype=torch.float64), self.config
        )
        desired = (
            floor - states[ids, 6].double()[:, None] - self.config.gripper_action_mean
        ) / self.config.gripper_action_std
        required = (desired - raw)[eligible].max().clamp_min(0)
        before = self.bias.detach().double().clone()
        exact = before + required
        rounded = exact.to(self.bias.dtype)
        if rounded.double() < exact:
            rounded = torch.nextafter(rounded, torch.full_like(rounded, float("inf")))
        self.bias.copy_(rounded)
        actual_shift = self.bias.double() - before
        return dict(
            method="minimum_upward_training_bias",
            training_observations=len(ids),
            eligible_training_entries=int(eligible.sum()),
            eligible_target_m=eligible_target_m,
            prediction_floor_m=prediction_floor_m,
            normalized_bias_shift=float(actual_shift),
            unconstrained_width_shift_m=float(actual_shift * self.config.gripper_action_std),
            minimum_headspace_margin=float((raw + actual_shift - desired)[eligible].min()),
            inference_clipping=False,
        )

    def replacement(self):
        return (
            slice(6, 7),
            self.weight[None],
            self.bias[None],
            "gripper_refit.json",
            ["model.action_head.weight[6]", "model.action_head.bias[6]"],
        )

    @torch.no_grad()
    def calibrate_local_floor(
        self,
        features,
        states,
        absolute_targets,
        valid,
        train_indices,
        *,
        eligible_target_m=0.04,
        prediction_floor_m=0.035,
        relative_regularization=0.01,
        max_constraints=128,
        max_sweeps=2000,
        tolerance=1e-8,
    ):
        """Minimize train prediction change subject to train-only width bounds.

        The quadratic metric covers every valid training head input, plus ridge.
        A cutting-plane dual coordinate solve changes only the gripper head.
        Failure to converge raises before modifying parameters. No runtime clamp.
        """
        if features.ndim != 3 or states.shape != (len(features), 7):
            raise ValueError("expected N,H,D features and N,7 states")
        if valid.shape != features.shape[:2] or absolute_targets.shape != valid.shape:
            raise ValueError("targets and padding must match N,H")
        if valid.dtype != torch.bool or not len(train_indices):
            raise ValueError("nonempty train indices and boolean padding required")
        if not 0 < prediction_floor_m <= eligible_target_m < float("inf"):
            raise ValueError("finite positive floor no greater than eligible target required")
        if relative_regularization <= 0 or min(max_constraints, max_sweeps, tolerance) <= 0:
            raise ValueError("positive regularization, solver budget and tolerance required")
        ids = torch.as_tensor(train_indices, dtype=torch.long)
        mask = valid[ids]
        eligible = mask & (absolute_targets[ids] >= eligible_target_m)
        if not eligible.any():
            raise ValueError("no eligible training entries")
        dim = features.shape[-1] + 1
        metric = torch.zeros(dim, dim, dtype=torch.float64)
        count = 0
        for start in range(0, len(ids), 32):
            batch = ids[start : start + 32]
            x = features[batch].double()[valid[batch]]
            x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], -1)
            metric += x.T @ x
            count += len(x)
        metric /= count
        ridge = relative_regularization * metric[:-1, :-1].diag().mean()
        if not torch.isfinite(ridge) or ridge <= 0:
            raise ValueError("invalid training feature energy")
        metric += ridge * torch.eye(dim, dtype=metric.dtype)
        factor = torch.linalg.cholesky(metric)
        x = features[ids].double()[eligible]
        x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], -1)
        anchor = states[ids, 6].double()[:, None].expand_as(eligible)[eligible]
        floor = inverse_positive_width(
            torch.tensor(prediction_floor_m, dtype=torch.float64), self.config
        )
        target = (floor - anchor - self.config.gripper_action_mean) / self.config.gripper_action_std
        prior = torch.cat([self.weight.double(), self.bias.double()[None]])
        required = target - x @ prior
        delta = torch.zeros_like(prior)
        active = []
        multipliers = torch.empty(0, dtype=torch.float64)
        total_sweeps = 0
        active_set_polishes = []
        for iteration in range(max_constraints + 1):
            violation = required - x @ delta
            worst = int(violation.argmax())
            if float(violation[worst]) <= tolerance:
                break
            if iteration == max_constraints or worst in active:
                raise RuntimeError("local gripper constraint budget or precision exhausted")
            active.append(worst)
            a = x[active]
            inverse_a = torch.cholesky_solve(a.T, factor)
            gram = a @ inverse_a
            multipliers = torch.cat([multipliers, torch.zeros(1, dtype=torch.float64)])
            residual = required[active] - gram @ multipliers
            for sweep in range(max_sweeps):
                for j in range(len(active)):
                    updated = (multipliers[j] + residual[j] / gram[j, j]).clamp_min(0)
                    change = updated - multipliers[j]
                    multipliers[j] = updated
                    residual -= gram[:, j] * change
                kkt = torch.where(multipliers > 0, residual.abs(), residual.clamp_min(0))
                if float(kkt.max()) <= tolerance:
                    break
            else:
                multipliers, face_steps = _polish_nonnegative_dual(
                    gram,
                    required[active],
                    multipliers,
                    tolerance=tolerance,
                    max_steps=2 * len(active) + 1,
                )
                active_set_polishes.append(dict(constraints=len(active), steps=face_steps))
            total_sweeps += sweep + 1
            delta = inverse_a @ multipliers
        fitted = prior + delta
        if not torch.isfinite(fitted).all():
            raise RuntimeError("nonfinite calibrated gripper head")
        self.weight.copy_(fitted[:-1].to(self.weight.dtype))
        self.bias.copy_(fitted[-1].to(self.bias.dtype))
        rounded = torch.cat([self.weight.double(), self.bias.double()[None]])
        return dict(
            method="minimum_training_prediction_change_with_width_constraints",
            training_observations=len(ids),
            valid_training_entries=count,
            eligible_training_entries=len(x),
            active_constraints=len(active),
            solver_sweeps=total_sweeps,
            relative_regularization=relative_regularization,
            active_set_polishes=active_set_polishes,
            regularization=float(ridge),
            normalized_parameter_change_l2=float(delta.norm()),
            quadratic_change=float(delta @ metric @ delta),
            minimum_headspace_margin_before_rounding=float((x @ fitted - target).min()),
            minimum_headspace_margin_after_rounding=float((x @ rounded - target).min()),
            prediction_floor_m=prediction_floor_m,
            eligible_target_m=eligible_target_m,
            inference_clipping=False,
        )
