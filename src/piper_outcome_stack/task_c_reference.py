"""Stage C arithmetic and checks on independent recorded RTC action chunks."""

from contextlib import contextmanager

import numpy as np
import torch

from .policy_execution import compare_reference_chunk, recorded_target_check


@contextmanager
def fp32_runtime():
    """Keep the same backend settings through the reference check and the live worker."""
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn
    previous_cudnn = cudnn.allow_tf32, cudnn.benchmark, cudnn.deterministic
    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        cudnn.allow_tf32, cudnn.benchmark, cudnn.deterministic = False, False, True
        yield
    finally:
        cudnn.allow_tf32, cudnn.benchmark, cudnn.deterministic = previous_cudnn
        torch.set_float32_matmul_precision(previous_precision)
        torch.set_num_threads(previous_threads)


def read_reference(path, task, policy_type="smolvla"):
    with np.load(path, allow_pickle=False) as source:
        tasks = source["tasks"]
        if tasks.ndim != 1 or set(tasks.tolist()) != {task}:
            raise ValueError("C reference instructions differ from the selected skill")
        if "inference_precision" not in source or source["inference_precision"].item() != "fp32":
            raise ValueError(
                "C requires an FP32 RTC reference for SmolVLA or FP32 TE reference for ACT"
            )
        family = source["policy_type"].item() if "policy_type" in source else "smolvla"
        if family != policy_type:
            raise ValueError("C reference policy family differs from the selected checkpoint")
        keys = ["images", "states", "expected_absolute_chunks"]
        if policy_type == "act":
            keys.append("expected_ensemble_targets")
        else:
            keys.extend(
                ["seeds", "rtc_seeds", "expected_rtc_chunks", "expected_rtc_execution_chunks"]
            )
        reference = {key: source[key] for key in keys}
    n = len(tasks)
    images, states = reference["images"], reference["states"]
    if images.shape != (n, 480, 640, 3) or images.dtype != np.uint8 or states.shape != (n, 7):
        raise ValueError("C reference needs aligned 640x480 RGB images and seven-axis states")
    if reference["expected_absolute_chunks"].shape != (n, 50, 7):
        raise ValueError("C reference must contain one 50x7 chunk per observation")
    if policy_type == "act":
        if reference["expected_ensemble_targets"].shape != (n, 2, 7):
            raise ValueError("ACT reference requires the two absolute ensemble targets")
        return reference
    for name in ("seeds", "rtc_seeds"):
        seeds = reference[name]
        if seeds.shape != (n,) or not np.issubdtype(seeds.dtype, np.integer):
            raise ValueError(f"C reference needs one integer {name} value per observation")
    for name in ("expected_rtc_chunks", "expected_rtc_execution_chunks"):
        if reference[name].shape != (n, 50, 7):
            raise ValueError(f"C reference {name} must contain one 50x7 chunk per observation")
    return reference


def act_reference_outputs(policy, image, state):
    """Replay two control ticks on one recorded observation with a fresh ensemble."""
    policy.reset_execution()
    targets = []
    for tick in range(2):
        chunk, index, inferred = policy.execution_chunk(image, state, tick)
        if tick == 0:
            first = chunk.copy()
        targets.append(policy.select_target(chunk, chunk_index=index, new_chunk=inferred))
    return first, np.stack(targets)


def verify_act_reference(policy, reference, safety):
    checks = []
    for i, (image, state) in enumerate(zip(reference["images"], reference["states"], strict=True)):
        chunk, targets = act_reference_outputs(policy, image, state)
        error = compare_reference_chunk(chunk, reference["expected_absolute_chunks"][i], i)
        ensemble_error = compare_reference_chunk(
            targets, reference["expected_ensemble_targets"][i], f"{i} TE"
        )
        bounds = [recorded_target_check(state, target, safety) for target in targets]
        checks.append(
            dict(
                sample=i,
                **error,
                recorded_target_check=bounds[0],
                ensemble_error=ensemble_error,
                ensemble_target_checks=bounds,
            )
        )
    policy.reset_execution()
    return checks


def verify_rtc_reference(executor, reference, safety):
    """Rebuild each prefix from this machine's first prediction exactly as in the reference."""
    policy = executor.predictor
    checks = []
    for i, (image, state) in enumerate(zip(reference["images"], reference["states"], strict=True)):
        torch.manual_seed(int(reference["seeds"][i]))
        first = policy.predict_chunk(image, state, gripper_lower_bound="zero")
        first_error = compare_reference_chunk(
            first.actions, reference["expected_absolute_chunks"][i], i
        )

        prefix = first.committed_model_actions[executor.n_action_steps :].copy()
        torch.manual_seed(int(reference["rtc_seeds"][i]))
        second = policy.predict_chunk(
            image,
            state,
            previous_model_actions=prefix,
            inference_delay=executor.rtc_initial_delay,
            overlap_steps=len(prefix),
            gripper_lower_bound="zero",
        )
        rtc_error = compare_reference_chunk(
            second.actions, reference["expected_rtc_chunks"][i], f"{i} RTC"
        )
        execution = second.execution_actions
        execution_error = compare_reference_chunk(
            execution, reference["expected_rtc_execution_chunks"][i], f"{i} projected RTC"
        )
        rejected = []
        previous = state
        for step, target in enumerate(execution):
            bounds = recorded_target_check(previous, target, safety)
            if bounds["status"] == "rejected":
                rejected.append(dict(step=step, reason=bounds["reason"]))
            previous = target
        checks.append(
            dict(
                sample=i,
                **first_error,
                recorded_target_check=recorded_target_check(state, first.actions[0], safety),
                rtc_prefix_steps=len(prefix),
                rtc_error=rtc_error,
                rtc_execution_error=execution_error,
                rtc_commands_checked=len(execution),
                rtc_command_rejections=rejected,
            )
        )
    return checks
