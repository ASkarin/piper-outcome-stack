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


def read_reference(path, task):
    with np.load(path, allow_pickle=False) as source:
        tasks = source["tasks"]
        if tasks.ndim != 1 or set(tasks.tolist()) != {task}:
            raise ValueError("C reference instructions differ from the selected skill")
        if "inference_precision" not in source or source["inference_precision"].item() != "fp32":
            raise ValueError("C requires an FP32 RTC reference from the training machine")
        reference = {
            key: source[key]
            for key in (
                "images",
                "states",
                "seeds",
                "rtc_seeds",
                "expected_absolute_chunks",
                "expected_rtc_chunks",
                "expected_rtc_execution_chunks",
            )
        }
    n = len(tasks)
    images, states = reference["images"], reference["states"]
    if images.shape != (n, 480, 640, 3) or images.dtype != np.uint8 or states.shape != (n, 7):
        raise ValueError("C reference needs aligned 640x480 RGB images and seven-axis states")
    for name in ("seeds", "rtc_seeds"):
        seeds = reference[name]
        if seeds.shape != (n,) or not np.issubdtype(seeds.dtype, np.integer):
            raise ValueError(f"C reference needs one integer {name} value per observation")
    for name in (
        "expected_absolute_chunks",
        "expected_rtc_chunks",
        "expected_rtc_execution_chunks",
    ):
        if reference[name].shape != (n, 50, 7):
            raise ValueError(f"C reference {name} must contain one 50x7 chunk per observation")
    return reference


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
