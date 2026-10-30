"""Real observation freshness, command bounds and read-only shadow execution."""

import math
import time
import numpy as np
import torch
from piper_policy import policy_execution as policy_contract


def verify_reference_inputs(predictor, path, safety, *, warmup=30):
    """Reproduce recorded reference chunks, check first targets, then warm up.

    Returns (images, states, rows); each row keeps the target check as a report entry.
    """
    from lerobot_robot_outcome_piper.execution_constraints import check_execution_target

    with np.load(path) as data:
        images, states = data["images"], data["states"]
        expected = data["expected_absolute_chunks"]
        seeds = data["seeds"] if "seeds" in data else None
    if len(states) == 0 or len(images) != len(states) or len(expected) != len(states):
        raise ValueError("reference inputs need matching, non-empty images/states/chunks")
    rows = []
    for i in range(len(states)):
        if seeds is not None:
            torch.manual_seed(int(seeds[i]))
        chunk = predictor.predict(images[i], states[i])
        delta = np.abs(chunk - expected[i])
        joint, grip = float(delta[:, :6].max()), float(delta[:, 6].max())
        if joint >= 1e-4 or grip >= 1e-5:
            raise ValueError(f"reference mismatch: sample={i}, joint={joint}, gripper={grip}")
        try:
            check_execution_target(states[i], chunk[0], safety)
            bounds = dict(status="passed")
        except ValueError as exc:
            bounds = dict(status="rejected", reason=str(exc))
        rows.append(dict(sample=i, joint_rad=joint, gripper_m=grip, recorded_target_check=bounds))
    for i in range(warmup):
        predictor.predict(images[i % len(states)], states[i % len(states)])
    return images, states, rows


def check_policy_observation(telemetry, timing, now):
    """Use the same input contract as get_observation and the pre-command age check."""
    if telemetry["quality"] != "checked":
        raise ValueError("policy needs a timing-checked observation")
    age = now - telemetry["oldest_received_monotonic_s"]
    if not math.isfinite(age) or age < 0 or age > timing.observation_max_age_s:
        raise ValueError(
            f"policy observation expired before candidate was ready: "
            f"age_s={age}, limit_s={timing.observation_max_age_s}"
        )
    return age


def plan_policy_gripper(goal, feedback, control, epoch, settings, safety, now):
    """Prepare the existing dispatch plan without advancing accepted reference state."""
    from lerobot_robot_outcome_piper.gripper_reference import gripper_position_candidate

    previous, revision = control.gripper_reference_snapshot(epoch)
    target = control.gripper_target
    if target is None:
        target = feedback
    reference = gripper_position_candidate(
        previous,
        target,
        feedback,
        goal,
        settings,
        now,
        safety.gripper_lower,
        safety.gripper_upper,
    )
    return dict(
        feedback_m=feedback,
        target_m=reference["target_m"],
        base_revision=revision,
        reference=reference,
    )


def predict_candidate(
    predictor,
    observation,
    telemetry,
    timing,
    safety,
    clock=time.monotonic,
    record_action_chunk=False,
    *,
    control=None,
    epoch=None,
    gripper_reference_settings=None,
):
    """One observation → absolute chunk → optional TE → checked target; sends nothing."""
    from lerobot_robot_outcome_piper.safety import ACTION_KEYS
    from lerobot_robot_outcome_piper.execution_constraints import check_execution_target

    started = clock()
    check_policy_observation(telemetry, timing, started)
    state = [float(observation[k]) for k in ACTION_KEYS]
    chunk, chunk_index, inferred = predictor.execution_chunk(
        observation[policy_contract.POLICY_CAMERA],
        state,
        telemetry["sequence"],
        observed_monotonic_s=telemetry["observed_monotonic_s"],
    )
    predicted = clock()
    target = predictor.select_target(chunk, chunk_index=chunk_index, new_chunk=inferred)
    recorded_prediction = {}
    if record_action_chunk:
        recorded_prediction = dict(
            generation_observation_sequence=predictor.generation_sequence,
            anchor=list(predictor.generation_state),
            candidate_chunk_index=chunk_index,
            inference_performed=inferred,
            selected_target_rad_m=target.tolist(),
            asynchronous_execution=getattr(predictor, "execution_details", None),
        )
        if inferred:
            recorded_prediction["absolute_action_chunk_rad_m"] = chunk.tolist()
    try:
        if gripper_reference_settings is not None:
            recorded_prediction["policy_goal_rad_m"] = target.tolist()
            if control is None or epoch is None:
                raise ValueError("policy gripper reference needs the current control epoch")
            goal = float(target[6])
            if (
                getattr(predictor, "execution_details", {}).get("phase") == "initial_hold"
                and control.gripper_target is not None
            ):
                goal = control.gripper_target
            gripper_plan = plan_policy_gripper(
                goal,
                state[6],
                control,
                epoch,
                gripper_reference_settings,
                safety,
                predicted,
            )
            target = np.asarray(target, dtype=np.float64).copy()
            target[6] = gripper_plan["target_m"]
            recorded_prediction.update(
                gripper_plan=gripper_plan,
                selected_target_rad_m=target.tolist(),
            )
        if control is None:
            check_execution_target(state, target, safety)
        else:
            check_execution_target(
                state,
                target,
                safety,
                last_gripper_target=control.gripper_target,
            )
    except ValueError as exc:
        error = ValueError(f"{exc}; observed_rad_m={state}; candidate_rad_m={target.tolist()}")
        if recorded_prediction:
            error.policy_prediction = recorded_prediction
        raise error from exc
    checked = clock()
    try:
        age = check_policy_observation(telemetry, timing, checked)
    except ValueError as exc:
        if recorded_prediction:
            exc.policy_prediction = recorded_prediction
        exc.policy_timing = {
            "input_age_before_prediction_s": started - telemetry["oldest_received_monotonic_s"],
            "inference_s": predicted - started,
            "selection_and_validation_s": checked - predicted,
            "age_at_rejection_s": checked - telemetry["oldest_received_monotonic_s"],
            "inference_performed": inferred,
        }
        raise
    return dict(zip(ACTION_KEYS, map(float, target))), {
        **recorded_prediction,
        "observation_sequence": telemetry["sequence"],
        "anchor": predictor.generation_state,
        "generation_observation_sequence": predictor.generation_sequence,
        "inference_performed": inferred,
        "n_action_steps": predictor.n_action_steps,
        "inference_s": predicted - started,
        "validation_s": checked - predicted,
        "observation_age_at_candidate_s": age,
        "executed_chunk_steps": chunk_index,
        "candidate_chunk_index": chunk_index,
        "temporal_ensemble_coeff": predictor.temporal_ensemble_coeff,
        "temporal_ensemble_contributors": predictor.last_contributors
        if predictor.temporal_ensembler is not None
        else 1,
        "asynchronous_execution": getattr(predictor, "execution_details", None),
    }


def run_shadow(robot, predictor, safety, cycles, fps, rows, clock=time.monotonic, sleep=time.sleep):
    """Connected read-only robot only. No enable, mode change or action dispatch."""
    if robot.config.execution_mode != "read_only" or robot.config.capture_timing is None:
        raise ValueError("shadow requires explicit read_only and measured capture_timing")
    if cycles <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("shadow requires positive cycles and fps")
    for i in range(cycles):
        started = clock()
        observation = robot.get_observation()
        observed = clock()
        target, entry = predict_candidate(
            predictor,
            observation,
            robot.last_observation_telemetry,
            robot.config.capture_timing,
            safety,
            clock,
        )
        ended = clock()
        entry.update(
            cycle=i,
            target=target,
            acquisition_s=observed - started,
            cycle_work_s=ended - started,
            input_telemetry=robot.last_observation_telemetry,
        )
        rows.append(entry)
        sleep(max(0, 1 / fps - (clock() - started)))
        entry["cycle_elapsed_s"] = clock() - started
        if (i + 1) % 50 == 0:
            print(f"shadow {i + 1}/{cycles}: work={1000 * entry['cycle_work_s']:.2f}ms", flush=True)
