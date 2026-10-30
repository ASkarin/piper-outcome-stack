"""Native RTC with the explicit filtered or frozen unfiltered execution recipe."""

from functools import partial
import math
import time
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from .async_chunks import RTCTimeline, ChunkWorker
from .execution_constraints import project_gripper


class JointLowPassPlan:
    """Forecast the causal 50Hz filter; preserve its state across published chunks."""

    cutoff_hz = 5.0

    def __init__(self, initial_state):
        self.sos = butter(2, self.cutoff_hz, fs=50, output="sos")
        self.initial_zi = sosfilt_zi(self.sos)[:, :, None] * initial_state[None, None, :6]
        self.origin = self.start_index = self.states = self.actions = None

    def publish(self, raw, origin_tick, publication_tick):
        start = publication_tick - origin_tick
        if not 0 <= start < len(raw):
            raise TimeoutError("low-pass plan has no remaining actions")
        if self.origin is None:
            zi = self.initial_zi.copy()
        else:
            previous_index = publication_tick - self.origin
            if not self.start_index <= previous_index <= len(self.actions):
                raise TimeoutError("low-pass continuity exceeds the previous action horizon")
            zi = self.states[previous_index].copy()
        actions = project_gripper(raw, "zero").astype(np.float64)
        states = np.repeat(zi[None], len(raw) + 1, axis=0)
        for index in range(start, len(raw)):
            actions[index : index + 1, :6], zi = sosfilt(
                self.sos, raw[index : index + 1, :6], axis=0, zi=zi
            )
            states[index + 1] = zi
        self.origin, self.start_index = origin_tick, start
        self.actions, self.states = actions, states
        return actions


class AsyncSmolVLAExecutor:
    """Retain original timestamps and raw targets; validate before SDK dispatch."""

    temporal_ensemble_coeff = None
    temporal_ensembler = None
    ensemble_updates = 0
    last_contributors = 1
    rtc_guidance = "native_joint6"
    rtc_num_steps = 10
    rtc_initial_delay = 20

    def __init__(self, predictor, *, request_stride=25, joint_lowpass=True, clock=time.monotonic):
        if type(joint_lowpass) is not bool:
            raise ValueError("joint_lowpass must explicitly select a boolean recipe")
        self.predictor = predictor
        self.joint_lowpass = joint_lowpass
        self.clock = clock
        self.worker = None
        self.n_action_steps = request_stride
        self.reset_execution()

    def reset_execution(self):
        if self.worker is not None:
            self.worker.close()
        self.worker = None
        self.timeline = RTCTimeline(
            initial_delay=self.rtc_initial_delay, request_stride=self.n_action_steps
        )
        self.origin = self.initial_state = None
        self.generation_state = self.generation_sequence = None
        self.origins = {}
        self.published = None
        self.joint_filter = None
        self.execution_details = {}
        self.predictor.model.reset()
        self.predictor.configure_rtc(True, num_steps=self.rtc_num_steps, guidance=self.rtc_guidance)

    def execution_chunk(self, image, state, sequence, *, observed_monotonic_s):
        now = self.clock()
        if not math.isfinite(observed_monotonic_s) or not observed_monotonic_s <= now:
            raise ValueError("invalid asynchronous observation time")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("asynchronous execution requires finite seven-axis state")
        if self.origin is None:
            self.origin = observed_monotonic_s
            self.initial_state = state.copy()
            if self.joint_lowpass:
                self.joint_filter = JointLowPassPlan(self.initial_state)
            self.generation_state, self.generation_sequence = state.tolist(), sequence
            self.worker = ChunkWorker(
                partial(self.predictor.predict_chunk, gripper_lower_bound="zero")
            )
        observation_tick = math.floor((observed_monotonic_s - self.origin) / self.timeline.period)
        if observation_tick < 0:
            raise ValueError("asynchronous observation time moved backwards")
        result = self.worker.collect()
        if result is not None:
            record = self.origins[result.observation_tick]
            record["inference_seconds"] = result.inference_seconds
            result.inference_seconds = self.clock() - record["observed_monotonic_s"]
            self.timeline.schedule(result)
        tick = math.floor((self.clock() - self.origin) / self.timeline.period)
        target, timing = self.timeline.target(tick)
        if target is None and tick >= 50:
            raise TimeoutError("first inference exceeded its action horizon")
        active = self.timeline.active
        changed = active is not None and self.published != active.observation_tick
        if changed:
            active.execution_actions = (
                self.joint_filter.publish(active.actions, active.observation_tick, tick)
                if self.joint_lowpass
                else project_gripper(active.actions, "zero")
            )
            active.committed_model_actions = self.predictor.encode_execution_actions(
                active.execution_actions
            )
        if self.worker.future is None and self.timeline.needs_request(observation_tick):
            if self.origins and observation_tick <= max(self.origins):
                raise ValueError("new asynchronous request needs a newer observation")
            snapshot, observed, options = self.timeline.capture_request(
                image, state, observation_tick
            )
            previous = self.timeline.active
            consumed = (
                observation_tick - previous.observation_tick if previous is not None else None
            )
            prefix = options["previous_model_actions"]
            request = {
                "guidance": self.rtc_guidance,
                "delay_steps": options["inference_delay"],
                "overlap_steps": options["overlap_steps"],
                "previous_plan_origin_tick": previous.observation_tick
                if previous is not None
                else None,
                "consumed_steps": consumed,
                "prefix_model_actions": prefix.tolist() if prefix is not None else None,
                "prefix_execution_actions": previous.execution_actions[consumed:].tolist()
                if previous is not None
                else None,
                "guided_action_indices": list(range(6)),
                "gripper_prefix_role": "unguided model goal; live reference is committed by robot",
            }
            self.origins[observation_tick] = {
                "state": state.tolist(),
                "sequence": sequence,
                "observed_monotonic_s": observed_monotonic_s,
                "rtc_request": request,
            }
            self.worker.submit(snapshot, observed, observation_tick, **options)
        if target is None:
            self.execution_details = {"phase": "initial_hold", "control_tick": tick}
            return np.repeat(self.initial_state[None], 50, axis=0), 0, False
        self.published = active.observation_tick
        record = self.origins[active.observation_tick]
        self.generation_state, self.generation_sequence = record["state"], record["sequence"]
        self.execution_details = {
            **timing,
            "control_tick": tick,
            "generation_observed_monotonic_s": record["observed_monotonic_s"],
            "inference_seconds": record["inference_seconds"],
            "rtc": {
                k: v
                for k, v in record["rtc_request"].items()
                if k not in ("prefix_model_actions", "prefix_execution_actions")
            },
            "target_transform": self.target_transform,
            "joint_filter_cutoff_hz": 5.0 if self.joint_lowpass else None,
        }
        if changed:
            self.execution_details["rtc_request"] = record["rtc_request"]
            self.execution_details[
                "filtered_execution_actions_rad_m"
                if self.joint_lowpass
                else "execution_actions_rad_m"
            ] = active.execution_actions.tolist()
            if self.joint_lowpass:
                self.execution_details["filter_start_index"] = self.joint_filter.start_index
        self.origins = {k: v for k, v in self.origins.items() if k >= active.observation_tick}
        return active.actions, timing["chunk_index"], changed

    def select_target(self, chunk, *, chunk_index=0, new_chunk=True):
        raw = chunk[chunk_index].copy()
        target = (
            self.timeline.active.execution_actions[chunk_index].copy()
            if self.timeline.active is not None
            else project_gripper(raw, "zero")
        )
        self.execution_details = {
            **self.execution_details,
            "raw_target_rad_m": raw.tolist(),
            "gripper_correction_m": float(target[6] - raw[6]),
            "target_transform": self.target_transform,
            "joint_filter_cutoff_hz": 5.0 if self.joint_lowpass else None,
            "gripper_projection": "zero",
        }
        return target

    @property
    def target_transform(self):
        return (
            "butterworth2_5hz_joints_and_gripper_max_zero"
            if self.joint_lowpass
            else "raw_joints_and_gripper_max_zero"
        )
