"""Summaries of observed receive times; no hardware-synchronization inference."""

from collections import defaultdict
import math
import numpy as np


def distribution(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return dict(
        count=len(values),
        min=float(a.min()),
        p50=float(np.percentile(a, 50)),
        p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)),
        max=float(a.max()),
    )


class TimingSummary:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.previous = {}
        self.counters = defaultdict(int)
        self.stream_times = defaultdict(list)
        self.observation_times = []

    def add(self, observation, action=None):
        now = float(observation["observed_monotonic_s"])
        received = list(map(float, observation["feedback"]["received_monotonic_s"]))
        if len(received) != 5 or not all(math.isfinite(t) and 0 <= t <= now for t in received):
            raise ValueError("invalid robot receive-time vector")
        if self.observation_times and now <= self.observation_times[-1]:
            raise ValueError("observation monotonic time did not advance")
        self.observation_times.append(now)
        self.metrics["joint_group_skew_s"].append(max(received[:3]) - min(received[:3]))
        self.metrics["robot_max_age_s"].append(now - min(received))
        for name, metadata in observation["cameras"].items():
            t = float(metadata["received_monotonic_s"])
            device_t = float(metadata["device_timestamp_ms"])
            frame = int(metadata["frame_number"])
            domain = metadata["timestamp_domain"]
            if (
                not all(math.isfinite(v) for v in (t, device_t))
                or not 0 <= t <= now
                or device_t < 0
                or frame < 0
            ):
                raise ValueError(f"invalid camera timing: {name}")
            self.metrics[f"{name}.frame_age_s"].append(now - t)
            self.metrics[f"{name}.image_state_skew_s"].append(max(abs(t - s) for s in received))
            previous = self.previous.get(name)
            if previous:
                old_frame, old_t, old_device_t, old_domain = previous
                self.counters[f"{name}.duplicate_frames"] += int(frame == old_frame)
                self.counters[f"{name}.backwards_frames"] += int(frame < old_frame)
                self.counters[f"{name}.unselected_or_unobserved_frames"] += max(
                    0, frame - old_frame - 1
                )
                self.counters[f"{name}.device_time_anomalies"] += int(
                    device_t <= old_device_t or domain != old_domain
                )
                self.counters[f"{name}.host_time_anomalies"] += int(t < old_t)
                self.metrics[f"{name}.selected_frame_interval_s"].append(t - old_t)
            self.previous[name] = (frame, t, device_t, domain)
            self.stream_times[name].append(t)
        if action:
            generated = action.get("generated_monotonic_s")
            if generated is not None:
                generated = float(generated)
                if not math.isfinite(generated) or generated < now:
                    raise ValueError("action generation precedes its observation")
                self.metrics["observation_to_action_generation_s"].append(generated - now)
            for command in action.get("commands", []):
                start, end = command.get("started_monotonic_s"), command.get("ended_monotonic_s")
                if (
                    start is None
                    or end is None
                    or not all(math.isfinite(v) for v in (start, end))
                    or end < start
                ):
                    raise ValueError("SDK command timing is incomplete or invalid")
                if start < now:
                    raise ValueError("new SDK call precedes this observation")
                self.metrics["observation_to_sdk_start_s"].append(start - now)
                self.metrics[f"{command['name']}.sdk_duration_s"].append(end - start)
                self.counters[f"{command['name']}.{command['result']}"] += 1
            # Retained commands intentionally do not become new SDK-call samples.

    def result(self):
        def hz(times):
            return (
                (len(times) - 1) / (times[-1] - times[0])
                if len(times) > 1 and times[-1] > times[0]
                else None
            )

        return dict(
            samples=len(self.observation_times),
            metrics={k: distribution(v) for k, v in self.metrics.items()},
            counters=dict(self.counters),
            observation_hz=hz(self.observation_times),
            selected_stream_hz={k: hz(v) for k, v in self.stream_times.items()},
            semantics="host receive-time association; device clocks remain separate; SDK return is not arrival",
            frame_gap_semantics="IDs skipped at Dataset sampling rate are not proof of camera drops",
        )


class TimingValidator:
    """Use the same timing rules with bounded per-frame memory and work.

    Full distributions are computed after hardware disconnect, never at raw seal.
    """

    def __init__(self):
        self._sample = TimingSummary()

    def check(self, observation, action=None):
        if "feedback" not in observation:
            raise ValueError("raw frame requires complete feedback timing")
        self._sample.add(observation, action)
        anomalies = (
            "duplicate_frames",
            "backwards_frames",
            "device_time_anomalies",
            "host_time_anomalies",
        )
        if any(v for k, v in self._sample.counters.items() if k.endswith(anomalies)):
            raise ValueError("raw frame timing anomaly; not recorded")
        # Retain only the previous camera tuple and observation clock. No history
        # is freed in one large batch while the control receiver is active.
        self._sample.metrics.clear()
        self._sample.stream_times.clear()
        self._sample.counters.clear()
        self._sample.observation_times[:] = self._sample.observation_times[-1:]


def summarize_events(events):
    events = list(events)
    accepted = {(e["episode_index"], e["attempt"]) for e in events if e["event"] == "episode_saved"}
    all_rows, startup, steady = TimingSummary(), TimingSummary(), TimingSummary()
    missing = 0
    ingest = []
    for event in events:
        if (
            event["event"] == "frame_ingest_returned"
            and (event["episode_index"], event["attempt"]) in accepted
        ):
            duration = event["duration_s"]
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("invalid frame ingest duration")
            ingest.append(duration)
        if (
            event["event"] == "frame_pending"
            and (event["episode_index"], event["attempt"]) not in accepted
        ):
            continue
        if event["event"] not in ("frame_pending", "measurement"):
            continue
        obs, action = event["observation"], event.get("action")
        if "feedback" not in obs:
            missing += 1
            continue
        all_rows.add(obs, action)
        (startup if event.get("phase") == "startup" else steady).add(obs, action)
    return dict(
        all_samples=all_rows.result(),
        startup=startup.result(),
        steady=steady.result(),
        missing_timing_samples=missing,
        frame_ingest_duration_s=distribution(ingest),
    )
