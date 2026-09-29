import json
import pytest
from lerobot_robot_outcome_piper.timing_report import TimingSummary, summarize_events
from lerobot_robot_outcome_piper.scene import SceneContext
from piper_outcome_stack.ops.capture_timing import load_read_only_config


def observation(i):
    now = 1 + i * 0.05
    return dict(
        observed_monotonic_s=now,
        feedback=dict(
            received_monotonic_s=[now - 0.004, now - 0.003, now - 0.002, now - 0.001, now - 0.002]
        ),
        cameras={
            "d435": dict(
                frame_number=i * 2,
                device_timestamp_ms=1000 + i * 50,
                timestamp_domain="device",
                received_monotonic_s=now - 0.01,
            )
        },
    )


def test_timing_statistics_keep_retained_commands_out_of_sdk_latency():
    summary = TimingSummary()
    for i in range(3):
        now = 1 + i * 0.05
        action = dict(
            generated_monotonic_s=now + 0.001,
            commands=[],
            retained_gripper_command=dict(started_monotonic_s=0),
        )
        if i == 1:
            action["commands"] = [
                dict(
                    name="move_j",
                    started_monotonic_s=now + 0.002,
                    ended_monotonic_s=now + 0.003,
                    result="sdk_returned",
                )
            ]
        summary.add(observation(i), action)
    r = summary.result()
    assert r["observation_hz"] == pytest.approx(20)
    assert r["metrics"]["observation_to_sdk_start_s"]["count"] == 1
    assert r["counters"]["d435.unselected_or_unobserved_frames"] == 2
    assert r["counters"]["d435.duplicate_frames"] == 0
    assert r["metrics"]["d435.frame_age_s"]["max"] == pytest.approx(0.01)


def test_timing_anomalies_and_discarded_attempts_are_not_hidden():
    a = observation(0)
    b = observation(1)
    b["cameras"]["d435"]["frame_number"] = 0
    b["cameras"]["d435"]["timestamp_domain"] = "changed"
    s = TimingSummary()
    s.add(a)
    s.add(b)
    assert s.result()["counters"]["d435.duplicate_frames"] == 1
    assert s.result()["counters"]["d435.device_time_anomalies"] == 1
    with pytest.raises(ValueError, match="advance"):
        s.add(b)
    events = [
        dict(event="frame_pending", episode_index=0, attempt=0, observation=a),
        dict(event="frame_pending", episode_index=0, attempt=1, observation=b),
        dict(event="episode_saved", episode_index=0, attempt=1),
    ]
    assert summarize_events(events)["all_samples"]["samples"] == 1


def test_future_timing_and_invalid_action_interval_fail():
    a = observation(0)
    a["feedback"]["received_monotonic_s"][0] = 2
    with pytest.raises(ValueError):
        TimingSummary().add(a)
    a = observation(0)
    with pytest.raises(ValueError, match="precedes"):
        TimingSummary().add(a, dict(generated_monotonic_s=0))
    with pytest.raises(ValueError, match="SDK"):
        TimingSummary().add(a, dict(commands=[dict(started_monotonic_s=2, ended_monotonic_s=1)]))


def test_measurement_rejects_motion_config_without_constructing_robot(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps(
            dict(
                can_interface="can0",
                firmware="v189",
                feedback_timeout_s=0.2,
                execution_mode="motion",
                safety_path="/synthetic.json",
            )
        )
    )
    with pytest.raises(ValueError, match="read_only"):
        load_read_only_config(p)
    with pytest.raises(ValueError, match="scene_id"):
        SceneContext("", "base", "view", "area")
