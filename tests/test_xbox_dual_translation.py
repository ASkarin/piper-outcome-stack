from action_evidence import dispatched
import numpy as np
import pytest
from lerobot_robot_outcome_piper.teleop_control import (
    TeleopControl,
    TranslationStrategy,
    TeleopMode,
)
from lerobot_robot_outcome_piper.processor import input_deltas
from lerobot_robot_outcome_piper.position_ik import solve_position, grasp_position
from test_processor import safety


def test_x_edges_and_conflict_invalidate_reference():
    c = TeleopControl()
    c.confirm_hold()
    c.observe(False, True, translation_switch=True)
    assert c.translation_strategy is TranslationStrategy.WRIST_PRIORITY
    c.observe(False, True)
    before = c.epoch
    c.observe(False, True, translation_switch=True)
    assert c.translation_strategy is TranslationStrategy.FIXED_ORIENTATION
    assert c.epoch > before
    c.observe(False, True, translation_switch=True)
    assert c.mode_event is None
    c.observe(False, True)
    c.observe(False, True, mode_switch=True, translation_switch=True)
    assert not c.mode_event["accepted"] and c.mode is TeleopMode.TRANSLATION
    c.observe(False, True)
    c.observe(False, True, mode_switch=True)
    assert c.mode is TeleopMode.ORIENTATION
    c.observe(False, True)
    c.observe(False, True, translation_switch=True)
    assert not c.mode_event["accepted"]


def test_default_yaw_unused_but_fixed_orientation_retains_it():
    raw = dict(stick_x=0, stick_y=0, stick_z=0, stick_yaw=1, left_trigger=0, right_trigger=0)
    assert input_deltas(raw, TeleopMode.TRANSLATION, 0.005, 0.01, 0.008)[1] == [0, 0, 0]
    assert input_deltas(
        raw, TeleopMode.TRANSLATION, 0.005, 0.01, 0.008, TranslationStrategy.FIXED_ORIENTATION
    )[1] == [0, 0, 0.01]


@pytest.mark.parametrize(
    "degrees",
    [
        [53.217, 88.246, -37.998, 23.157, 1.832, -25.515],
        [54.811, 99.327, -62.038, 1.329, 43.671, -22.818],
    ],
)
def test_real_grasp_down_fixed_wrist(degrees):
    from dataclasses import replace
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

    q = np.deg2rad(degrees)
    s = replace(
        safety(),
        joint_lower=tuple(np.deg2rad([-154, 0, -175, -102, -75, -170])),
        joint_upper=tuple(np.deg2rad([154, 195, 0, 102, 75, 170])),
    )
    target = grasp_position(fk_from_mdh(list(get_mdh("piper")), q.tolist())) + [0, 0, -0.005]
    goal, detail = solve_position(q, q, target, s, 0.03, 0.03, timeout=0.2, max_nfev=100)
    assert detail["level"] == "fixed_wrist"
    assert goal[3:] == pytest.approx(q[3:], abs=1e-12)
    assert np.linalg.norm(grasp_position(fk_from_mdh(list(get_mdh("piper")), goal)) - target) < 1e-5


def test_raw_seal_discard_and_resume(tmp_path):
    from lerobot_robot_outcome_piper.xbox_raw import RawFrames
    import json

    root = tmp_path / "raw"
    kwargs = dict(fps=20, features={"image": {"dtype": "video"}}, robot_type="outcome_piper")
    sink = RawFrames(root, **kwargs)
    sink.prepare_episode().result()
    sink.add_frame({"image": np.zeros((4, 4, 3), np.uint8), "action": np.zeros(7)})
    sink.save_episode()
    assert sink.num_episodes == 1
    sink.prepare_episode().result()
    sink.add_frame({"image": np.ones((4, 4, 3), np.uint8), "action": np.ones(7)})
    sink.clear_episode_buffer()
    sink.finalize()
    sink.finish_capture(True)
    assert sorted(
        json.loads(p.read_text())["status"] for p in root.glob("attempts/*/result.json")
    ) == ["discarded", "saved"]
    resumed = RawFrames(root, resume=True, **kwargs)
    assert resumed.num_episodes == 1
    resumed.finalize()
    resumed.finish_capture(True)


def test_weighted_stage_and_deadline(monkeypatch):
    from types import SimpleNamespace
    from lerobot_robot_outcome_piper import position_ik
    from lerobot_robot_outcome_piper.errors import OutcomePiperControlTimeout
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

    q = np.array([0.2, 0.5, -0.4, 0.2, 0.5, -0.2])
    target = grasp_position(fk_from_mdh(list(get_mdh("piper")), q.tolist())) + [0, 0, -0.001]
    monkeypatch.setattr(
        position_ik, "least_squares", lambda fun, x, **kw: SimpleNamespace(success=False, x=x)
    )
    goal, detail = solve_position(q, q, target, safety(), 0.03, 0.03, timeout=1, max_nfev=100)
    assert detail["level"] == "weighted_wrist" and detail["position_error_m"] < 1e-5
    with pytest.raises(OutcomePiperControlTimeout):
        solve_position(q, q, target, safety(), 0.03, 0.03, timeout=1e-12, max_nfev=100)


def test_raw_official_conversion_two_episodes_n_to_n(tmp_path):
    from types import SimpleNamespace
    from lerobot_robot_outcome_piper.xbox_raw import RawFrames, convert
    from lerobot_robot_outcome_piper.recording import TelemetryDataset
    from lerobot_robot_outcome_piper.safety import ACTION_KEYS

    features = {
        "observation.state": dict(dtype="float32", shape=(7,), names=list(ACTION_KEYS)),
        "action": dict(dtype="float32", shape=(7,), names=list(ACTION_KEYS)),
        "observation.images.d435": dict(
            dtype="image", shape=(16, 16, 3), names=["height", "width", "channels"]
        ),
    }
    sink = RawFrames(
        tmp_path / "raw",
        fps=20,
        features=features,
        robot_type="outcome_piper",
        conversion_options={"use_videos": False},
    )
    robot = SimpleNamespace(config=SimpleNamespace(scene=None), last_depth_frames={})
    audit = TelemetryDataset(sink, robot)
    for ep in range(2):
        audit.prepare_episode().result()
        for i in range(3):
            n = ep * 3 + i
            robot.last_observation_telemetry = {
                "sequence": n,
                "quality": "checked",
                "observed_monotonic_s": n * 0.05,
                "cameras": {},
            }
            values = {k: float(n) * 0.001 for k in ACTION_KEYS}
            robot.last_action_telemetry = dispatched(values, n)
            audit.add_frame(
                {
                    "observation.state": np.zeros(7, np.float32),
                    "action": np.array(list(values.values()), np.float32),
                    "observation.images.d435": np.full((16, 16, 3), n, np.uint8),
                    "task": "synthetic",
                }
            )
        audit.save_episode()
    sink.finalize()
    audit.complete()
    audit.close()
    sink.finish_capture(True)
    report = convert(sink.root, tmp_path / "converted", "local/raw-test")
    assert report["frames"] == 6 and report["episodes"] == 2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/raw-test", root=tmp_path / "converted")
    assert float(ds[5]["action"][0]) == pytest.approx(0.005)
    with pytest.raises(FileExistsError):
        convert(sink.root, tmp_path / "converted", "local/raw-test")


def test_raw_writer_failure_and_running_conversion_rejected(tmp_path, monkeypatch):
    from lerobot_robot_outcome_piper.xbox_raw import RawFrames, convert

    sink = RawFrames(
        tmp_path / "raw", fps=20, features={"image": {"dtype": "image"}}, robot_type="outcome_piper"
    )
    with pytest.raises(ValueError, match="closed"):
        convert(sink.root, tmp_path / "out", "local/test")
    monkeypatch.setattr(np, "save", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    sink.prepare_episode().result()
    sink.add_frame({"image": np.zeros((3, 3, 3), np.uint8)})
    with pytest.raises(OSError, match="disk full"):
        sink.save_episode()
    with pytest.raises(OSError):
        sink.finalize()
    sink.finish_capture(False)
    import json

    assert all(
        json.loads(p.read_text())["status"] != "saved"
        for p in sink.root.glob("attempts/*/result.json")
    )


def test_pause_candidates_never_delete_and_exclude_moving_feedback(tmp_path):
    import json
    from lerobot_robot_outcome_piper.pause_review import candidates

    raw = tmp_path / "raw"
    (raw / "attempts" / "one").mkdir(parents=True)
    (raw / "telemetry" / "one").mkdir(parents=True)
    (raw / "raw.json").write_text("{}")
    (raw / "capture.json").write_text(json.dumps({"status": "closed"}))
    attempt = raw / "attempts" / "one"
    (attempt / "result.json").write_text(json.dumps(dict(status="saved", episode_index=0)))
    states = []
    events = []
    for i in range(65):
        state = [0.0] * 7
        if i >= 30:
            state[0] = i * 0.01
        states.append(dict(frame_index=i, values={"observation.state": state}))
        events.append(
            dict(
                event="frame_pending",
                episode_index=0,
                attempt=0,
                frame_index=i,
                observation={"observed_monotonic_s": i * 0.05},
                action=dict(
                    result="holding",
                    hold_confirmed=True,
                    hold_id=1,
                    retained_gripper_command={"target": 0.03},
                ),
            )
        )
    (attempt / "frames.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
    events.append(dict(event="episode_saved", episode_index=0, attempt=0))
    (raw / "telemetry" / "one" / "events.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in events)
    )
    before = (attempt / "frames.jsonl").read_bytes()
    report = candidates(raw)
    assert report["removed_frames"] == 0 and len(report["candidates"]) == 1
    assert report["candidates"][0]["last_frame"] == 29
    assert not report["candidates"][0]["selected_for_removal"]
    assert (attempt / "frames.jsonl").read_bytes() == before
