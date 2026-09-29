from dataclasses import asdict, replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from lerobot_robot_outcome_piper.continuous_reference import ReferenceSettings, reference_candidate
from lerobot_robot_outcome_piper.teleop_control import TeleopControl, TeleopState
from lerobot_robot_outcome_piper.processor import OutcomePiperAction
from test_xbox_pause import session as session, settle
from test_plugin import valid_action, xbox_config

S = ReferenceSettings(0.05, 0.2, 2.0)


def candidate(previous=None, now=0.0, position=None, rotation=None, xyz=None, rv=None):
    return reference_candidate(
        previous,
        [0, 0, 0] if position is None else position,
        np.eye(3) if rotation is None else rotation,
        np.eye(3),
        [0.0025, 0, 0] if xyz is None else xyz,
        [0, 0, 0.01] if rv is None else rv,
        0.0025,
        0.01,
        S,
        now,
    )


def test_ramp_continuous_integration_and_base_rotation():
    prev = None
    pos = [0, 0, 0]
    rot = np.eye(3)
    vel = []
    for i in range(10):
        value = candidate(prev, i * 0.05, pos, rot)
        vel.append(value["velocity"][0])
        assert not value["lead_limited"]
        pos, rot = value["position"], value["rotation"]
        prev = value
    assert vel[:4] == pytest.approx([0.0125, 0.025, 0.0375, 0.05])
    assert np.allclose(Rotation.from_matrix(rot).as_rotvec()[:2], 0)
    assert all(v <= 0.05 + 1e-12 for v in vel)


def test_stationary_feedback_bounds_reference_without_accumulating_input():
    prev = None
    for i in range(100):
        prev = candidate(prev, i * 0.05)
    assert prev["position"][0] <= 0.005 + 1e-10
    assert Rotation.from_matrix(prev["rotation"]).magnitude() <= 0.02 + 1e-10
    assert prev["lead_limited"] and np.linalg.norm(prev["velocity"]) < 1e-6
    resumed = candidate(prev, 100.0)
    assert resumed["dt_s"] == S.period_s  # No integration of the long missing interval.


def test_stop_epoch_resets_reference_and_rejects_old_revision():
    c = TeleopControl()
    c.state = TeleopState.RUNNING
    c.epoch = 1
    plan = {"base_revision": 0, "reference": candidate()}
    assert c.commit_reference(1, plan)
    assert not c.reference_valid(1, plan)
    c.request_hold()
    assert c.reference_snapshot(c.epoch)[0] is None
    assert not c.commit_reference(1, plan)
    c.stop(True)
    assert c.reference_snapshot(c.epoch)[0] is None


@pytest.mark.parametrize(
    "outcome", ["success", "failure", "gripper_failure", "release", "stale", "B"]
)
def test_reference_commits_only_after_valid_sdk_success(session, outcome):
    robot, arm, c, clock = session
    settle(session)
    _, epoch = c.observe(True, True)
    a = OutcomePiperAction(valid_action(0.01, 0.035), intent="run", epoch=epoch)
    a.generated_monotonic_s = clock.now
    a.reference_plan = {
        "base_revision": c.reference_revision,
        "reference": candidate(now=clock.now),
    }
    if outcome == "failure":
        arm.move_j = lambda q: (_ for _ in ()).throw(OSError("SDK failed"))
    if outcome == "gripper_failure":
        arm.gripper.move_gripper_m = lambda *a, **kw: (_ for _ in ()).throw(OSError("SDK failed"))
    if outcome == "release":
        original = arm.move_j

        def release(q):
            original(q)
            c.request_hold()

        arm.move_j = release
    if outcome == "stale":
        c.request_hold()
    if outcome == "B":
        robot.request_emergency_stop("B")
    if outcome in ("failure", "gripper_failure", "B"):
        with pytest.raises(RuntimeError):
            robot.send_action(a)
    else:
        robot.send_action(a)
    assert (c.reference_state is not None) == (outcome == "success")


def test_config_explicit_reference_period():
    cfg = replace(xbox_config(), streaming_reference=asdict(S))
    assert cfg.streaming_reference["ramp_time_s"] == 0.2
    with pytest.raises(ValueError, match="control_hz"):
        replace(cfg, streaming_reference=asdict(replace(S, period_s=0.1)))


def test_processor_does_not_commit_before_dispatch(monkeypatch):
    from test_processor import processor, observation, raw_action, require_sdk_kinematics
    from lerobot.types import TransitionKey

    k = require_sdk_kinematics()
    p = processor(streaming_reference=asdict(S))
    monkeypatch.setattr(k, "fk_from_mdh", lambda *_: [0.1, 0.1, 0.3, 0, 0, 0])
    captured = []
    monkeypatch.setattr(p, "_solve", lambda q, t: captured.append(t) or q)
    p._current_transition = {TransitionKey.OBSERVATION: observation([0.1] * 6)}
    a = p.action(raw_action(dx=0.01))
    assert a.reference_plan is not None and p.control.reference_state is None
    assert captured[0][0] == pytest.approx(0.1025)
    assert len(a) == 7 and "reference_plan" not in a
    p.action(raw_action())
    assert not p.control.reference_valid(a.epoch, a.reference_plan)


def test_reference_action_record_finalize_reload_audit(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from lerobot.processor import RobotActionProcessorStep, RobotProcessorPipeline
    from lerobot.processor.converters import (
        robot_action_observation_to_transition,
        transition_to_robot_action,
    )
    from lerobot_robot_outcome_piper.action_audit import verify_telemetry
    from test_recording_telemetry import offline_fixture_record as record_with_telemetry
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from test_recording_telemetry import devices, configuration

    robot, _, _, _ = devices(monkeypatch)
    generated = candidate()
    values = valid_action(0.01 + generated["position"][0], 0.035)
    action = OutcomePiperAction(values, intent="run", epoch=1)
    action.reference_plan = {"base_revision": 0, "reference": generated}

    @dataclass
    class ReferenceStep(RobotActionProcessorStep):
        def action(self, raw):
            return action

        def transform_features(self, features):
            return features

    pipeline = RobotProcessorPipeline(
        steps=[ReferenceStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    original = robot.send_action

    def send(received):
        assert received.reference_plan == action.reference_plan
        result = original(received)
        robot.last_action_telemetry["reference_plan"] = received.reference_plan
        return result

    robot.send_action = send
    cfg = configuration(tmp_path / "dataset", repo_id="local/continuous-reference")
    record_with_telemetry(cfg, teleop_action_processor=pipeline)
    loaded = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)
    assert verify_telemetry(loaded.root, loaded) == {"episodes": 1, "frames": 1}
    np.testing.assert_allclose(loaded[0]["action"], list(values.values()), atol=1e-8)
    assert loaded.features["action"]["shape"] == (7,)


def test_processor_reset_discards_velocity_and_invalidates_pending_reference():
    from test_processor import processor

    p = processor(streaming_reference=asdict(S))
    c = p.control
    plan = {"base_revision": c.reference_revision, "reference": candidate()}
    assert c.commit_reference(c.epoch, plan)
    pending = {"base_revision": c.reference_revision, "reference": candidate()}
    p.reset()
    assert c.reference_snapshot(c.epoch)[0] is None
    assert not c.reference_valid(c.epoch, pending)
