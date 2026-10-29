from types import SimpleNamespace
import numpy as np
import pytest
from test_xbox_pause import session as session, settle
from test_policy_motion import Input, trial
from piper_outcome_stack.policy_completion import (
    PlacementCompletion,
    locate_green_block,
    locate_blue_block,
)

CFG = {
    "placement_polygon_px": [[0, 0], [200, 0], [200, 200], [0, 200]],
    "sample_period_s": 0.1,
    "stable_s": 0.4,
    "max_gap_s": 0.22,
    "max_centre_span_px": 4.0,
    "open_width_m": 0.08,
}
INSIDE = {"box_xywh": [50, 50, 30, 30], "area_pixels": 900, "centre": [65, 65]}
OUTSIDE = {"box_xywh": [300, 50, 30, 30], "area_pixels": 900, "centre": [315, 65]}


def test_requires_transport_opening_and_stationary_object():
    d = PlacementCompletion(CFG)
    for i in range(10):
        assert not d.observe(INSIDE, 0.1, i * 0.1)["candidate_complete"]
    d.observe(OUTSIDE, 0.05, 1.0)
    for i in range(10):
        assert not d.observe(INSIDE, 0.054, 1.1 + i * 0.1)["candidate_complete"]
    # Stability is already established while holding; do not wait another
    # 0.4 seconds after the brief release begins.
    assert d.observe(INSIDE, 0.09, 2.11)["candidate_complete"]
    d.reset()
    assert not d.observe(INSIDE, 0.09, 4.0)["candidate_complete"]


@pytest.mark.parametrize(
    "interruption",
    [None, OUTSIDE, {"box_xywh": [70, 50, 30, 30], "area_pixels": 900, "centre": [85, 65]}],
)
def test_missing_outside_or_moving_detection_breaks_stability(interruption):
    d = PlacementCompletion(CFG)
    d.observe(OUTSIDE, 0.05, 0.0)
    d.observe(INSIDE, 0.05, 0.05)
    for i in range(4):
        d.observe(INSIDE, 0.1, 0.1 + i * 0.1)
    assert not d.observe(interruption, 0.1, 0.45)["candidate_complete"]
    assert not d.observe(INSIDE, 0.1, 0.5)["candidate_complete"]
    assert not d.observe(INSIDE, 0.1, 1.0)["candidate_complete"]


def test_actual_rgb_detection_and_duplicate_frame_not_counted():
    image = np.zeros((480, 640, 3), np.uint8)
    image[50:80, 50:80] = [100, 190, 130]
    obj = locate_green_block(image)
    assert obj["box_xywh"] == [50, 50, 30, 30]
    d = PlacementCompletion(CFG)
    d.observe(OUTSIDE, 0.05, 0.0)
    assert d.update(image, 0.1, 1.0, 1) is not None
    assert d.update(image, 0.1, 2.0, 1) is None
    assert not d.update(image, 0.1, 2.1, 2)["candidate_complete"]


def test_open_object_entering_region_without_closed_phase_does_not_complete():
    d = PlacementCompletion({**CFG, "object_color": "blue", "open_width_m": 0.085})
    d.observe(OUTSIDE, 0.1, 0.0)
    for i in range(1, 12):
        assert not d.observe(INSIDE, 0.1, i * 0.1)["candidate_complete"]
    d.observe(INSIDE, 0.075, 1.2)
    assert d.observe(INSIDE, 0.087, 1.3)["candidate_complete"]


def test_stale_held_history_does_not_authorize_release_after_gap():
    d = PlacementCompletion(CFG)
    d.observe(OUTSIDE, 0.05, 0.0)
    for i in range(1, 8):
        d.observe(INSIDE, 0.05, i * 0.1)
    assert not d.observe(INSIDE, 0.09, 1.5)["candidate_complete"]
    for i in range(1, 8):
        assert not d.observe(INSIDE, 0.09, 1.5 + i * 0.1)["candidate_complete"]


def test_blue_detection_rejects_large_pale_wall_and_green_distractor():
    image = np.zeros((480, 640, 3), np.uint8)
    image[:70, 180:500] = [110, 155, 195]  # Pale blue wall is larger than the object.
    image[250:310, 400:450] = [100, 190, 130]
    image[330:370, 60:100] = [25, 165, 220]
    assert locate_blue_block(image)["box_xywh"] == [60, 330, 40, 40]
    assert locate_green_block(image)["box_xywh"] == [400, 250, 50, 60]
    d = PlacementCompletion({**CFG, "object_color": "blue"})
    d.observe(OUTSIDE, 0.05, 0.0)
    assert d.update(image, 0.09, 0.1, 1)["object"]["box_xywh"] == [60, 330, 40, 40]
    with pytest.raises(ValueError, match="object_color"):
        PlacementCompletion({**CFG, "object_color": "unknown"})


# Existing fake-robot fixture: no real hardware is opened.


@pytest.mark.parametrize("cancel", [None, "LB", "B"])
def test_completion_holds_without_sending_another_policy_action(session, monkeypatch, cancel):
    settle(session)
    robot = session[0]
    original = robot.get_observation
    raw = Input()
    from lerobot_robot_outcome_piper.timing import CaptureTiming

    robot.config.capture_timing = CaptureTiming(
        camera_max_age_s=0.05,
        joint_max_skew_s=0.01,
        image_state_max_skew_s=0.05,
        observation_max_age_s=0.1,
    )

    def observation():
        obs = original()
        robot.last_observation_telemetry["cameras"] = {
            "d435": {"frame_number": robot.last_observation_telemetry["sequence"]}
        }
        return {**obs, "d435": np.zeros((480, 640, 3), np.uint8)}

    monkeypatch.setattr(robot, "get_observation", observation)

    def update(*args):
        if cancel == "LB":
            raw.hold = False
        if cancel == "B":
            raw.b = True
        return {"candidate_complete": True}

    detector = SimpleNamespace(reset=lambda: None, update=update)

    def forbidden(*args):
        pytest.fail("completed candidate must not generate another policy action")

    monkeypatch.setattr("piper_outcome_stack.policy_motion.predict_candidate", forbidden)
    if cancel == "B":
        with pytest.raises(RuntimeError, match="Xbox B"):
            trial(session, raw, completion_detector=detector)
    else:
        result, rows = trial(session, raw, completion_detector=detector)
        assert result["policy_actions_sent"] == 0 and result["hold_confirmed"]
        assert result["end_reason"] == (
            "input_changed_during_completion_check" if cancel else "placement_completion_candidate"
        )
        assert "task_outcome" not in result
