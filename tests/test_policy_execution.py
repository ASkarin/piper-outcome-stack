from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from piper_outcome_stack.policy_execution import (
    ACTChunkPredictor,
    check_policy_observation,
    predict_candidate,
    run_shadow,
    verify_reference_inputs,
)
from lerobot_robot_outcome_piper.safety import ACTION_KEYS


def predictor():
    cached = {}

    def pre(batch):
        cached["anchor"] = batch["observation.state"].clone()
        return batch

    def post(chunk):
        return chunk + cached["anchor"]

    model = NS(
        config=NS(
            chunk_size=3,
            input_features={
                "observation.images.d435": NS(shape=(3, 2, 2)),
            },
        ),
        predict_action_chunk=lambda batch: torch.full((1, 3, 7), 0.001),
    )
    return ACTChunkPredictor(model, pre, post)


def test_entire_chunk_keeps_original_anchor_after_next_observation():
    p = predictor()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    first = p.predict(image, np.zeros(7))
    second = p.predict(image, np.ones(7))
    np.testing.assert_allclose(first, 0.001)
    np.testing.assert_allclose(second, 1.001)
    second[0] = 10
    np.testing.assert_allclose(first, 0.001)


@pytest.mark.parametrize(
    "image,state",
    [
        (np.zeros((2, 2, 3), dtype=float), np.zeros(7)),
        (np.zeros((3, 2, 2), dtype=np.uint8), np.zeros(7)),
        (np.zeros((2, 2, 3), dtype=np.uint8), [float("nan")] * 7),
    ],
)
def test_invalid_inputs_fail(image, state):
    with pytest.raises(ValueError):
        predictor().predict(image, state)


def test_expiry_future_and_unchecked_input():
    timing = NS(observation_max_age_s=0.1)
    meta = dict(quality="checked", oldest_received_monotonic_s=1.0)
    assert check_policy_observation(meta, timing, 1.05) == pytest.approx(0.05)
    for now in (0.9, 1.101, float("nan")):
        with pytest.raises(ValueError):
            check_policy_observation(meta, timing, now)
    with pytest.raises(ValueError):
        check_policy_observation({**meta, "quality": "measurement_only"}, timing, 1.05)


def observation():
    return {**dict(zip(ACTION_KEYS, [0.0] * 7)), "d435": np.zeros((2, 2, 3), np.uint8)}


def test_expiration_during_inference_rejects_before_return(monkeypatch):
    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.execution_constraints.check_execution_target", lambda *a: None
    )
    clock = iter((1.0, 1.11, 1.12))
    with pytest.raises(ValueError, match="expired") as caught:
        predict_candidate(
            predictor(),
            observation(),
            dict(quality="checked", oldest_received_monotonic_s=1.0, sequence=1),
            NS(observation_max_age_s=0.1),
            None,
            lambda: next(clock),
        )
    assert caught.value.policy_timing["inference_s"] == pytest.approx(0.11)
    assert caught.value.policy_timing["selection_and_validation_s"] == pytest.approx(0.01)
    assert caught.value.policy_timing["age_at_rejection_s"] == pytest.approx(0.12)


def test_limits_rejection_propagates_without_clipping(monkeypatch):
    seen = []

    def reject(state, target, safety):
        seen.append(target.copy())
        raise ValueError("outside workspace")

    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.execution_constraints.check_execution_target", reject
    )
    with pytest.raises(ValueError, match="outside workspace"):
        predict_candidate(
            predictor(),
            observation(),
            dict(quality="checked", oldest_received_monotonic_s=1.0, sequence=1),
            NS(observation_max_age_s=0.1),
            None,
            lambda: 1.0,
        )
    np.testing.assert_allclose(seen[0], 0.001)


def test_shadow_uses_fresh_observations_and_never_dispatches(monkeypatch):
    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.execution_constraints.check_execution_target", lambda *a: None
    )

    class Robot:
        config = NS(execution_mode="read_only", capture_timing=NS(observation_max_age_s=0.1))
        count = 0

        def get_observation(self):
            self.count += 1
            self.last_observation_telemetry = dict(
                quality="checked",
                oldest_received_monotonic_s=1.0,
                sequence=self.count,
            )
            return observation()

        def send_action(self, *a):
            pytest.fail("shadow must not send")

        def enable(self):
            pytest.fail("shadow must not enable")

    robot, rows = Robot(), []
    run_shadow(robot, predictor(), None, 3, 50, rows, lambda: 1.0, lambda _: None)
    assert robot.count == 3
    assert [r["observation_sequence"] for r in rows] == [1, 2, 3]
    robot.config = NS(execution_mode="motion", capture_timing=robot.config.capture_timing)
    with pytest.raises(ValueError, match="read_only"):
        run_shadow(robot, predictor(), None, 3, 50, [], lambda: 1.0, lambda _: None)


def _reference(tmp_path, n=2, offset=0.0):
    p = predictor()
    images = np.zeros((n, 2, 2, 3), np.uint8)
    states = np.zeros((n, 7), np.float32)
    expected = np.stack([p.predict(images[i], states[i]) for i in range(n)]) + offset
    path = tmp_path / "reference.npz"
    np.savez(path, images=images, states=states, expected_absolute_chunks=expected)
    return p, path


def test_reference_inputs_report_rejections_without_raising(tmp_path, monkeypatch):
    import lerobot_robot_outcome_piper.execution_constraints as constraints

    def check(state, target, safety):
        raise ValueError("outside workspace")

    monkeypatch.setattr(constraints, "check_execution_target", check)
    p, path = _reference(tmp_path)
    images, states, rows = verify_reference_inputs(p, path, None, warmup=3)
    assert len(images) == len(states) == 2
    assert [r["recorded_target_check"]["status"] for r in rows] == ["rejected", "rejected"]


def test_reference_inputs_reject_mismatch_and_empty(tmp_path, monkeypatch):
    import lerobot_robot_outcome_piper.execution_constraints as constraints

    monkeypatch.setattr(constraints, "check_execution_target", lambda *a: None)
    p, path = _reference(tmp_path, offset=1e-3)
    with pytest.raises(ValueError, match="reference mismatch"):
        verify_reference_inputs(p, path, None)
    empty = tmp_path / "empty.npz"
    np.savez(
        empty,
        images=np.zeros((0, 2, 2, 3), np.uint8),
        states=np.zeros((0, 7)),
        expected_absolute_chunks=np.zeros((0, 3, 7)),
    )
    with pytest.raises(ValueError, match="non-empty"):
        verify_reference_inputs(p, empty, None)
