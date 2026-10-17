from types import SimpleNamespace as NS
import numpy as np

from piper_outcome_stack.smolvla_shadow import compare_pair


def test_three_methods_use_same_snapshot_noise_and_hypothetical_projected_prefix(monkeypatch):
    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    calls = []

    class Predictor:
        def configure_rtc(self, enabled, **kwargs):
            self.mode = (enabled, kwargs)

        def predict_chunk(self, image, state, **kwargs):
            calls.append((self.mode, image.copy(), state.copy(), kwargs))
            raw = np.zeros((50, 7), np.float32)
            raw[:, 6] = -0.001
            projected = raw.copy()
            projected[:, 6] = 0
            committed = np.ones((50, 7), np.float32) * -1.7
            return NS(
                actions=raw,
                model_actions=raw.copy(),
                execution_actions=projected,
                committed_model_actions=committed,
            )

    pair = {
        "first_image": np.zeros((480, 640, 3), np.uint8),
        "second_image": np.ones((480, 640, 3), np.uint8),
        "first_state": np.zeros(7),
        "second_state": np.ones(7),
    }
    result = compare_pair(Predictor(), pair, seed=42)
    assert len(calls) == 4
    assert "targets" not in result
    np.testing.assert_array_equal(calls[1][3]["noise"], calls[2][3]["noise"])
    np.testing.assert_array_equal(calls[2][3]["noise"], calls[3][3]["noise"])
    assert "previous_model_actions" not in calls[1][3]
    for _, image, state, options in calls[2:]:
        assert np.all(image == 1) and np.all(state == 1)
        np.testing.assert_array_equal(
            options["previous_model_actions"], result["old_committed"][25:]
        )
        assert options["overlap_steps"] == 25 and options["inference_delay"] == 19
    assert np.all(result["old_raw"][:, 6] < 0) and np.all(result["old_execution"][:, 6] == 0)
