from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from piper_policy.smolvla import IMAGE_FEATURE, TASK_A, SmolVLAChunkPredictor, check_contract


def config():
    action = NS(shape=(7,))
    return NS(
        type="smolvla",
        input_features={
            "observation.state": NS(shape=(7,)),
            IMAGE_FEATURE: NS(shape=(3, 480, 640)),
        },
        output_features={"action": action},
        action_feature=action,
        adapt_to_pi_aloha=False,
        use_delta_joint_actions_aloha=False,
        empty_cameras=0,
        n_obs_steps=1,
        chunk_size=50,
        n_action_steps=2,
        use_amp=False,
    )


def test_smolvla_image_scaling_language_and_absolute_output():
    calls = []

    def pre(batch):
        calls.append(batch)
        return batch

    model = NS(
        config=config(), reset=lambda: None, predict_action_chunk=lambda batch: torch.ones(1, 50, 7)
    )
    predictor = SmolVLAChunkPredictor(model, pre, lambda x: x * 2, TASK_A)
    image = np.full((480, 640, 3), 255, np.uint8)
    chunk, index, inferred = predictor.execution_chunk(image, np.arange(7), 1)
    assert inferred and index == 0 and np.all(chunk == 2)
    assert calls[0]["task"] == TASK_A
    assert calls[0][IMAGE_FEATURE].dtype == torch.float32
    assert calls[0][IMAGE_FEATURE].min() == calls[0][IMAGE_FEATURE].max() == 1
    np.testing.assert_array_equal(calls[0]["observation.state"], np.arange(7))
    _, index, inferred = predictor.execution_chunk(image, np.ones(7), 2)
    assert index == 1 and not inferred and len(calls) == 1
    _, index, inferred = predictor.execution_chunk(image, np.ones(7), 3)
    assert index == 0 and inferred and len(calls) == 2
    predictor.configure_temporal_ensemble(None)
    with pytest.raises(ValueError, match="without temporal ensemble"):
        predictor.configure_temporal_ensemble(0.01)


@pytest.mark.parametrize(
    "field,value",
    [("empty_cameras", 2), ("adapt_to_pi_aloha", True), ("use_delta_joint_actions_aloha", True)],
)
def test_wrong_robot_adaptations_are_rejected(field, value):
    cfg = config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match="Aloha"):
        check_contract(cfg)
