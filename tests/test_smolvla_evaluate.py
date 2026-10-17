import numpy as np

from piper_outcome_stack.smolvla_evaluate import action_errors, command_envelope_counts


def test_padding_is_excluded_and_physical_units_are_kept_separate():
    prediction = np.zeros((1, 3, 7))
    target = np.zeros_like(prediction)
    prediction[0, :2, :6] = np.pi / 180
    prediction[0, :2, 6] = 0.002
    prediction[0, 2] = 999  # Padding must not alter the reported error.
    result = action_errors(prediction, target, [[True, True, False]])
    np.testing.assert_allclose(result["chunk_joint_mae_deg"], np.ones(6))
    assert result["chunk_gripper_mae_mm"] == 2
    assert result["valid_horizon_targets"] == 2


def test_envelope_counts_report_violations_without_clipping_predictions():
    state = np.zeros((3, 7))
    prediction = np.zeros((3, 50, 7))
    prediction[0, 0, 0] = 2.0
    prediction[1, 0, 6] = -0.001
    prediction[2, 0, 6] = 0.03
    original = prediction.copy()
    result = command_envelope_counts(
        state,
        prediction,
        {
            "joint_lower": [-1] * 6,
            "joint_upper": [1] * 6,
            "max_joint_step": [0.1] * 6,
            "gripper_lower": 0,
            "gripper_upper": 0.12,
            "max_gripper_step": 0.015,
        },
    )
    assert result["any_rejection"] == 3
    assert result["joint_box_rejections"] == 1 and result["gripper_box_rejections"] == 1
    np.testing.assert_array_equal(prediction, original)
