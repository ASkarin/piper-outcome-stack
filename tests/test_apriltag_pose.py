import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from piper_outcome_stack.apriltag_pose import (
    square_points,
    pose_solutions,
    relative_pose,
    camera_matrix,
)


def project(rotation, translation, k):
    return cv2.projectPoints(
        square_points(0.024),
        Rotation.from_matrix(rotation).as_rotvec(),
        np.asarray(translation, float),
        k,
        np.zeros(5),
    )[0].reshape(4, 2)


def test_known_two_tag_geometry_and_both_planar_hypotheses():
    k = np.array([[1360.0, 0, 960.0], [0, 1361.0, 540.0], [0, 0, 1.0]])
    r0 = Rotation.from_euler("xyz", [2.4, 0.3, 0.2]).as_matrix()
    t0 = np.array([-0.2, 0.1, 1.1])
    r1 = Rotation.from_euler("xyz", [2.8, -0.2, -0.1]).as_matrix()
    t1 = np.array([0.1, -0.1, 0.8])
    a = pose_solutions(project(r0, t0, k), 0.024, k)
    b = pose_solutions(project(r1, t1, k), 0.024, k)
    assert len(a) == len(b) == 2 and all(v["positive_corner_depths"] for v in a + b)
    np.testing.assert_allclose(a[0]["translation_camera_m"], t0, atol=1e-11)
    rel = relative_pose(a[0], b[0])
    np.testing.assert_allclose(rel["translation_reference_m"], r0.T @ (t1 - t0), atol=1e-11)
    np.testing.assert_allclose(rel["rotation_matrix"], r0.T @ r1, atol=1e-10)
    assert a[1]["corner_reprojection_rms_pixels"] > a[0]["corner_reprojection_rms_pixels"]


def test_metric_scale_and_inverse_distortion_boundary():
    k = np.array([[1360.0, 0, 960.0], [0, 1361.0, 540.0], [0, 0, 1.0]])
    r = Rotation.from_euler("xyz", [2.5, 0.3, 0.2]).as_matrix()
    t = np.array([0.1, -0.1, 1.0])
    corners = project(r, t, k)
    a = pose_solutions(corners, 0.024, k)
    b = pose_solutions(corners, 0.048, k)
    np.testing.assert_allclose(
        b[0]["translation_camera_m"], 2 * np.asarray(a[0]["translation_camera_m"])
    )
    metadata = dict(
        rotation=None,
        alignment="native",
        intrinsics=dict(fx=1360, fy=1361, ppx=960, ppy=540, coeffs=[0.01, 0, 0, 0, 0]),
    )
    with pytest.raises(ValueError, match="nonzero distortion"):
        camera_matrix(metadata)
