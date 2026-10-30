import cv2
import numpy as np
import pytest

from piper_outcome_stack.apriltag_static import detections_for_image, make_detector, summarize


def test_two_exact_25h9_markers_are_assigned_by_id():
    image = np.full((200, 360), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
    image[40:160, 30:150] = cv2.aruco.generateImageMarker(dictionary, 2, 120)
    image[40:160, 210:330] = cv2.aruco.generateImageMarker(dictionary, 1, 120)
    detections = detections_for_image(
        cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), "rgb8", make_detector()
    )
    assert {tag["id"] for tag in detections} == {1, 2}
    summary = summarize([dict(detections=detections)], reference_id=2, tool_id=1)
    assert summary["both_visible_frames"] == 1
    assert summary["tag_statistics"]["1"]["centre_std_pixels"] is None
    assert not summary["pose_estimated"] and not summary["rigid_mount_verified"]


def test_missing_tags_are_reported_not_counted_as_passed():
    image = np.full((100, 100, 3), 255, np.uint8)
    assert detections_for_image(image, "rgb8", make_detector()) == []
    summary = summarize([dict(detections=[])], reference_id=2, tool_id=1)
    assert summary["both_visible_frames"] == 0
    assert summary["tag_statistics"]["2"]["frames_detected"] == 0
    assert summary["tag_statistics"]["2"]["minimum_edge_pixels"] is None


def test_unsupported_image_format_fails():
    detector = make_detector()
    with pytest.raises(ValueError, match="uint8"):
        detections_for_image(np.zeros((100, 100), np.uint16), "rgb8", detector)
    with pytest.raises(ValueError, match="pixel format"):
        detections_for_image(np.zeros((100, 100, 3), np.uint8), "z16", detector)
