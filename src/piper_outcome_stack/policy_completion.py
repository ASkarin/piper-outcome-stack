"""Fixed-scene green-block placement candidate; separate from operator outcome labels."""

from collections import deque
import math
import time

import cv2
import numpy as np


def locate_green_block(image):
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError("completion requires the configured 640x480 RGB uint8 image")
    a = image[::2, ::2].astype(np.int16)
    red, green, blue = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mask = (
        (green - red > 12) & (green - blue > 8) & (green > 1.08 * red) & (green > 1.03 * blue)
    ).astype(np.uint8)
    n, _, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
    candidates = [i for i in range(1, n) if stats[i, 4] >= 20]
    if not candidates:
        return None
    i = max(candidates, key=lambda i: stats[i, 4])
    return {
        "box_xywh": (stats[i, :4] * 2).tolist(),
        "area_pixels": int(stats[i, 4] * 4),
        "centre": (centres[i] * 2).tolist(),
    }


class PlacementCompletion:
    """Requires observed transport into the placement region, opening and stability.

    A candidate ends motion through the existing confirmed-hold path. It does not
    certify task success; outcome annotation remains an independent operator step.
    """

    def __init__(self, config):
        self.config = dict(config)
        self.polygon = np.asarray(config["placement_polygon_px"], np.float32)
        if self.polygon.shape != (4, 2) or not np.isfinite(self.polygon).all():
            raise ValueError("placement polygon must contain four finite pixel coordinates")
        for key in (
            "sample_period_s",
            "stable_s",
            "max_gap_s",
            "max_centre_span_px",
            "open_width_m",
        ):
            if not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"invalid completion parameter {key}")
        self.reset()

    def reset(self):
        self.seen_outside = False
        self.window = deque()
        self.next_due = float("-inf")
        self.last_sequence = None

    def update(self, image, gripper_m, now, sequence):
        if now < self.next_due or sequence == self.last_sequence:
            return None
        self.next_due = now + self.config["sample_period_s"]
        self.last_sequence = sequence
        started = time.perf_counter()
        result = self.observe(locate_green_block(image), gripper_m, now)
        result["processing_s"] = time.perf_counter() - started
        return result

    def observe(self, obj, gripper_m, now):
        """Geometry/state stage, also used with recorded extracted detections."""
        if not math.isfinite(now) or not math.isfinite(gripper_m):
            raise ValueError("completion requires finite time and gripper feedback")
        inside = False
        if obj is not None:
            x, y, w, h = obj["box_xywh"]
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
            inside = all(
                cv2.pointPolygonTest(self.polygon, (float(a), float(b)), False) >= 0
                for a, b in corners
            )
            if not inside:
                self.seen_outside = True
        eligible = self.seen_outside and inside and gripper_m >= self.config["open_width_m"]
        if not eligible:
            self.window.clear()
        else:
            if self.window and now - self.window[-1][0] > self.config["max_gap_s"]:
                self.window.clear()
            self.window.append((now, obj["centre"]))
            while len(self.window) > 1 and self.window[1][0] <= now - self.config["stable_s"]:
                self.window.popleft()
            centres = np.array([c for _, c in self.window])
            if np.ptp(centres, axis=0).max() > self.config["max_centre_span_px"]:
                self.window.clear()
                self.window.append((now, obj["centre"]))
        stable = now - self.window[0][0] if self.window else 0.0
        complete = len(self.window) >= 4 and stable >= self.config["stable_s"]
        return {
            "candidate_complete": bool(complete),
            "object": obj,
            "inside_placement_region": bool(inside),
            "seen_outside": self.seen_outside,
            "observed_gripper_m": float(gripper_m),
            "stable_s": float(stable),
        }
