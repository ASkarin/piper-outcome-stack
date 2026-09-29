"""Official RealSense configuration/processing with atomic RGB/depth publication."""

import math
import threading
import time
import numpy as np
import cv2
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from .camera import CameraTelemetry, read_new_frame, validate_frame


def frame_intrinsics(frame):
    intr = frame.profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
        "reference": "native_before_image_rotation",
    }


class TimedRealSenseCamera(RealSenseCamera):
    def __init__(self, config):
        super().__init__(config)
        self.metadata_condition = threading.Condition(self.frame_lock)
        self.latest_metadata = {}
        self.latest_frames = {}
        self.consumed_frame_number = None
        self.capture_error = None
        self._warming_up = False
        self.max_frame_age_s = None
        self.wait_service = None
        self.last_read_diagnostics = {}

    def connect(self, warmup=True):
        # RealSense startup can reuse depth frames and restart its frame counter.
        # This phase finishes before any caller can consume observations.
        self._warming_up = True
        try:
            return super().connect(warmup=warmup)
        finally:
            self._warming_up = False

    def _postprocess_image(self, image, depth_frame=False):
        if depth_frame:
            # The pinned upstream method applies RGB/BGR conversion even to depth.
            # Numeric depth needs only dimension validation and the same rotation.
            if image.shape != (self.capture_height, self.capture_width):
                raise RuntimeError("depth dimensions differ from the configured profile")
            return cv2.rotate(image, self.rotation) if self.rotation is not None else image
        return super()._postprocess_image(image)

    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                frameset = self._read_from_hardware()
                received = time.monotonic()
                raw = {}
                if self.use_rgb:
                    raw["color"] = frameset.get_color_frame()
                if self.use_depth:
                    raw["depth"] = frameset.get_depth_frame()
                if any(not frame for frame in raw.values()):
                    raise RuntimeError("a configured camera stream is missing")
                images, metadata = {}, {}
                extrinsics = None
                if self.use_rgb and self.use_depth:
                    ext = raw["depth"].profile.get_extrinsics_to(raw["color"].profile)
                    extrinsics = {
                        "rotation_column_major": list(ext.rotation),
                        "translation_m": list(ext.translation),
                    }
                for stream, frame in raw.items():
                    array = np.asanyarray(frame.get_data())
                    depth = stream == "depth"
                    if depth and (array.dtype != np.uint16 or array.ndim != 2):
                        raise RuntimeError("depth must be native uint16 Z16")
                    images[stream] = self._postprocess_image(array, depth_frame=depth).copy()
                    scale = float(frame.get_units()) if depth else None
                    if depth and (not math.isfinite(scale) or scale <= 0):
                        raise RuntimeError("invalid device depth scale")
                    metadata[stream] = CameraTelemetry(
                        int(frame.get_frame_number()),
                        float(frame.get_timestamp()),
                        str(frame.get_frame_timestamp_domain()),
                        received,
                        time.monotonic(),
                        stream=stream,
                        pixel_format="z16"
                        if depth
                        else ("bgr8" if self.color_mode == ColorMode.BGR else "rgb8"),
                        serial_number=self.serial_number,
                        intrinsics=frame_intrinsics(frame),
                        depth_scale_m=scale,
                        depth_to_color=extrinsics,
                        rotation=self.rotation,
                    )
                with self.metadata_condition:
                    repeated = False
                    for stream, current in metadata.items():
                        previous = self.latest_metadata.get(stream)
                        if previous is not None and current.frame_number == previous.frame_number:
                            repeated = True
                            continue
                        validate_frame(None if self._warming_up else previous, current)
                    # A frameset may reuse one stream. Do not publish it as a new RGBD pair;
                    # readers retain their existing bounded wait for a complete fresh pair.
                    if repeated:
                        continue
                    self.latest_frames = images
                    self.latest_metadata = metadata
                    self.latest_color_frame = images.get("color")
                    # Preserve official warmup/read_depth interfaces too.
                    self.latest_depth_frame = images.get("depth")
                    self.latest_timestamp = time.perf_counter()
                    self.metadata_condition.notify_all()
                self.new_frame_event.set()
            except Exception as exc:
                with self.metadata_condition:
                    self.capture_error = exc
                    self.metadata_condition.notify_all()
                self.new_frame_event.set()
                return

    def read_with_metadata(self, timeout_s):
        return read_new_frame(self, timeout_s)

    def _stop_read_thread(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=11.0)
            if self.thread.is_alive():
                raise RuntimeError("camera capture thread did not terminate")
        self.thread = None
        with self.metadata_condition:
            self.latest_color_frame = self.latest_depth_frame = self.latest_timestamp = None
            self.latest_metadata, self.latest_frames = {}, {}
            self.consumed_frame_number = self.capture_error = None
            self.metadata_condition.notify_all()
