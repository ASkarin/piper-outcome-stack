"""One RGB publication extension; device setup and image processing stay upstream."""

import threading
import time
import numpy as np
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from .camera import CameraTelemetry, read_new_frame, validate_frame


class TimedRealSenseCamera(RealSenseCamera):
    def __init__(self, config):
        super().__init__(config)
        self.metadata_condition = threading.Condition(self.frame_lock)
        self.latest_metadata = None
        self.consumed_frame_number = None
        self.capture_error = None

    def _read_loop(self):
        # Upstream discards device metadata and retries read failures. Override
        # only publication: preserve metadata and fail on the first read error.
        while not self.stop_event.is_set():
            try:
                frames = self._read_from_hardware()
                received = time.monotonic()
                raw = frames.get_color_frame()
                if not raw:
                    raise RuntimeError("D435 color frame is missing")
                rgb = self._postprocess_image(np.asanyarray(raw.get_data())).copy()
                metadata = CameraTelemetry(
                    int(raw.get_frame_number()),
                    float(raw.get_timestamp()),
                    str(raw.get_frame_timestamp_domain()),
                    received,
                    time.monotonic(),
                )
                with self.metadata_condition:
                    validate_frame(self.latest_metadata, metadata)
                    self.latest_color_frame = rgb
                    self.latest_metadata = metadata
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
            # Official hardware wait is bounded at ten seconds.
            self.thread.join(timeout=11.0)
            if self.thread.is_alive():
                raise RuntimeError("D435 capture thread did not terminate")
        self.thread = None
        with self.metadata_condition:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.latest_metadata = None
            self.consumed_frame_number = None
            self.capture_error = None
            self.metadata_condition.notify_all()
