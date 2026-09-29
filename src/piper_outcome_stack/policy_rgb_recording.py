"""Best-effort diagnostic RGB snapshots; no encoding or disk I/O in submit()."""

import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import time

from PIL import Image


class PolicyRGBRecorder:
    """Single control-thread producer, bounded queue, one PNG writer thread."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self._index = (self.directory / "frames.jsonl").open("x")
        self._queue = Queue(maxsize=8)
        self._stop = Event()
        self._next_due = float("-inf")
        self.sampled = self.submitted = self.written = self.dropped = 0
        self.error = None
        self._thread = Thread(target=self._worker, name="policy-rgb-writer", daemon=True)
        self._thread.start()

    def submit(self, image, metadata, now):
        """Copy an existing policy input at <=10Hz, never wait for the writer."""
        if now < self._next_due:
            return None
        started = time.perf_counter()
        self._next_due = now + 0.1
        frame_id = self.sampled
        self.sampled += 1
        result = {"frame_id": frame_id}
        if self.error is not None or self._stop.is_set():
            result.update(status="unavailable", error=self.error)
        elif self._queue.full():
            self.dropped += 1
            result["status"] = "queue_full"
        else:
            # Camera buffers may be reused: retain our own bytes, not a mutable view.
            item = (image.copy(), {**metadata, "frame_id": frame_id})
            try:
                self._queue.put_nowait(item)
            except Full:
                self.dropped += 1
                result["status"] = "queue_full"
            else:
                self.submitted += 1
                result["status"] = "queued"
        result["submit_s"] = time.perf_counter() - started
        return result

    def _write_frame(self, image, metadata):
        name = f"frame-{metadata['frame_id']:06d}.png"
        Image.fromarray(image).save(self.directory / name, compress_level=1)
        self._index.write(json.dumps({**metadata, "file": name}) + "\n")
        self._index.flush()

    def _worker(self):
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                self._write_frame(*item)
                self.written += 1
        except Exception as exc:
            # Diagnostic failure must not block the robot's input/hold handling.
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._index.close()

    def close(self):
        """Called only after the robot stop/disconnect path; bounded drain wait."""
        self._stop.set()
        self._thread.join(timeout=5)
        alive = self._thread.is_alive()
        return {
            "directory": str(self.directory),
            "format": "lossless PNG RGB",
            "max_fps": 10,
            "queue_capacity": 8,
            "sampled": self.sampled,
            "submitted": self.submitted,
            "written": self.written,
            "queue_full_drops": self.dropped,
            "unwritten": self.submitted - self.written,
            "writer_alive": alive,
            "error": self.error,
            "status": "failed" if self.error else "incomplete" if alive else "completed",
        }
