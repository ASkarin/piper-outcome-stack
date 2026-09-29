import json
from threading import Event, Thread

import numpy as np
from PIL import Image

from piper_outcome_stack.policy_rgb_recording import PolicyRGBRecorder


def test_lossless_snapshot_and_timestamp_pairing(tmp_path):
    writer = PolicyRGBRecorder(tmp_path / "rgb")
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    expected = image.copy()
    meta = dict(
        row_index=17,
        observation_sequence=123,
        observed_monotonic_s=4.2,
        camera_frame_number=88,
        camera_timestamp_ms=456.0,
    )
    assert writer.submit(image, meta, 4.2)["status"] == "queued"
    image[:] = 0
    assert writer.submit(image, meta, 4.25) is None
    result = writer.close()
    assert result["written"] == 1 and result["status"] == "completed"
    entry = json.loads((tmp_path / "rgb/frames.jsonl").read_text())
    assert all(entry[k] == v for k, v in meta.items())
    np.testing.assert_array_equal(
        np.asarray(Image.open(tmp_path / "rgb" / entry["file"])), expected
    )


def test_slow_writer_drops_instead_of_blocking_producer(tmp_path, monkeypatch):
    entered, release, done = Event(), Event(), Event()
    original = PolicyRGBRecorder._write_frame

    def slow(self, *args):
        entered.set()
        release.wait(3)
        original(self, *args)

    monkeypatch.setattr(PolicyRGBRecorder, "_write_frame", slow)
    writer = PolicyRGBRecorder(tmp_path / "rgb")
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    writer.submit(image, {}, 0)
    assert entered.wait(1)

    def produce():
        for i in range(20):
            writer.submit(image, {}, i + 1)
        done.set()

    producer = Thread(target=produce, daemon=True)
    producer.start()
    try:
        assert done.wait(1), "recording producer blocked on a slow writer"
        assert writer._queue.qsize() == 8
        assert writer.dropped == 12
    finally:
        release.set()
        producer.join(2)
        result = writer.close()
    assert result["written"] == 9 and result["unwritten"] == 0


def test_disk_failure_is_visible_and_does_not_block_submit(tmp_path, monkeypatch):
    def broken(*args):
        raise OSError("disk full")

    monkeypatch.setattr(PolicyRGBRecorder, "_write_frame", broken)
    writer = PolicyRGBRecorder(tmp_path / "rgb")
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    writer.submit(image, {}, 0)
    writer._thread.join(1)
    assert writer.submit(image, {}, 1)["status"] == "unavailable"
    result = writer.close()
    assert result["status"] == "failed" and result["unwritten"] == 1
    assert "disk full" in result["error"]
