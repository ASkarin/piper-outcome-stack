"""Raw Xbox frame storage and offline official Dataset conversion."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import argparse
import copy
import threading
import json
import os
import shutil
import time
import uuid
import numpy as np


from .raw_io import write_json


class RawFrames:
    is_raw = True
    image_writer = None

    def __init__(
        self,
        root,
        *,
        fps,
        features,
        robot_type,
        resume=False,
        conversion_options=None,
        capture_context=None,
    ):
        self.root = Path(root)
        self.fps, self.features = fps, features
        self.meta = SimpleNamespace(robot_type=robot_type)
        self.root.mkdir(parents=True, exist_ok=resume)
        spec = dict(
            schema="piper-xbox-raw-v1",
            fps=fps,
            features=features,
            robot_type=robot_type,
            conversion_options=conversion_options or {},
            capture_context=capture_context or {},
        )
        if resume:
            if (self.root / "teach-conversion.json").exists():
                raise ValueError("Xbox cannot resume a converted teach Dataset")
            if json.loads((self.root / "raw.json").read_text()) != json.loads(json.dumps(spec)):
                raise ValueError("raw resume schema or scene differs")
            if json.loads((self.root / "capture.json").read_text())["status"] != "closed":
                raise ValueError("resume requires a normally closed capture")
        else:
            write_json(self.root / "raw.json", spec)
        self.num_episodes = sum(
            json.loads(p.read_text())["status"] == "saved"
            for p in self.root.glob("attempts/*/result.json")
        )
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xbox-raw-writer")
        self.pending = deque()
        self._queue_lock = threading.RLock()
        self._writer_error = None
        self._queue_error = None
        self._closed = False
        self.preparation = None
        # Start the one worker before any hardware connection.
        self.pool.submit(lambda: None).result()
        self.path = self.stream = None
        self.frames = 0
        write_json(self.root / "capture.json", {"status": "running"})

    def check_writer(self):
        with self._queue_lock:
            if self._writer_error is not None:
                raise self._writer_error
            if self._queue_error is not None:
                raise self._queue_error
            while self.pending and self.pending[0][0].done():
                self.pending.popleft()[0].result()

    def submit_io(self, fn, *args, frame=False):
        """Never wait for disk capacity on the caller/control thread."""
        with self._queue_lock:
            self.check_writer()
            if self._closed:
                raise RuntimeError("raw writer is closed")
            if len(self.pending) >= 32 or (
                frame and sum(is_frame for _, is_frame in self.pending) >= 8
            ):
                error = RuntimeError(
                    "raw writer queue full; no frames dropped; episode cannot be saved"
                )
                self._queue_error = error
                raise error

            def run():
                try:
                    with self._queue_lock:
                        error = self._writer_error
                    if error is not None:
                        raise error
                    return fn(*args)
                except BaseException as exc:
                    with self._queue_lock:
                        if self._writer_error is None:
                            self._writer_error = exc
                    raise

            future = self.pool.submit(run)
            self.pending.append((future, frame))
            return future

    def flush_io(self):
        with self._queue_lock:
            pending = list(self.pending)
        error = None
        for future, _ in pending:
            try:
                future.result()
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error
        self.check_writer()

    def prepare_episode(self, depth_directory=None, completed=None):
        if self.preparation is not None:
            if self.frames:
                raise RuntimeError("cannot prepare an unsealed recording")
            return self.preparation

        def prepare():
            start, cpu = time.monotonic(), time.thread_time()
            self.path = self.root / "attempts" / uuid.uuid4().hex
            self.path.mkdir(parents=True)
            self.stream = (self.path / "frames.jsonl").open("x")
            if depth_directory is not None:
                Path(depth_directory).mkdir(parents=True, exist_ok=False)
            write_json(
                self.path / "result.json",
                dict(status="recording", frames=0, episode_index=self.num_episodes),
            )
            if completed is not None:
                completed(
                    dict(wall_s=time.monotonic() - start, thread_cpu_s=time.thread_time() - cpu)
                )

        self.preparation = self.submit_io(prepare)
        return self.preparation

    def add_frame(self, frame, *, depth_files=None, write_event=None, row=None):
        self.check_writer()
        if self.preparation is None or not self.preparation.done():
            raise RuntimeError("raw attempt must finish preparation before accepting frames")
        self.preparation.result()
        if self.path is None:
            raise RuntimeError("raw attempt is not prepared")
        values, files, pixels = {}, {}, {}
        for key, value in frame.items():
            if self.features.get(key, {}).get("dtype") in ("video", "image"):
                name = f"{self.frames:06d}-{len(files)}.npy"
                files[key] = name
                pixels[name] = np.array(value, copy=True)
            else:
                values[key] = (
                    value.tolist() if isinstance(value, np.ndarray) else copy.deepcopy(value)
                )
        depth = {name: np.array(value, copy=True) for name, value in (depth_files or {}).items()}
        path, index = self.path, self.frames
        enqueued = time.monotonic()

        def store():
            started, cpu = time.monotonic(), time.thread_time()
            for name, value in depth.items():
                with (self.root / name).open("xb") as stream:
                    np.savez(stream, depth=value)  # Uncompressed, same existing format.
            for name, value in pixels.items():
                with (path / name).open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
            self.stream.write(
                json.dumps(dict(frame_index=index, values=values, files=files), allow_nan=False)
                + "\n"
            )
            if write_event is not None:
                write_event("frame_pending", **row)
                write_event(
                    "frame_written",
                    episode_index=row["episode_index"],
                    frame_index=index,
                    attempt=row["attempt"],
                    queue_wait_s=started - enqueued,
                    write_wall_s=time.monotonic() - started,
                    write_thread_cpu_s=time.thread_time() - cpu,
                    completed_monotonic_s=time.monotonic(),
                    semantics="frame payload written; existing NPY fsync completed; episode not sealed",
                )

        self.submit_io(store, frame=True)
        self.frames += 1

    def _close_attempt(self, status):
        self.check_writer()
        if self.path is None:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        # An empty prepared attempt is never a saved demonstration.
        if status == "saved" and not self.frames:
            status = "discarded"
        write_json(
            self.path / "result.json",
            dict(status=status, frames=self.frames, episode_index=self.num_episodes),
        )
        if status == "saved":
            self.num_episodes += 1
        self.path = self.stream = None
        self.preparation = None
        self.frames = 0

    def save_episode(self):
        self.flush_io()
        self.submit_io(self._close_attempt, "saved").result()

    def clear_episode_buffer(self):
        self.flush_io()
        self.submit_io(self._close_attempt, "discarded").result()

    def finalize(self):
        # Drain/close the last attempt, but keep the session writer for final telemetry.
        try:
            self.flush_io()
            self.submit_io(self._close_attempt, "interrupted").result()
        except BaseException as exc:
            # Called from the save worker or after hardware disconnect, never a control tick.
            self.pool.shutdown(wait=True)
            self._closed = True
            if self.stream is not None and not self.stream.closed:
                self.stream.close()
            if self.path is not None:
                write_json(
                    self.path / "result.json",
                    dict(status="failed", frames=self.frames, error=str(exc)),
                )
            raise

    def close(self):
        try:
            self.flush_io()
        finally:
            self.pool.shutdown(wait=True)
            self._closed = True

    def finish_capture(self, success):
        try:
            if success:
                self.flush_io()
            write_json(self.root / "capture.json", {"status": "closed" if success else "failed"})
        finally:
            self.pool.shutdown(wait=True)
            self._closed = True


def convert(raw_root, output, repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig, DepthEncoderConfig
    from .action_audit import verify_telemetry
    from .episode_save import compress_depth_file

    raw_root, output = Path(raw_root), Path(output)
    if json.loads((raw_root / "capture.json").read_text())["status"] != "closed":
        raise ValueError("conversion requires a normally closed raw capture")
    spec = json.loads((raw_root / "raw.json").read_text())
    if spec["schema"] != "piper-xbox-raw-v1":
        raise ValueError("not Xbox raw data")
    if output.exists():
        raise FileExistsError(output)
    attempts = []
    for path in raw_root.glob("attempts/*"):
        result = json.loads((path / "result.json").read_text())
        if result["status"] == "saved":
            attempts.append((result["episode_index"], path, result))
    attempts.sort()
    if not attempts or [i for i, _, _ in attempts] != list(range(len(attempts))):
        raise ValueError("saved episode indices invalid")
    options = dict(spec["conversion_options"])
    for key, cls in (("rgb_encoder", RGBEncoderConfig), ("depth_encoder", DepthEncoderConfig)):
        if options.get(key) is not None:
            options[key] = cls(**options[key])
    from .xbox_recovery import audit_raw

    audit_raw(raw_root)
    started = time.monotonic()
    ds = LeRobotDataset.create(
        repo_id,
        spec["fps"],
        root=output,
        robot_type=spec["robot_type"],
        features=spec["features"],
        image_writer_threads=2,
        **options,
    )
    try:
        for index, path, result in attempts:
            count = 0
            for line in (path / "frames.jsonl").open():
                row = json.loads(line)
                if row["frame_index"] != count:
                    raise ValueError("noncontiguous raw frames")
                frame = {
                    k: np.asarray(v, dtype=np.float32) if isinstance(v, list) else v
                    for k, v in row["values"].items()
                }
                frame.update(
                    {
                        k: np.load(path / name, allow_pickle=False)
                        for k, name in row["files"].items()
                    }
                )
                ds.add_frame(frame)
                count += 1
            if count != result["frames"]:
                raise ValueError("raw frame count differs")
            ds.save_episode()
    finally:
        ds.finalize()
    encoding_s = time.monotonic() - started
    shutil.copytree(raw_root / "telemetry", output / "telemetry")
    compression_started = time.monotonic()
    for path in (output / "telemetry").glob("*/raw_depth/*/*.npz"):
        compress_depth_file(path, "npz")
    compression_s = time.monotonic() - compression_started
    loaded = LeRobotDataset(repo_id, root=output)
    audit = verify_telemetry(output, loaded)
    report = dict(
        status="complete",
        source=str(raw_root.resolve()),
        action_rule="actual_dispatched_target_or_retained_hold",
        **audit,
        episodes_source=[
            dict(episode_index=i, attempt=str(p.relative_to(raw_root)), frames=r["frames"])
            for i, p, r in attempts
        ],
        encoding_and_write_s=encoding_s,
        compression_s=compression_s,
        total_s=time.monotonic() - started,
    )
    from .pause_review import candidates

    write_json(output / "pause-candidates.json", candidates(raw_root))
    write_json(output / "xbox-conversion.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.raw_root, args.output, args.repo_id), indent=2))
    return 0
