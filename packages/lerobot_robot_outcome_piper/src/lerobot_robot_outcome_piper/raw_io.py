"""Shared raw-file completion primitives, independent of capture/control source."""

import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_queue(pending, limit):
    while pending and pending[0].done():
        pending[0].result()
        pending.popleft()
    if len(pending) >= limit:
        raise RuntimeError("raw writer queue full; recording ended without dropping frames")


def drain_queue(pending):
    # Consume every submitted write even if an earlier one failed.
    error = None
    failed = []
    while pending:
        future = pending.popleft()
        try:
            future.result()
        except BaseException as exc:
            failed.append(future)
            if error is None:
                error = exc
    pending.extend(failed)  # A failed write stays failed on subsequent save/close calls.
    if error is not None:
        raise error
