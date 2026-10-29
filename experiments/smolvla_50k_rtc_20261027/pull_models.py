"""piper-local pulls the two completed checkpoints in parallel; no device access."""

from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import time

BASE = Path(__file__).resolve().parent


def save(value):
    path = BASE / "transfer-status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            dict(updated_at=datetime.now(timezone.utc).isoformat(), hardware_io=False, **value),
            indent=2,
        )
        + "\n"
    )
    temp.replace(path)


def main():
    spec = json.loads((BASE / "spec.json").read_text())
    if (BASE / "transfer-status.json").exists():
        raise FileExistsError("this transfer already has state")
    active = []
    for arm, row in spec["models"].items():
        output = BASE / "checkpoints" / arm
        output.mkdir(parents=True, exist_ok=False)
        command = [
            "rsync",
            "-az",
            "--partial",
            "--info=progress2",
            "piper-training:" + row["source_checkpoint"] + "/",
            str(output) + "/",
        ]
        log = (BASE / (arm + "-transfer.log")).open("xb")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        active.append((arm, process, command))
    while True:
        workers = []
        for arm, process, command in active:
            code = process.poll()
            workers.append(
                dict(
                    model=arm,
                    pid=process.pid,
                    status="running" if code is None else "complete" if code == 0 else "failed",
                    exit_code=code,
                )
            )
            if code not in (None, 0):
                raise RuntimeError(
                    f"{arm} checkpoint transfer failed: exit{code}; see {arm}-transfer.log"
                )
        save(dict(status="running", workers=workers))
        if all(p.poll() == 0 for _, p, _ in active):
            break
        time.sleep(5)
    for arm, row in spec["models"].items():
        for expected in row["files"]:
            path = BASE / "checkpoints" / arm / expected["name"]
            if not path.is_file() or path.stat().st_size != expected["bytes"]:
                raise RuntimeError(f"checkpoint file incomplete: {path}")
    save(dict(status="complete", workers=workers, models=list(spec["models"])))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        old = (
            json.loads((BASE / "transfer-status.json").read_text())
            if (BASE / "transfer-status.json").exists()
            else {}
        )
        save(dict(status="failed", workers=old.get("workers", []), error=repr(error)))
        raise
