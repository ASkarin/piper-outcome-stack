"""Resume this shadow transfer only after a verified transport failure; never retry preflight."""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def write_status(path, value):
    temporary = path.with_name(path.name + ".resume-tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def network_failure(status, log):
    if status.get("status") != "failed":
        return False
    stages = status.get("stages", [])
    if not stages or stages[-1].get("exit_code") not in (10, 12, 30, 35, 255):
        return False
    message = log.lower()
    non_network = (
        "permission denied",
        "no space left",
        "read-only file system",
        "input/output error",
        "no such file or directory",
        "checksum mismatch",
        "out of memory",
        "segmentation fault",
        "host key verification failed",
        "remote host identification has changed",
    )
    if any(value in message for value in non_network):
        return False
    transport = (
        "connection timed out",
        "connection reset",
        "broken pipe",
        "network is unreachable",
        "no route to host",
        "io timeout",
        "i/o timeout",
        "timed out during banner exchange",
    )
    return any(value in message for value in transport) or (
        "timeout, server " in message and "not responding" in message
    )


def active_local(root):
    rows = subprocess.check_output(["ps", "-eo", "pid,stat,args"], text=True).splitlines()[1:]
    found = []
    for row in rows:
        parts = row.strip().split(None, 2)
        if len(parts) != 3 or parts[1].startswith("Z") or int(parts[0]) == os.getpid():
            continue
        if str(root) in parts[2] and any(
            term in parts[2]
            for term in (
                "transfer.py",
                "prepare-after-transfer.py",
                "smolvla_shadow",
                "rsync",
            )
        ):
            found.append({"pid": int(parts[0]), "command": parts[2]})
    return found


def remote_probe(sources):
    script = """from pathlib import Path
import json,subprocess
sources=SOURCES
found=[]
for row in subprocess.check_output(['ps','-eo','pid,stat,args'],text=True).splitlines()[1:]:
 parts=row.strip().split(None,2)
 if len(parts)==3 and not parts[1].startswith('Z') and 'rsync' in parts[2] and any(p in parts[2] for p in sources):
  found.append({'pid':int(parts[0]),'command':parts[2]})
print(json.dumps({'sources_exist':all(Path(p).exists() for p in sources),'processes':found}))
""".replace("SOURCES", repr(tuple(sources)))
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=12",
                "-o",
                "ServerAliveInterval=5",
                "-o",
                "ServerAliveCountMax=2",
                "piper-training",
                "python3 -",
            ],
            input=script,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"reachable": False, "reason": "direct SSH probe timed out"}
    if result.returncode:
        fatal = any(
            marker in result.stderr.lower()
            for marker in (
                "permission denied",
                "authentication failed",
                "too many authentication failures",
                "host key verification failed",
                "remote host identification has changed",
                "invalid format",
                "command not found",
                "traceback",
                "syntaxerror",
            )
        )
        return {"reachable": False, "reason": result.stderr[-1000:], "needs_attention": fatal}
    return {"reachable": True, **json.loads(result.stdout)}


def resume(root, sources):
    with (root / "resume.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy", "reason": "resume helper already running"}
        status = json.loads((root / "transfer-status.json").read_text())
        if status.get("status") != "failed":
            return {"status": "not_failed", "transfer_status": status.get("status")}
        running = active_local(root)
        if running:
            return {
                "status": "busy",
                "reason": "transfer/preparation still active",
                "processes": running,
            }
        if (root / "preflight").exists() or (root / "preflight.log").exists():
            return {
                "status": "needs_attention",
                "reason": "preflight already started; do not retry it",
            }
        preparation = json.loads((root / "preparation-status.json").read_text())
        if preparation.get("status") == "failed":
            if "direct transfer failed:" not in preparation.get("error", ""):
                return {
                    "status": "needs_attention",
                    "reason": "preparation failed for a non-transfer reason",
                }
        elif preparation.get("status") != "waiting_for_transfer":
            return {"status": "needs_attention", "reason": "unexpected preparation stage"}
        with (root / "transfer.log").open("rb") as log:
            log.seek(max(0, (root / "transfer.log").stat().st_size - 65536))
            tail = log.read().decode(errors="replace")
        if not network_failure(status, tail):
            return {
                "status": "needs_attention",
                "reason": "latest failure is not an established network interruption",
            }
        remote = remote_probe(sources)
        if not remote["reachable"]:
            return {
                "status": "needs_attention"
                if remote.get("needs_attention")
                else "waiting_for_connection",
                "reason": remote["reason"],
            }
        if not remote["sources_exist"]:
            return {"status": "needs_attention", "reason": "transfer source missing"}
        if remote["processes"]:
            return {
                "status": "busy",
                "reason": "remote sender still active",
                "processes": remote["processes"],
            }
        for script in ("transfer.py", "prepare-after-transfer.py"):
            if not (root / script).is_file():
                return {"status": "needs_attention", "reason": f"missing task script: {script}"}
        history = root / "resume-attempts"
        history.mkdir(exist_ok=True)
        attempt = Path(
            tempfile.mkdtemp(
                prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"),
                dir=history,
            )
        )
        for name in (
            "transfer-status.json",
            "transfer-launch.json",
            "transfer.log",
            "preparation-status.json",
            "preparation-launch.json",
            "preparation.log",
        ):
            path = root / name
            if path.exists():
                shutil.copy2(path, attempt / name)
        # Prevent the replacement waiter from observing the previous failed status.
        write_status(
            root / "transfer-status.json",
            {
                "status": "starting",
                "stages": [],
                "resume_attempt": str(attempt),
                "started": time.time(),
            },
        )
        write_status(
            root / "preparation-status.json",
            {
                "status": "starting",
                "hardware_io": False,
                "resume_attempt": str(attempt),
            },
        )
        launched = {}
        for script, log_name, launch_name in (
            ("transfer.py", "transfer.log", "transfer-launch.json"),
            ("prepare-after-transfer.py", "preparation.log", "preparation-launch.json"),
        ):
            with (root / log_name).open("w") as log:
                process = subprocess.Popen(
                    ["python3", str(root / script)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            record = {
                "pid": process.pid,
                "script": str(root / script),
                "resume_attempt": str(attempt),
            }
            write_status(root / launch_name, record)
            launched[script] = record
        result = {
            "status": "resumed",
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "attempt": str(attempt),
            "launched": launched,
            "partial_files_preserved": True,
            "preflight_retried": False,
        }
        write_status(attempt / "resume.json", result)
        write_status(root / "latest-resume.json", result)
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(resume(args.root, args.source), ensure_ascii=False))
