import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "infra/local-controller/operator"))
import provenance

spec = importlib.util.spec_from_file_location(
    "operator_entry", Path(__file__).parents[1] / "infra/local-controller/operator/operator.py"
)
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


def profile(tmp_path):
    p = tmp_path / "record.json"
    p.write_text(json.dumps({"robot": {}, "teleop": {}, "dataset": {}}))
    return dict(
        python="/runtime/bin/python",
        source="/source",
        runs=str(tmp_path / "runs"),
        record_config=str(p),
        ssd_mount="/mnt/ssd",
        ssd_uuid="measured",
        raw_parent=str(tmp_path / "raw"),
        dataset_parent=str(tmp_path / "datasets"),
    )


def test_record_uses_existing_cli_unique_paths_and_logs(tmp_path, monkeypatch):
    p = profile(tmp_path)
    calls = []
    monkeypatch.setattr(
        provenance, "capture", lambda profile, run: (run / "environment.json").write_text("{}")
    )
    monkeypatch.setattr(op.subprocess, "check_output", lambda *a, **k: "measured\n")
    monkeypatch.setattr(op, "logged", lambda c, r: calls.append((c, r)) or 0)
    for _ in range(2):
        assert op.execute(NS(command="record"), p) == 0
    assert calls[0][1] != calls[1][1]
    for command, run in calls:
        assert command[:4] == ["sudo", "piper-socketcan", "exec", "--"]
        c = json.loads((run / "config.json").read_text())
        assert c["raw_root"] != c["dataset"]["root"]
        assert not c["resume"]


def test_status_fixed_query_and_convert_no_sudo(tmp_path, monkeypatch):
    p = profile(tmp_path)
    calls = []
    monkeypatch.setattr(op.subprocess, "call", lambda c: calls.append(c) or 0)
    op.execute(NS(command="status"), p)
    assert calls[-1] == ["sudo", "-n", "/usr/local/sbin/piper-query", "status"]
    session = tmp_path / "session"
    session.mkdir()
    (session / "config.json").write_text(
        json.dumps({"raw_root": "/raw a", "dataset": {"root": "/data a", "repo_id": "local/id"}})
    )
    op.execute(NS(command="convert", session=session), p)
    assert "sudo" not in calls[-1] and "/raw a" in calls[-1]


def test_recover_does_not_use_record_or_enable(tmp_path, monkeypatch):
    p = profile(tmp_path)
    calls = []
    monkeypatch.setattr(op, "logged", lambda c, r: calls.append(c) or 0)
    op.execute(NS(command="recover"), p)
    assert calls[0][5].endswith("/operator/recover.py")
    assert "enable" not in calls[0] and "record" not in calls[0]
