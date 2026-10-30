import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import hashlib
import pytest

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
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: NS(returncode=0, stdout="measured\n"))
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


def test_missing_ssd_stops_before_hardware(tmp_path, monkeypatch):
    import pytest

    p = profile(tmp_path)
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: NS(returncode=1, stdout=""))
    monkeypatch.setattr(op, "logged", lambda *a: pytest.fail("must not connect"))
    with pytest.raises(ValueError, match="SSD未挂载"):
        op.execute(NS(command="record"), p)
    assert not Path(p["runs"]).exists()


def test_readonly_ssd_stops_before_hardware(tmp_path, monkeypatch):
    import pytest

    p = profile(tmp_path)
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: NS(returncode=0, stdout="measured"))

    def denied(**kwargs):
        raise OSError("Read-only file system")

    monkeypatch.setattr(op.tempfile, "TemporaryFile", denied)
    monkeypatch.setattr(op, "logged", lambda *a: pytest.fail("must not connect"))
    with pytest.raises(ValueError, match="SSD不可写"):
        op.execute(NS(command="record"), p)


def test_b_uses_separate_config_and_data_directory(tmp_path, monkeypatch):
    p = profile(tmp_path)
    b = tmp_path / "record-b.json"
    b.write_text(json.dumps({"robot": {}, "dataset": {}, "recording_tasks": {"1": "green left"}}))
    p["record_b_config"] = str(b)
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: NS(returncode=0, stdout="measured"))
    monkeypatch.setattr(provenance, "capture", lambda *a: None)
    calls = []
    monkeypatch.setattr(op, "logged", lambda c, r: calls.append(r) or 0)
    op.execute(NS(command="record", mode="b", layout="2", split="train"), p)
    cfg = json.loads((calls[0] / "config.json").read_text())
    assert cfg["recording_position"] == "B-layout2"
    assert cfg["recording_split"] == "train"
    assert cfg["dataset"]["num_episodes"] == 5
    assert cfg["recording_task_goal"] == 5
    assert cfg["recording_tasks"] == {"1": "green left"}
    assert "/task-b/" in cfg["raw_root"]
    assert "/task-b/" in cfg["dataset"]["root"]
    assert json.loads(Path(p["record_config"]).read_text())["dataset"] == {}


def test_b_prompts_before_hardware_and_development_budget(monkeypatch):
    answers = iter(["2", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    cfg = {"robot": {}, "dataset": {}, "recording_tasks": {str(i): str(i) for i in range(1, 5)}}
    result = op.configure_b_session(cfg, NS())
    assert result["recording_split"] == "development"
    assert result["recording_position"] == "B-layout1"
    assert result["dataset"]["num_episodes"] == 4
    assert result["recording_task_goal"] == 1


def test_b_invalid_layout_does_not_guess():
    import pytest

    with pytest.raises(ValueError, match="未知布局"):
        op.configure_b_session({}, NS(layout="3", split="train"))


def test_act_routes_to_explicit_frozen_bundle_without_direct_hardware(tmp_path, monkeypatch):
    p = profile(tmp_path)
    p["act_bundle"] = str(tmp_path / "frozen-bundle")
    calls = []
    monkeypatch.setattr(op.subprocess, "call", lambda command: calls.append(command) or 0)
    op.execute(NS(command="act", skill="blue-left", layout="2", verify_only=True), p)
    assert calls == [
        [p["python"], str(tmp_path / "frozen-bundle/run.py"), "blue-left", "2", "--verify-only"]
    ]


def test_smolvla_routes_frozen_recipe_and_verify_only_does_not_load_models(tmp_path, monkeypatch):
    p = profile(tmp_path)
    bundle = tmp_path / "smolvla"
    bundle.mkdir()
    entry = bundle / "run-policy.py"
    entry.write_text("# frozen entry\n")
    (bundle / "freeze.json").write_text(
        json.dumps(
            {
                "models": {"original": {}, "flat": {}},
                "tasks": {"blue-left": "blue left"},
                "files": [
                    {
                        "path": entry.name,
                        "bytes": entry.stat().st_size,
                        "sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    p["smolvla_bundle"] = str(bundle)
    calls = []
    monkeypatch.setattr(op.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    args = NS(command="smolvla", model="flat", skill="blue-left", layout="2", verify_only=True)
    assert op.execute(args, p) == 0 and calls == []
    args.verify_only = False
    op.execute(args, p)
    assert calls == [[p["python"], str(entry), "run", "flat", "blue-left", "2"]]
    entry.write_text("# changed\n")
    with pytest.raises(ValueError, match="Frozen SmolVLA asset changed"):
        op.execute(args, p)
    assert len(calls) == 1


def test_c_capture_has_separate_paths_and_retains_skill_context(tmp_path, monkeypatch):
    p = profile(tmp_path)
    cfg = {
        "robot": {"scene": {"scene_id": "synthetic-c"}},
        "teleop": {},
        "dataset": {"fps": 50, "num_episodes": 1},
        "task_c": {
            "drawer_id": "drawer01",
            "position_id": "C-p1",
            "objects": {"cup01": "蓝色杯子"},
            "duration_s": {"store_cup": 2},
        },
    }
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    p["record_c_config"] = str(path)
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **kw: NS(returncode=0, stdout="measured"))
    monkeypatch.setattr(provenance, "capture", lambda *a: None)
    calls = []
    monkeypatch.setattr(op, "logged", lambda cmd, run: calls.append((cmd, run)) or 0)
    op.execute(
        NS(
            command="record",
            mode="c",
            skill="store_cup",
            object_id="cup01",
            chain_id=None,
            split="development",
            layout=None,
        ),
        p,
    )
    command, run = calls[0]
    saved = json.loads((run / "config.json").read_text())
    assert command[:4] == ["sudo", "piper-socketcan", "exec", "--"]
    assert "/task-c/" in saved["raw_root"] and "/task-c/" in saved["dataset"]["root"]
    assert saved["recording_context"]["skill_id"] == "store_cup"
    assert saved["recording_context"]["object_id"] == "cup01"
    assert "蓝色杯子" in saved["dataset"]["single_task"]


def test_c_check_shortcut_never_uses_sudo(tmp_path, monkeypatch):
    p = profile(tmp_path)
    calls = []
    monkeypatch.setattr(op.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    op.execute(
        NS(
            command="c",
            mode="check",
            config=tmp_path / "c.json",
            skill="open_drawer",
            object_id=None,
            chain_id=None,
        ),
        p,
    )
    assert calls[0][:4] == [p["python"], "-m", "piper_outcome_stack.task_c", "check"]
    assert "sudo" not in calls[0]
