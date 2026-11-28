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


@pytest.mark.parametrize(
    ("extra", "start", "split", "goal_count"),
    [
        ([], "03", "train", 50),
        (["--split", "development"], "03", "development", 6),
        (["--start", "01"], "01", "train", 20),
        (["--start", "02"], "02", "train", 20),
        (["--start", "01", "--split", "development"], "01", "development", 4),
        (["--split", "development", "--start", "02"], "02", "development", 4),
    ],
)
def test_pusht_record_shortcut(tmp_path, monkeypatch, extra, start, split, goal_count):
    p = profile(tmp_path)
    reference, goal = tmp_path / "start-03.png", tmp_path / "target-G.png"
    reference.touch()
    selected_reference = tmp_path / f"start-{start}.png"
    selected_reference.touch()
    goal.touch()
    cfg = {
        "robot": {
            "scene": {"scene_id": "synthetic-C"},
            "cameras": {"d435": {"use_rgb": True, "use_depth": False}},
        },
        "teleop": {},
        "dataset": {"fps": 50, "num_episodes": 2},
        "task_c": {
            "start_reference": str(reference),
            "goal_reference": str(goal),
            "position_id": "start-03",
            "duration_s": 2,
        },
        "export_depth": False,
    }
    config_path = tmp_path / "record-c.json"
    config_path.write_text(json.dumps(cfg))
    p["record_c_config"] = str(config_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(p))
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: NS(returncode=0, stdout="measured"))
    monkeypatch.setattr(provenance, "capture", lambda *a: None)
    calls = []
    monkeypatch.setattr(op, "logged", lambda command, run: calls.append((command, run)) or 0)
    op.main(["--profile", str(profile_path), "record", "c", *extra])
    cmd, run = calls[0]
    saved = json.loads((run / "config.json").read_text())
    assert cmd[:4] == ["sudo", "piper-socketcan", "exec", "--"]
    assert (
        f"/task-c/{split}/" in saved["raw_root"] and f"/task-c/{split}/" in saved["dataset"]["root"]
    )
    assert saved["recording_split"] == split and saved["recording_position"] == f"start-{start}"
    assert saved["recording_tasks"] == {str(int(start)): saved["dataset"]["single_task"]}
    assert saved["recording_context"]["start_id"] == saved["recording_position"]
    assert saved["recording_context"]["start_reference"] == str(selected_reference)
    assert saved["recording_context"]["goal_reference"] == str(goal)
    assert saved["recording_target_precision"] == "fine"
    assert saved["dataset"]["num_episodes"] == goal_count
    assert saved["robot"]["cameras"]["d435"] == {"use_rgb": True, "use_depth": False}
    assert not saved["export_depth"]


@pytest.mark.parametrize("mode", ["a", "b"])
def test_record_start_is_specific_to_push_t(mode, tmp_path):
    with pytest.raises(ValueError, match="--start"):
        op.execute(NS(command="record", mode=mode, start="01"), profile(tmp_path))


def test_c_check_uses_active_config_without_privileges(tmp_path, monkeypatch):
    p = profile(tmp_path)
    p["record_c_config"] = str(tmp_path / "c.json")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(p))
    calls = []
    monkeypatch.setattr(op.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    op.main(["--profile", str(path), "c", "check"])
    assert calls[0][:6] == [
        p["python"],
        "-m",
        "piper_outcome_stack.task_c",
        "check",
        "--config",
        p["record_c_config"],
    ]
    assert "sudo" not in calls[0]


def test_c_model_selector_binds_only_the_requested_candidate(tmp_path, monkeypatch):
    p = profile(tmp_path)
    p["record_c_config"] = str(tmp_path / "default.json")
    candidate = tmp_path / "act.json"
    candidate.write_text(json.dumps({"policy": {"variant": "act"}}))
    p["c_policy_configs"] = {"act": str(candidate)}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(p))
    calls = []
    monkeypatch.setattr(op.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert op.main(["--profile", str(path), "c", "check", "act"]) == 0
    assert calls[0][calls[0].index("--config") + 1] == str(candidate)
    assert "sudo" not in calls[0]
    import pytest

    with pytest.raises(SystemExit):
        op.main(["--profile", str(path), "c", "check", "flat"])
    assert len(calls) == 1
