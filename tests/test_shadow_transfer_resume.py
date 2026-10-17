import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[1] / "infra/transfers/shadow_resume.py"
    spec = importlib.util.spec_from_file_location("shadow_resume", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failed_state(root):
    state = {"status": "failed", "stages": [{"name": "checkpoint", "exit_code": 255}]}
    (root / "transfer-status.json").write_text(json.dumps(state))
    (root / "transfer.log").write_text("Timeout, server example not responding.\nBroken pipe\n")
    (root / "preparation-status.json").write_text(
        json.dumps({"status": "failed", "error": "direct transfer failed: rsync255"})
    )
    for name in ["transfer.py", "prepare-after-transfer.py"]:
        (root / name).write_text("# test-only entry\n")
    return state


def test_only_transport_errors_are_eligible(helper):
    state = {"status": "failed", "stages": [{"exit_code": 255}]}
    assert helper.network_failure(state, "Timeout, server example not responding.")
    assert not helper.network_failure(state, "connection unexpectedly closed")
    assert not helper.network_failure(state, "Permission denied\nBroken pipe")
    assert not helper.network_failure(state, "No space left on device\nBroken pipe")
    assert not helper.network_failure({"status": "complete"}, "connection reset")


@pytest.mark.parametrize(
    "condition,expected",
    [
        ("live", "busy"),
        ("unreachable", "waiting_for_connection"),
        ("sender", "busy"),
        ("preflight", "needs_attention"),
    ],
)
def test_refusal_never_starts_or_overwrites(helper, tmp_path, monkeypatch, condition, expected):
    failed_state(tmp_path)
    before = (tmp_path / "transfer-status.json").read_bytes()
    monkeypatch.setattr(
        helper, "active_local", lambda root: [{"pid": 77}] if condition == "live" else []
    )
    monkeypatch.setattr(
        helper,
        "remote_probe",
        lambda sources: (
            {"reachable": False, "reason": "timeout"}
            if condition == "unreachable"
            else {
                "reachable": True,
                "sources_exist": True,
                "processes": [{"pid": 88}] if condition == "sender" else [],
            }
        ),
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("must not launch")

    monkeypatch.setattr(helper.subprocess, "Popen", unexpected)
    if condition == "preflight":
        (tmp_path / "preflight.log").write_text("failed inference")
    assert helper.resume(tmp_path, ["/recorded/source"])["status"] == expected
    assert (tmp_path / "transfer-status.json").read_bytes() == before


def test_resume_preserves_failure_and_partial_and_initializes_before_launch(
    helper, tmp_path, monkeypatch
):
    failed_state(tmp_path)
    (tmp_path / "checkpoint").mkdir()
    partial = tmp_path / "checkpoint/model.safetensors"
    partial.write_bytes(b"partial-model")
    monkeypatch.setattr(helper, "active_local", lambda root: [])
    monkeypatch.setattr(
        helper,
        "remote_probe",
        lambda sources: {"reachable": True, "sources_exist": True, "processes": []},
    )
    calls = []

    def start(cmd, **kwargs):
        assert json.loads((tmp_path / "transfer-status.json").read_text())["status"] == "starting"
        calls.append(cmd)
        return NS(pid=900 + len(calls))

    monkeypatch.setattr(helper.subprocess, "Popen", start)
    result = helper.resume(tmp_path, ["/recorded/source"])
    assert result["status"] == "resumed" and len(calls) == 2
    assert partial.read_bytes() == b"partial-model"
    archive = Path(result["attempt"])
    assert json.loads((archive / "transfer-status.json").read_text())["status"] == "failed"
    assert "Broken pipe" in (archive / "transfer.log").read_text()
    assert helper.resume(tmp_path, ["/recorded/source"])["status"] == "not_failed"
    assert len(calls) == 2


def test_direct_ssh_auth_error_is_not_treated_as_network_wait(helper, tmp_path, monkeypatch):
    failed_state(tmp_path)
    monkeypatch.setattr(helper, "active_local", lambda root: [])
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *a, **kw: NS(returncode=255, stderr="Permission denied (publickey)", stdout=""),
    )
    assert helper.resume(tmp_path, ["/recorded/source"])["status"] == "needs_attention"
    assert not (tmp_path / "resume-attempts").exists()
