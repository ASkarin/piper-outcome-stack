"""Fixed diagnostic wire requests and privilege boundary; no physical I/O."""

import importlib.util
import json
from pathlib import Path
import socket
import struct
from types import SimpleNamespace as NS
import pytest

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "infra/local-controller/query"


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SOURCE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q = load("piper-query")
installer = load("install-query")


class FakeSocket:
    def __init__(self, clock, frames):
        self.clock, self.frames, self.sent = clock, list(frames), []

    def settimeout(self, value):
        self.timeout = value

    def send(self, frame):
        self.sent.append(struct.unpack("=IB3x8s", frame))
        return len(frame)

    def recv(self, size):
        self.clock[0] += 0.02
        if self.frames:
            return self.frames.pop(0)
        self.clock[0] += self.timeout
        raise socket.timeout()


def packet(cid, data):
    return struct.pack("=IB3x8s", cid, len(data), data)


@pytest.fixture
def clock(monkeypatch):
    value = [100.0]
    monkeypatch.setattr(q.time, "monotonic", lambda: value[0])
    return value


def test_only_read_query_frames_are_available():
    assert q.requests("status") == q.requests("link") == ()
    assert q.requests("firmware") == ((0x4AF, b"\x01"),)
    assert q.requests("limits") == tuple(
        (0x472, bytes([j, 1, 0, 0, 0, 0, 0, 0])) for j in range(1, 7)
    )
    assert q.requests("acceleration") == tuple(
        (0x472, bytes([j, 2, 0, 0, 0, 0, 0, 0])) for j in range(1, 7)
    )
    for mode in ("enable", "disable", "reset", "mode", "motion", "exec"):
        with pytest.raises(ValueError):
            q.requests(mode)


def test_state_preserves_faults_without_transmitting(clock):
    frames = [
        packet(0x2A1, bytes([0, 1, 255, 0, 0, 0, 0, 2])),
        packet(0x261, bytes([0, 230, 0, 35, 30, 96, 0, 0])),
        packet(0x2A8, bytes(8)),
    ]
    for cid in (0x2A5, 0x2A6, 0x2A7):
        frames.append(packet(cid, struct.pack(">ii", 1000, -2000)))
    sock = FakeSocket(clock, frames)
    report = {}
    q.query("status", "test-only", sock, report)
    assert sock.sent == []
    assert report["status"] == "partial_feedback"
    assert "0x262" in report["missing_feedback_ids"]
    assert report["decoded_latest"]["joint_deg"] == [1, -2] * 3
    assert report["decoded_latest"]["controller"]["arm_status"] == 1
    assert report["decoded_latest"]["controller"]["err_code"] == 2
    assert report["decoded_latest"]["drivers"][1]["enabled"]
    assert report["decoded_latest"]["drivers"][1]["driver_error"]


def test_firmware_single_send_and_complete_raw_reply(clock):
    data = bytearray(88)
    for start, text in (
        (0, b"H-V1.2-1"),
        (16, b"10"),
        (32, b"ARM_MC"),
        (60, b"S-V1.9-0"),
        (68, b"260711"),
        (76, b"15"),
    ):
        data[start : start + len(text)] = text
    sock = FakeSocket(clock, [packet(0x4AF, data[i : i + 8]) for i in range(0, 88, 8)])
    report = {}
    q.query("firmware", "test-only", sock, report)
    assert len(sock.sent) == 1 and sock.sent[0][:2] == (0x4AF, 1)
    assert report["firmware"]["software_version"] == "S-V1.9-0"
    assert len(report["replies"][0]) == 11


def test_incomplete_query_keeps_raw_evidence_and_does_not_retry(clock):
    sock = FakeSocket(clock, [packet(0x4AF, b"H-V1.2-1")])
    report = {}
    with pytest.raises(RuntimeError, match="no retry"):
        q.query("firmware", "test-only", sock, report)
    assert len(sock.sent) == 1 and len(report["replies"][0]) == 1
    assert report["latest_frames"]["0x4af"]["data"] == b"H-V1.2-1".hex()


def test_limits_send_once_per_joint_and_decode_units(clock):
    sock = FakeSocket(
        clock, [packet(0x473, struct.pack(">BhhHB", j, 1500, -1500, 300, 0)) for j in range(1, 7)]
    )
    report = {}
    q.query("limits", "test-only", sock, report)
    assert len(sock.sent) == 6
    assert all(r["min_deg"] == -150 and r["max_speed_rad_s"] == 3 for r in report["joint_limits"])


def test_acceleration_queries_and_units(clock):
    frames = [packet(0x47C, struct.pack(">BH5x", j, 250 if j == 5 else 500)) for j in range(1, 7)]
    sock = FakeSocket(clock, frames)
    report = {}
    q.query("acceleration", "test-only", sock, report)
    assert len(sock.sent) == 6
    assert [x["max_joint_acc_rad_s2"] for x in report["joint_acceleration_limits"]] == [
        5.0,
        5.0,
        5.0,
        5.0,
        2.5,
        5.0,
    ]
    assert [x["joint"] for x in report["joint_acceleration_limits"]] == list(range(1, 7))


def test_acceleration_wrong_joint_does_not_satisfy_request(clock):
    sock = FakeSocket(clock, [packet(0x47C, struct.pack(">BH5x", 2, 500))])
    report = {}
    with pytest.raises(RuntimeError, match="no retry"):
        q.query("acceleration", "test-only", sock, report)
    assert len(sock.sent) == 1 and report["replies"] == [[]]
    assert "0x47c" in report["latest_frames"]


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["status", "extra"],
        ["exec"],
        ["firmware", "--interface", "other"],
        ["status", "--worker"],
        ["acceleration", "2.5"],
        ["set-acceleration"],
    ],
)
def test_bad_arguments_rejected_before_namespace_or_device_access(monkeypatch, args):
    monkeypatch.setattr(q.os, "geteuid", lambda: 0)
    monkeypatch.setattr(q.subprocess, "run", lambda *a, **k: pytest.fail("must not execute"))
    with pytest.raises(ValueError):
        q.main(args)


def test_nonprivileged_direct_worker_cannot_enter_namespace(monkeypatch):
    monkeypatch.setattr(q.os, "geteuid", lambda: 1234)
    monkeypatch.setattr(q.os.path, "samefile", lambda *a: False)
    with pytest.raises(RuntimeError, match="piper-can"):
        q.worker("status", {"uid": 1234})


@pytest.mark.parametrize("mode", ["firmware", "limits", "acceleration"])
def test_active_query_refuses_existing_namespace_session(monkeypatch, tmp_path, mode):
    cfg = tmp_path / "cfg"
    cfg.write_text(json.dumps({"uid": 1234, "administrator": "operator"}))
    monkeypatch.setattr(q, "CONFIG", cfg)
    monkeypatch.setattr(q.os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setenv("SUDO_USER", "operator")
    monkeypatch.setattr(q.subprocess, "check_output", lambda *a, **k: "999\n")
    monkeypatch.setattr(
        q.subprocess, "run", lambda *a, **k: pytest.fail("must not enter namespace")
    )
    with pytest.raises(RuntimeError, match="active session"):
        q.main([mode])


def test_bridge_drops_privileges_and_uses_only_immutable_system_python(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg"
    cfg.write_text(
        json.dumps({"uid": 1234, "gid": 1234, "administrator": "operator", "interface": "can0"})
    )
    monkeypatch.setattr(q, "CONFIG", cfg)
    monkeypatch.setattr(q.os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setenv("SUDO_USER", "operator")
    calls = []
    monkeypatch.setattr(
        q.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or NS(returncode=0)
    )
    assert q.main(["status"]) == 0
    cmd, kw = calls[0]
    assert "--reuid=1234" in cmd and "--bounding-set=-all" in cmd and "--no-new-privs" in cmd
    assert cmd[-5:] == ["-I", "-S", "/usr/local/libexec/piper-query.py", "status", "--worker"]
    assert kw["env"] == q.SAFE_ENV and kw["timeout"] == 20
    assert json.loads(kw["input"])["uid"] == 1234


def test_sudoers_is_exact_admin_and_five_literal_commands():
    lines = installer.sudoers_text("operator").splitlines()[1:]
    assert len(lines) == 5
    assert all("NOPASSWD: NOSETENV: /usr/local/sbin/piper-query " in line for line in lines)
    assert [line.rsplit(" ", 1)[1] for line in lines] == list(q.MODES)
    for account in ("ALL", "operator\nALL", "%sudo", "root:bad"):
        with pytest.raises(ValueError):
            installer.sudoers_text(account)
    entry = (SOURCE / "piper-query").read_text()
    assert "/usr/bin/env -i" in entry and "/usr/bin/python3 -I -S" in entry
    assert ".venv" not in entry and "piper-socketcan exec" not in entry


def test_transmit_whitelist_matches_pinned_official_codec():
    parser = pytest.importorskip("pyAgxArm.protocols.can_protocol.drivers.piper.default.parser")
    msgs = pytest.importorskip("pyAgxArm.protocols.can_protocol.msgs.piper.default")
    codec = parser.Codec()
    assert q.requests("firmware")[0] == (
        0x4AF,
        bytes(codec.encode_4AF_req_firmware(msgs.ArmMsgReqFirmware())),
    )
    for joint, (cid, data) in enumerate(q.requests("limits"), start=1):
        msg = msgs.ArmMsgSearchMotorMaxAngleSpdAccLimit(joint, 1)
        assert cid == 0x472 and data == bytes(
            codec.encode_472_search_motor_max_angle_spd_acc_limit(msg)
        )
    for joint, (cid, data) in enumerate(q.requests("acceleration"), start=1):
        assert cid == 0x472 and data == bytes(
            codec.encode_472_search_motor_max_angle_spd_acc_limit(
                msgs.ArmMsgSearchMotorMaxAngleSpdAccLimit(joint, 2)
            )
        )
    model = msgs.ArmMsgFeedbackAllCurrentMotorMaxAccLimit()
    codec.decode_47C_motor_max_acc_limit(model, bytearray(struct.pack(">BH5x", 5, 250)))
    assert model.joints[4].max_joint_acc == 2.5


def update_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for key in ["ENTRY", "WORKER", "CONFIG", "SUDOERS"]:
        path = tmp_path / key.lower()
        monkeypatch.setattr(installer, key, path)
    cfg = {
        "administrator": "operator",
        "uid": 1234,
        "gid": 1234,
        "interface": "can0",
        "usb": {"serial": "unchanged"},
    }
    installer.CONFIG.write_text(json.dumps(cfg))
    installer.ENTRY.write_text("launcher")
    installer.WORKER.write_text("old-worker")
    installer.SUDOERS.write_text(installer.sudoers_text("operator", installer.MODES[:-1]))
    for path, mode in [
        (installer.ENTRY, 0o755),
        (installer.WORKER, 0o644),
        (installer.CONFIG, 0o600),
        (installer.SUDOERS, 0o440),
    ]:
        path.chmod(mode)
    (source / "piper-query").write_text("launcher")
    (source / "previous-worker.txt").write_text("old-worker")
    (source / "piper-query.py").write_text("new-worker")
    return source, cfg, NS(pw_uid=1234, pw_gid=1234)


def test_update_preserves_binding_and_requires_reviewed_baseline(tmp_path, monkeypatch):
    source, cfg, user = update_fixture(tmp_path, monkeypatch)
    assert installer.update_payloads(source, "operator", user)[0] == cfg
    installer.WORKER.write_text("local-change")
    with pytest.raises(RuntimeError, match="reviewed update baseline"):
        installer.update_payloads(source, "operator", user)
    assert installer.WORKER.read_text() == "local-change"


@pytest.mark.parametrize("fail_final", [False, True])
def test_update_replaces_only_worker_rules_and_rolls_back_on_failure(
    tmp_path, monkeypatch, fail_final
):
    import os
    import subprocess

    source, cfg, user = update_fixture(tmp_path, monkeypatch)
    before = {
        p: p.read_bytes()
        for p in [installer.CONFIG, installer.ENTRY, installer.WORKER, installer.SUDOERS]
    }
    monkeypatch.setattr(installer, "secure_directory", lambda p: None)
    real_stat = Path.stat

    def root_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path in before:
            fields = list(result)
            fields[4] = 0
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", root_stat)

    def validate(cmd, **kwargs):
        if cmd == ["/usr/sbin/visudo", "-c"] and fail_final:
            raise subprocess.CalledProcessError(1, cmd)
        return NS(returncode=0)

    monkeypatch.setattr(installer.subprocess, "run", validate)
    if fail_final:
        with pytest.raises(subprocess.CalledProcessError):
            installer.update_existing(source, "operator", user)
        assert all(p.read_bytes() == data for p, data in before.items())
    else:
        installer.update_existing(source, "operator", user)
        assert installer.WORKER.read_text() == "new-worker"
        assert installer.SUDOERS.read_text() == installer.sudoers_text("operator")
    assert installer.CONFIG.read_bytes() == before[installer.CONFIG]
    assert installer.ENTRY.read_bytes() == before[installer.ENTRY]
