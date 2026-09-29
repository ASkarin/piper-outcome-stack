from pathlib import Path
import runpy
from types import SimpleNamespace
import pytest

m = runpy.run_path(str(Path(__file__).parents[1] / "infra/acceptance/piper_acceleration_query.py"))


class Reader:
    def __init__(self, missing=None, error=False):
        self.calls = []
        self.missing = missing
        self.error = error

    def connect(self):
        self.calls.append("connect")

    def disconnect(self):
        self.calls.append("disconnect")

    def get_joint_acc_limits(self, joint, **kwargs):
        self.calls.append((joint, kwargs))
        return (
            None
            if joint == self.missing
            else SimpleNamespace(msg=SimpleNamespace(max_joint_acc=5.0), timestamp=123.0)
        )

    def has_comm_error(self):
        return self.error

    def get_comm_error(self):
        return "injected error"


def test_only_six_queries_and_no_parameter_changes():
    arm = Reader()
    report = {"joints": []}
    m["query"](arm, report)
    assert arm.calls == [
        "connect",
        *[(i, {"timeout": 1.0, "min_interval": 0.0}) for i in range(1, 7)],
        "disconnect",
    ]
    assert report["status"] == "read_complete" and len(report["joints"]) == 6
    assert all(x["max_joint_acc_rad_s2"] == 5.0 for x in report["joints"])


def test_missing_reply_preserves_partial_read_and_does_not_retry():
    arm = Reader(missing=3)
    report = {"joints": []}
    with pytest.raises(RuntimeError, match="no response"):
        m["query"](arm, report)
    assert len(report["joints"]) == 2 and report["status"] == "failed"
    assert [x[0] for x in arm.calls if isinstance(x, tuple)] == [1, 2, 3]
    assert arm.calls[-1] == "disconnect"


def test_communication_fault_stops_queries_without_recovery():
    arm = Reader(error=True)
    report = {"joints": []}
    with pytest.raises(RuntimeError, match="CAN error"):
        m["query"](arm, report)
    assert report["status"] == "failed" and not report["joints"]
    assert arm.calls[-1] == "disconnect" and len(arm.calls) == 3
