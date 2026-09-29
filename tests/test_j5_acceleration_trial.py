from pathlib import Path
from types import SimpleNamespace
import runpy
import pytest

m = runpy.run_path(
    str(Path(__file__).parents[1] / "infra/acceptance/piper_j5_acceleration_trial.py")
)


class Arm:
    def __init__(self, value=5.0, ok=True):
        self.value = value
        self.ok = ok
        self.writes = []
        self.closed = False

    def connect(self):
        pass

    def disconnect(self):
        self.closed = True

    def has_comm_error(self):
        return False

    def get_joint_acc_limits(self, j, **kw):
        assert j == 5
        return SimpleNamespace(msg=SimpleNamespace(max_joint_acc=self.value))

    def set_joint_acc_limits(self, j, *, max_joint_acc, timeout):
        self.writes.append((j, max_joint_acc))
        self.value = max_joint_acc
        return self.ok


@pytest.mark.parametrize("before,after", [(5.0, 2.5), (2.5, 5.0)])
def test_only_j5_written_once_and_readback_verified(before, after):
    arm = Arm(before)
    report = {}
    m["apply"](arm, before, after, report)
    assert arm.writes == [(5, after)] and arm.closed
    assert report["status"] == "readback_confirmed" and report["after_rad_s2"] == after


def test_unexpected_current_value_is_not_overwritten():
    arm = Arm(3.0)
    report = {}
    with pytest.raises(ValueError):
        m["apply"](arm, 5.0, 2.5, report)
    assert not arm.writes and arm.closed


def test_sdk_failure_records_actual_readback_without_retry_or_restore():
    arm = Arm(ok=False)
    report = {}
    with pytest.raises(RuntimeError):
        m["apply"](arm, 5.0, 2.5, report)
    assert arm.writes == [(5, 2.5)] and report["after_rad_s2"] == 2.5
    assert report["status"] == "failed" and arm.closed
