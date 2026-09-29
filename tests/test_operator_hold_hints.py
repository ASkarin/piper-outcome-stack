# ruff: noqa: F811
import logging
import pytest
from test_xbox_pause import session, settle  # noqa: F401
from lerobot_robot_outcome_piper.console import OperatorInfoFilter, OperatorFormatter


@pytest.mark.parametrize("phase", ["review", "saving", "finalizing"])
def test_hold_hint_does_not_offer_movement_in_non_control_phases(session, caplog, phase):
    robot, arm, control, clock = session
    settle(session)
    control.recording_phase = phase
    control.reconfirm_hold()
    clock.advance()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        robot._update_hold_locked()
    messages = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[保持已确认]")]
    assert len(messages) == 1
    assert "按LB" not in messages[0] and "推杆继续" not in messages[0]


def test_operator_formatter_keeps_details_in_file_record():
    record = logging.LogRecord(
        "test",
        logging.INFO,
        "file.py",
        1,
        "Xbox 机械臂新输入 epoch=%s: %s",
        (3, {"stick_x": 0.5}),
        None,
    )
    assert not OperatorInfoFilter().filter(record)
    assert "epoch=3" in record.getMessage()
    status = logging.LogRecord(
        "test", logging.INFO, "file.py", 1, "[保持已确认] 请等待保存。", (), None
    )
    assert (
        OperatorFormatter(logging.Formatter("%(levelname)s %(message)s")).format(status)
        == status.msg
    )
