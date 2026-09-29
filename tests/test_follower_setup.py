import sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "infra/acceptance"))
from piper_follower_setup import configure_once


@pytest.mark.parametrize(
    "states,expected,sent",
    [
        ([True], "feedback_already_present_no_change", 0),
        ([False, True], "feedback_started_before_command_no_change", 0),
        ([False, False, True], "feedback_restored", 1),
        ([False, False, False], "feedback_incomplete_after_single_request", 1),
    ],
)
def test_single_request_never_retries_or_enables(states, expected, sent):
    answers = iter(states)
    calls = []
    arm = NS(has_comm_error=lambda: False, set_follower_mode=lambda: calls.append("follower"))
    receiver = NS(wait_ready=lambda _: next(answers))
    report = {}
    configure_once(arm, receiver, report, confirm=lambda _: "", snapshot=lambda _: {})
    assert report["status"] == expected and calls == ["follower"] * sent


def test_cancel_and_communication_error_send_nothing():
    arm = NS(has_comm_error=lambda: True)
    rx = NS(wait_ready=lambda _: False)
    report = {}
    configure_once(arm, rx, report, confirm=lambda _: "cancel", snapshot=lambda _: {})
    assert report["status"] == "cancelled"
    with pytest.raises(RuntimeError, match="CAN error before"):
        configure_once(arm, rx, {}, confirm=lambda _: "", snapshot=lambda _: {})
