from types import SimpleNamespace as NS
import pytest
from lerobot_robot_outcome_piper.capture_gc import CaptureGC


class Collector:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.calls = []

    def isenabled(self):
        return self.enabled

    def collect(self, n):
        self.calls.append(("collect", n))

    def disable(self):
        self.calls.append("disable")
        self.enabled = False

    def enable(self):
        self.calls.append("enable")
        self.enabled = True


@pytest.mark.parametrize("enabled", [True, False])
def test_collection_is_outside_capture_and_prior_setting_restored(enabled):
    gc = Collector(enabled)
    ticks = iter([0.0, 0.09])
    guard = CaptureGC(gc, lambda: next(ticks))
    guard.prepare()
    guard.start()
    assert not gc.enabled
    with pytest.raises(RuntimeError, match="sampling"):
        guard.prepare()
    guard.stop()
    guard.stop()
    assert gc.enabled == enabled
    assert gc.calls == ([("collect", 2), "disable", "enable"] if enabled else [])
    assert guard.summary()["restored"]


def test_run_session_restores_gc_after_sampling_failure(tmp_path, monkeypatch):
    from lerobot_robot_outcome_piper import capture_gc
    from lerobot_robot_outcome_piper.teach_record import run_session
    from test_teach_recording import config, sample, Clock

    gc = Collector()
    guard = CaptureGC(gc)
    monkeypatch.setattr(capture_gc, "CaptureGC", lambda: guard)
    commands = iter(["start P3", None])
    clock = Clock()

    def read():
        assert not gc.enabled
        raise RuntimeError("synthetic frame failure")

    source = NS(feedback=lambda: sample(0, clock())[1], read=read)
    with pytest.raises(RuntimeError, match="frame failure"):
        run_session(
            source,
            config(),
            tmp_path,
            NS(poll=lambda: next(commands, None)),
            clock=clock,
            sleep=clock.sleep,
        )
    assert gc.enabled and guard.summary()["restored"]


def test_interrupt_during_disable_still_allows_restore():
    collector = Collector()

    def disable():
        collector.enabled = False
        raise KeyboardInterrupt()

    collector.disable = disable
    guard = CaptureGC(collector)
    with pytest.raises(KeyboardInterrupt):
        try:
            guard.start()
        finally:
            guard.stop()
    assert collector.enabled
