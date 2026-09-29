import io
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from lerobot_robot_outcome_piper.console import operator_console
from lerobot_robot_outcome_piper.teleoperator import OutcomePiperXbox
from test_plugin import FakeJoystick, xbox_config


def test_console_keeps_prompts_and_errors_and_archives_library_info(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    config = tmp_path / "record.json"
    config.write_text("{}")
    try:
        with operator_console([f"--config_path={config}"]) as log_path:
            root.info("Imported third-party plugin: example")
            root.info("Using video codec: codec")
            root.info("Created a socket")
            record = logging.LogRecord(
                "camera",
                logging.INFO,
                "camera_realsense.py",
                1,
                "TimedRealSenseCamera(1) disconnected.",
                (),
                None,
            )
            root.handle(record)
            root.info("Xbox 保持已确认")
            root.warning("camera unavailable")
            root.error("real controller fault")
        assert "Imported third-party" not in stream.getvalue()
        assert "video codec" not in stream.getvalue()
        assert "Created a socket" not in stream.getvalue()
        assert "disconnected." not in stream.getvalue()
        assert "disconnected." in log_path.read_text()
        assert "保持已确认" in stream.getvalue()
        assert "camera unavailable" in stream.getvalue()
        assert "real controller fault" in stream.getvalue()
        log = log_path.read_text()
        assert log_path.parent == tmp_path / "piper-runs" / "logs"
        assert not config.with_suffix(".log").exists()
        assert "Imported third-party" in log and "Created a socket" in log
        assert root.handlers == [handler] and not handler.filters
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)


def test_xbox_uses_headless_event_display_without_global_pygame_init(monkeypatch):
    joystick = FakeJoystick()
    joystick.init = lambda: None
    initialized = []

    def display_init():
        assert os.environ["SDL_VIDEODRIVER"] == "dummy"
        initialized.append("display")

    pygame = NS(
        init=lambda: (_ for _ in ()).throw(AssertionError("global pygame.init must not run")),
        display=NS(get_init=lambda: False, init=display_init),
        joystick=NS(
            init=lambda: initialized.append("joystick"),
            get_count=lambda: 1,
            Joystick=lambda i: joystick,
            quit=lambda: None,
        ),
        event=NS(pump=lambda: None, get=lambda kind: []),
        JOYDEVICEREMOVED=1,
        quit=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setenv("SDL_VIDEODRIVER", "existing-driver")
    xbox = OutcomePiperXbox(xbox_config())
    try:
        xbox.connect()
        assert initialized == ["display", "joystick"]
        assert os.environ["SDL_VIDEODRIVER"] == "existing-driver"
        assert not xbox.poll_emergency_stop()
        joystick.buttons[0] = 1
        assert xbox.poll_emergency_stop()
        assert xbox.get_action()["emergency_stop"]
    finally:
        xbox.disconnect()


def test_real_sdl_dummy_event_pump_needs_no_desktop_or_joystick_device():
    import importlib.util
    import pytest
    import subprocess

    if importlib.util.find_spec("pygame") is None:
        pytest.skip("pygame is installed in the Linux controller environment")
    script = """
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API.*',
                            category=UserWarning, module='pygame.pkgdata')
    import pygame
pygame.display.init()
assert pygame.display.get_driver() == 'dummy'
pygame.event.pump()
pygame.display.quit()
print('headless event pump passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "headless event pump passed" in result.stdout
    assert "XDG_RUNTIME_DIR" not in result.stderr
