"""Input measurement tests never import the robot or camera plugins."""

from types import SimpleNamespace as NS
import ast
from pathlib import Path
import pytest
from piper_outcome_stack.ops import xbox_input


def test_sampler_preserves_short_b_press_and_device_removal():
    clock = [0.0]

    def sleep(dt):
        clock[0] += dt

    events = [NS(type=1, instance_id=42, button=3), NS(type=2, instance_id=42, button=3)]

    def get():
        result = events[:]
        events.clear()
        return result

    pg = NS(
        JOYBUTTONDOWN=1, JOYBUTTONUP=2, JOYDEVICEREMOVED=3, event=NS(pump=lambda: None, get=get)
    )
    joystick = NS(
        get_instance_id=lambda: 42,
        get_init=lambda: True,
        get_numaxes=lambda: 1,
        get_numbuttons=lambda: 4,
        get_numhats=lambda: 0,
        get_axis=lambda i: 0.02,
        get_button=lambda i: 0,
    )
    log = []
    result = xbox_input.sample_stage(
        pg,
        joystick,
        0.1,
        60,
        lambda event, **kw: log.append((event, kw)),
        clock=lambda: clock[0],
        sleep=sleep,
    )
    assert result["pressed_buttons"] == [3]
    assert result["axes_min"] == [0.02] and result["axes_max"] == [0.02]
    assert [x[1]["pressed"] for x in log if x[0] == "button"] == [True, False]
    events.append(NS(type=3, instance_id=42))
    with pytest.raises(RuntimeError, match="disconnected"):
        xbox_input.sample_stage(
            pg, joystick, 0.1, 60, lambda *a, **k: None, clock=lambda: clock[0], sleep=sleep
        )


def test_button_mapping_requires_distinct_actual_presses():
    stages = {
        "neutral": {"pressed_buttons": []},
        "hold_button": {"pressed_buttons": [4]},
        "emergency_stop_button": {"pressed_buttons": [1]},
    }
    assert xbox_input.measured_buttons(stages) == {"hold_button": 4, "emergency_stop_button": 1}
    stages["emergency_stop_button"]["pressed_buttons"] = [4]
    with pytest.raises(ValueError, match="same button"):
        xbox_input.measured_buttons(stages)
    stages["emergency_stop_button"]["pressed_buttons"] = []
    with pytest.raises(ValueError, match="exactly one"):
        xbox_input.measured_buttons(stages)


def test_input_tool_has_no_control_backend_imports():
    tree = ast.parse(Path(xbox_input.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(x.name for x in node.names)
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("lerobot", "pyAgxArm", "can", "pyrealsense2", "socket", "subprocess")
    assert not any(name.startswith(forbidden) for name in imported)
