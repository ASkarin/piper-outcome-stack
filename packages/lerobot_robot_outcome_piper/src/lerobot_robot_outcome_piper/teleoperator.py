"""Xbox USB input with an explicit measured mapping."""

from __future__ import annotations

from typing import Any, Callable

from lerobot.teleoperators.teleoperator import Teleoperator

from .config import OutcomePiperXboxConfig
from .errors import (
    OutcomePiperStateError,
    OutcomePiperValidationError,
    OutcomePiperInputDisconnected,
)
from .input_safety import request_input_emergency_stop, request_input_fault_hold

AXIS_KEYS = ("stick_x", "stick_y", "stick_z", "stick_yaw", "left_trigger", "right_trigger")
CONTROL_KEYS = (
    "hold",
    "neutral",
    "emergency_stop",
    "mode_switch",
    "home",
    "work",
    "translation_switch",
)
RAW_ACTION_KEYS = (*AXIS_KEYS, *CONTROL_KEYS)


class OutcomePiperXbox(Teleoperator):
    config_class = OutcomePiperXboxConfig
    name = "outcome_piper_xbox"

    def __init__(
        self,
        config: OutcomePiperXboxConfig,
        *,
        joystick_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self._joystick_factory = joystick_factory
        self._joystick: Any | None = None
        self._pygame: Any | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {**dict.fromkeys(AXIS_KEYS, float), **dict.fromkeys(CONTROL_KEYS, bool)}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._joystick is not None and bool(self._joystick.get_init())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        raise OutcomePiperStateError("Xbox calibration is external and must be frozen in config")

    def configure(self) -> None:
        if not self.is_connected:
            raise OutcomePiperStateError("Xbox is not connected")

    def connect(self, calibrate: bool = True) -> None:
        try:
            self._connect(calibrate)
        except OutcomePiperInputDisconnected as exc:
            request_input_fault_hold(exc)
            raise
        except Exception as exc:
            request_input_emergency_stop(exc)
            raise

    def _connect(self, calibrate: bool) -> None:
        del calibrate
        if self._joystick is not None:
            raise OutcomePiperStateError("Xbox is already connected")
        if self._joystick_factory is not None:
            joystick = self._joystick_factory(self.config.device_guid)
        else:
            import os
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="pkg_resources is deprecated as an API.*",
                    category=UserWarning,
                    module="pygame.pkgdata",
                )
                import pygame

            # pygame events require its display subsystem. Xbox needs no window,
            # audio or desktop session; use SDL's explicit headless display driver.
            if not pygame.display.get_init():
                previous_driver = os.environ.get("SDL_VIDEODRIVER")
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                try:
                    pygame.display.init()
                finally:
                    if previous_driver is None:
                        os.environ.pop("SDL_VIDEODRIVER", None)
                    else:
                        os.environ["SDL_VIDEODRIVER"] = previous_driver
            pygame.joystick.init()
            matches = []
            for index in range(pygame.joystick.get_count()):
                candidate = pygame.joystick.Joystick(index)
                candidate.init()
                if candidate.get_guid() == self.config.device_guid:
                    matches.append(candidate)
                else:
                    candidate.quit()
            if len(matches) != 1:
                for candidate in matches:
                    candidate.quit()
                raise OutcomePiperStateError(
                    f"expected exactly one Xbox GUID {self.config.device_guid}, found {len(matches)}"
                )
            joystick = matches[0]
            self._pygame = pygame
        if not joystick.get_init():
            joystick.quit()
            raise OutcomePiperStateError("Xbox is disconnected during connection")
        if joystick.get_guid() != self.config.device_guid:
            joystick.quit()
            raise OutcomePiperStateError("Xbox device GUID does not match the frozen mapping")
        max_axis = max(
            self.config.axis_x,
            self.config.axis_y,
            self.config.axis_z,
            self.config.axis_yaw,
            self.config.axis_left_trigger,
            self.config.axis_right_trigger,
        )
        if joystick.get_numaxes() <= max_axis or joystick.get_numbuttons() <= max(
            self.config.hold_button,
            self.config.emergency_stop_button,
            self.config.mode_switch_button,
            self.config.translation_switch_button,
            self.config.home_button,
            self.config.work_pose_button if self.config.work_pose_button is not None else 0,
        ):
            joystick.quit()
            raise OutcomePiperStateError("Xbox device does not match the frozen axis/button layout")
        self._joystick = joystick

    def _axis(self, index: int, sign: int) -> float:
        assert self._joystick is not None
        value = float(self._joystick.get_axis(index)) * sign
        return 0.0 if abs(value) <= self.config.deadzone else value

    def _trigger(self, index: int, side: int) -> float:
        assert self._joystick is not None
        value = float(self._joystick.get_axis(index))
        rest = self.config.trigger_rest_values[side]
        pressed = self.config.trigger_pressed_values[side]
        activation = (value - rest) / (pressed - rest)
        if not 0.0 <= activation <= 1.0:
            raise OutcomePiperValidationError("Xbox trigger is outside its measured range")
        if activation <= self.config.deadzone:
            return 0.0
        return activation

    def poll_emergency_stop(self) -> bool:
        """Poll B during fault-hold confirmation without re-entering input-fault handling."""
        if not self.is_connected:
            return False
        if self._pygame is not None:
            self._pygame.event.pump()
        return bool(self._joystick.get_button(self.config.emergency_stop_button))

    def get_action(self) -> dict[str, float | bool]:
        try:
            return self._get_action()
        except OutcomePiperInputDisconnected as exc:
            request_input_fault_hold(exc)
            raise
        except Exception as exc:
            request_input_emergency_stop(exc)
            raise

    def _get_action(self) -> dict[str, float | bool]:
        if not self.is_connected:
            raise OutcomePiperInputDisconnected("Xbox is disconnected")
        if self._pygame is not None:
            self._pygame.event.pump()
            for event in self._pygame.event.get(self._pygame.JOYDEVICEREMOVED):
                if event.instance_id == self._joystick.get_instance_id():
                    self._joystick.quit()
                    raise OutcomePiperInputDisconnected("selected Xbox was disconnected")
        assert self._joystick is not None
        emergency = bool(self._joystick.get_button(self.config.emergency_stop_button))
        if emergency:
            request_input_emergency_stop("Xbox B/emergency-stop button pressed")
            return {
                **dict.fromkeys(AXIS_KEYS, 0.0),
                "hold": False,
                "neutral": False,
                "emergency_stop": True,
                "mode_switch": False,
                "translation_switch": False,
                "home": False,
                "work": False,
            }
        hold = bool(self._joystick.get_button(self.config.hold_button))
        x = self._axis(self.config.axis_x, self.config.axis_signs[0])
        y = self._axis(self.config.axis_y, self.config.axis_signs[1])
        z = self._axis(self.config.axis_z, self.config.axis_signs[2])
        yaw = self._axis(self.config.axis_yaw, self.config.axis_signs[3])
        left = self._trigger(self.config.axis_left_trigger, 0)
        right = self._trigger(self.config.axis_right_trigger, 1)
        neutral = all(v == 0.0 for v in (x, y, z, yaw, left, right))
        return {
            "stick_x": x,
            "stick_y": y,
            "stick_z": z,
            "stick_yaw": yaw,
            "left_trigger": left,
            "right_trigger": right,
            "hold": hold,
            "neutral": neutral,
            "emergency_stop": False,
            "mode_switch": bool(self._joystick.get_button(self.config.mode_switch_button)),
            "translation_switch": bool(
                self._joystick.get_button(self.config.translation_switch_button)
            ),
            "home": bool(self._joystick.get_button(self.config.home_button)),
            "work": self.config.work_pose_button is not None
            and bool(self._joystick.get_button(self.config.work_pose_button)),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if feedback:
            raise OutcomePiperValidationError("Xbox feedback is unsupported")

    def disconnect(self) -> None:
        joystick = self._joystick
        self._joystick = None
        if joystick is not None:
            joystick.quit()
        if self._pygame is not None:
            self._pygame.joystick.quit()
            self._pygame.quit()
            self._pygame = None
