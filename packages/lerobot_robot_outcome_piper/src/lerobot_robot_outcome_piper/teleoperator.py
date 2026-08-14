"""Xbox USB input with an explicit measured mapping."""

from __future__ import annotations

from typing import Any, Callable

from lerobot.teleoperators.teleoperator import Teleoperator

from .config import OutcomePiperXboxConfig
from .errors import OutcomePiperStateError, OutcomePiperValidationError
from .input_safety import request_input_emergency_stop

RAW_ACTION_KEYS = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_yaw",
    "delta_gripper",
    "hold",
)


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
        return {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_yaw": float,
            "delta_gripper": float,
            "hold": bool,
        }

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
            import pygame

            pygame.init()
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
        if (
            joystick.get_numaxes() <= max_axis
            or joystick.get_numbuttons() <= self.config.hold_button
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

    def get_action(self) -> dict[str, float | bool]:
        try:
            return self._get_action()
        except Exception as exc:
            request_input_emergency_stop(exc)
            raise

    def _get_action(self) -> dict[str, float | bool]:
        if not self.is_connected:
            raise OutcomePiperStateError("Xbox is disconnected")
        if self._pygame is not None:
            self._pygame.event.pump()
        assert self._joystick is not None
        hold = bool(self._joystick.get_button(self.config.hold_button))
        x = self._axis(self.config.axis_x, self.config.axis_signs[0])
        y = self._axis(self.config.axis_y, self.config.axis_signs[1])
        z = self._axis(self.config.axis_z, self.config.axis_signs[2])
        yaw = self._axis(self.config.axis_yaw, self.config.axis_signs[3])
        left = self._trigger(self.config.axis_left_trigger, 0)
        right = self._trigger(self.config.axis_right_trigger, 1)
        if not hold:
            x = y = z = yaw = left = right = 0.0
        return {
            "delta_x": x * self.config.xyz_step_m,
            "delta_y": y * self.config.xyz_step_m,
            "delta_z": z * self.config.xyz_step_m,
            "delta_yaw": yaw * self.config.yaw_step_rad,
            "delta_gripper": (right - left) * self.config.gripper_step_m,
            "hold": hold,
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
