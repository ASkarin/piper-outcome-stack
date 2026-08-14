"""The single supported pyAgxArm binding."""

from __future__ import annotations

from typing import Any, Callable


def create_piper(can_interface: str, firmware: str) -> Any:
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config

    config = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=firmware,
        interface="socketcan",
        channel=can_interface,
        auto_connect=False,
    )
    return AgxArmFactory.create_arm(config)


PiperFactory = Callable[[str, str], Any]
