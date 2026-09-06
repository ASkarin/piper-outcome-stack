from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


spec = importlib.util.spec_from_file_location(
    "piper_read_only_probe",
    Path(__file__).parents[1] / "infra/acceptance/piper_read_only_probe.py",
)
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)


@pytest.mark.parametrize(
    ("version", "driver"),
    [("S-V1.6-3", "default"), ("S-V1.8-3", "v183"), ("S-V1.8-8", "v188"), ("S-V1.8-9", "v189")],
)
def test_select_driver_from_actual_identity(version, driver):
    assert (
        probe_module.firmware_driver({"node_type": "ARM_MC", "software_version": version}) == driver
    )


def test_firmware_query_is_shared_by_all_pinned_piper_drivers():
    pytest.importorskip("pyAgxArm")
    from pyAgxArm.protocols.can_protocol.drivers.piper.default.driver import Driver
    from pyAgxArm.protocols.can_protocol.drivers.piper.versions.v183.driver import Driver as V183
    from pyAgxArm.protocols.can_protocol.drivers.piper.versions.v188.driver import Driver as V188
    from pyAgxArm.protocols.can_protocol.drivers.piper.versions.v189.driver import Driver as V189

    assert Driver.get_firmware is V183.get_firmware is V188.get_firmware is V189.get_firmware


class Arm:
    OPTIONS = NS(EFFECTOR=NS(AGX_GRIPPER="agx_gripper"))

    def __init__(self, calls, clock, driver):
        self.calls, self.clock, self.driver = calls, clock, driver
        self._parser = NS()
        self.missing_driver = False
        self.enabled = False

    def wrap(self, msg):
        return NS(msg=msg, timestamp=self.clock[0], hz=50.0)

    def connect(self):
        self.calls.append((self.driver, "connect"))

    def disconnect(self):
        self.calls.append((self.driver, "disconnect"))

    def get_firmware(self, **kwargs):
        self.calls.append((self.driver, "get_firmware"))
        return {"node_type": "ARM_MC", "software_version": "S-V1.8-9"}

    def has_comm_error(self):
        return False

    def get_fps(self):
        return 50.0

    def init_effector(self, name):
        self.calls.append((self.driver, "init_effector"))
        return self

    def get_gripper_status(self):
        from pyAgxArm.protocols.can_protocol.msgs.effector.agx_gripper.default import (
            ArmMsgFeedbackGripper,
        )

        return self.wrap(ArmMsgFeedbackGripper(value=0.03, mode="width"))

    def get_arm_status(self):
        return self.wrap(NS(ctrl_mode=0, arm_status=0, mode_feedback=0, err_code=0))

    def get_driver_states(self, index):
        if self.missing_driver and index == 6:
            return None
        return self.wrap(
            NS(
                foc_status=NS(driver_enable_status=self.enabled, driver_error_status=False),
                vol=24.0,
                foc_temp=25.0,
                motor_temp=25.0,
            )
        )

    def get_joint_angles(self):
        for name in ("joint_12", "joint_34", "joint_56"):
            setattr(self._parser, name, self.wrap(None))
        return self.wrap([0.0] * 6)


def setup(monkeypatch):
    pytest.importorskip("pyAgxArm")
    clock = [1000.0]
    monkeypatch.setattr(probe_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(probe_module.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        probe_module.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    calls = []
    arms = []

    def factory(interface, driver):
        arm = Arm(calls, clock, driver)
        arms.append(arm)
        return arm

    return factory, calls, arms


def test_probe_queries_once_then_collects_without_motion_calls(monkeypatch):
    factory, calls, _ = setup(monkeypatch)
    report = {}
    probe_module.probe(factory, "can-test", report, duration_s=0.3)
    assert report["status"] == "feedback_received_motors_disabled"
    assert report["samples"][-1]["gripper"]["value"]["value"] == 0.03
    assert calls == [
        ("default", "connect"),
        ("default", "get_firmware"),
        ("default", "disconnect"),
        ("v189", "connect"),
        ("v189", "init_effector"),
        ("v189", "disconnect"),
    ]


def test_probe_waits_for_passive_reception_before_firmware_query(monkeypatch):
    factory, calls, _ = setup(monkeypatch)

    def warming_factory(interface, driver):
        arm = factory(interface, driver)
        if driver == "default":
            rates = iter([0.0, 0.0, 50.0])
            arm.get_fps = lambda: next(rates)
            original_query = arm.get_firmware

            def query(**kwargs):
                assert arm.clock[0] >= 1000.09
                return original_query(**kwargs)

            arm.get_firmware = query
        return arm

    probe_module.probe(warming_factory, "can-test", {}, duration_s=0.3)
    assert calls.count(("default", "get_firmware")) == 1


def test_quiet_bus_still_gets_its_first_firmware_query(monkeypatch):
    factory, calls, _ = setup(monkeypatch)

    def quiet_factory(interface, driver):
        arm = factory(interface, driver)
        if driver == "default":
            arm.get_fps = lambda: 0.0
        return arm

    report = {}
    probe_module.probe(quiet_factory, "can-test", report, duration_s=0.3)
    assert report["passive_feedback_before_query"] is False
    assert report["status"] == "feedback_received_motors_disabled"
    assert calls.count(("default", "get_firmware")) == 1


@pytest.mark.parametrize("fault", ["missing_driver", "enabled"])
def test_probe_does_not_treat_missing_or_enabled_motor_as_pass(monkeypatch, fault):
    factory, calls, _ = setup(monkeypatch)

    def altered_factory(interface, driver):
        arm = factory(interface, driver)
        if driver == "v189":
            setattr(arm, fault, True)
        return arm

    with pytest.raises(RuntimeError):
        probe_module.probe(altered_factory, "can-test", {}, duration_s=0.3)
    assert calls[-1] == ("v189", "disconnect")


def test_stale_group_is_rejected(monkeypatch):
    factory, _, _ = setup(monkeypatch)
    arm = factory("can-test", "v189")
    sample = probe_module.snapshot(arm, arm)
    sample["joint_group_timestamps_s"][0] = 998.0
    with pytest.raises(RuntimeError, match="did not refresh"):
        probe_module.validate_snapshot(sample, 999.0)
