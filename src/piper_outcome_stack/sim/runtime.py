"""Headless or desktop MuJoCo runtime with an explicitly unidentified servo."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .model import ASSETS, FINGERS, JOINTS, ROOT, fmt, validate_action, validate_values

DEFAULT_CONFIG = ROOT / "configs/simulation/tabletop.json"


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    if (
        cfg.get("simulation_only") is not True
        or cfg.get("calibration_status") != "illustrative_unidentified"
    ):
        raise ValueError("S0/S1 requires an explicitly simulation_only unidentified scene")
    if (
        cfg.get("timestep_s") != 0.002
        or cfg.get("display_fps") != 30
        or cfg.get("servo_frequency_hz") != 2.0
    ):
        raise ValueError("S0/S1 uses 2 ms physics, 30 FPS display and a 2 Hz model servo")
    for name in ("base_xyz_m", "table_center_m", "table_half_size_m"):
        x = np.asarray(cfg[name], dtype=float)
        if x.shape != (3,) or not np.isfinite(x).all():
            raise ValueError(f"invalid {name}")
    if min(cfg["table_half_size_m"]) <= 0:
        raise ValueError("table half sizes must be positive")
    camera = cfg["camera"]
    if camera["name"] != "d435":
        raise ValueError("only a single virtual d435 RGB camera is supported")
    for key in ("width", "height"):
        if type(camera[key]) is not int or not 0 < camera[key] <= 2048:
            raise ValueError("camera size must be within 1..2048 pixels")
    for key, n in [
        ("position_m", 3),
        ("look_at_m", 3),
        ("focal_pixels", 2),
        ("principal_pixels", 2),
    ]:
        a = np.asarray(camera[key], dtype=float)
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError(f"invalid camera {key}")
    if min(camera["focal_pixels"]) <= 0:
        raise ValueError("camera focal lengths must be positive")
    return cfg


class Simulation:
    def __init__(self, config, *, asset_dir=ASSETS):
        self.config, self.asset_dir = config, Path(asset_dir)
        # The checked-in conversion is used in runs; tests verify reproducibility.
        xml = ET.fromstring((self.asset_dir / "piper.xml").read_text())
        xml.find("compiler").set("meshdir", str(self.asset_dir / "meshes"))
        world = xml.find("worldbody")
        world.find("body[@name='base_link']").set("pos", fmt(config["base_xyz_m"]))
        ET.SubElement(
            world,
            "geom",
            name="table",
            type="box",
            pos=fmt(config["table_center_m"]),
            size=fmt(config["table_half_size_m"]),
            rgba="0.55 0.43 0.3 1",
        )
        ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
        camera = config["camera"]
        direction = np.subtract(camera["look_at_m"], camera["position_m"])
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            raise ValueError("camera position and target coincide")
        forward = direction / norm
        right = np.cross(forward, [0, 0, 1])
        if np.linalg.norm(right) < 1e-9:
            raise ValueError("camera direction is parallel to its up vector")
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        ET.SubElement(
            world,
            "camera",
            name="d435",
            pos=fmt(camera["position_m"]),
            xyaxes=fmt(np.r_[right, up]),
            resolution=f"{camera['width']} {camera['height']}",
            sensorsize="0.0064 0.0048",
            focalpixel=fmt(camera["focal_pixels"]),
            # Config uses top-left image coordinates; MJCF uses opposite-signed offsets.
            principalpixel=fmt(
                np.array([camera["width"], camera["height"]]) / 2
                - np.asarray(camera["principal_pixels"])
            ),
        )
        visual = ET.SubElement(xml, "visual")
        ET.SubElement(
            visual, "global", offwidth=str(camera["width"]), offheight=str(camera["height"])
        )
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.jids = np.array([self.model.joint(n).id for n in (*JOINTS, *FINGERS)])
        self.qids = self.model.jnt_qposadr[self.jids]
        self.dids = self.model.jnt_dofadr[self.jids]
        self.actuator_ids = np.array([self.model.actuator(n + "_servo").id for n in JOINTS])
        self.gripper_actuator = self.model.actuator("gripper_servo").id
        self.renderer = None
        self.contact_counts = {}
        self.max_penetration_m = 0.0
        self.set_measured(np.zeros(7))
        mass = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, mass)
        effective = np.diag(mass)[self.dids[:6]]
        # Total width w -> [finger1=w/2, finger2=-w/2].
        g = np.zeros(self.model.nv)
        g[self.dids[6]] = 0.5
        g[self.dids[7]] = -0.5
        masses = np.r_[effective, g @ mass @ g]
        if np.any(masses <= 0) or not np.isfinite(masses).all():
            raise ValueError("invalid zero-position effective mass")
        omega = 2 * math.pi * config["servo_frequency_hz"]
        kp, kv = masses * omega**2, 2 * masses * omega
        ids = np.r_[self.actuator_ids, self.gripper_actuator]
        self.model.actuator_gainprm[ids, 0] = kp
        self.model.actuator_biasprm[ids, 1] = -kp
        self.model.actuator_biasprm[ids, 2] = -kv
        self.servo = {
            "identified": False,
            "frequency_hz": 2.0,
            "effective_mass": masses.tolist(),
            "kp": kp.tolist(),
            "kv": kv.tolist(),
            "force_limits": self.model.actuator_forcerange[ids].tolist(),
        }
        self.set_target(np.zeros(7))

    def set_measured(self, action, *, recorded=False):
        q = validate_values(action) if recorded else validate_action(action, self.asset_dir)
        self.data.qpos[self.qids] = np.r_[q[:6], q[6] / 2, -q[6] / 2]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def action(self):
        q = self.data.qpos[self.qids]
        return np.r_[q[:6], q[6] - q[7]].copy()

    def set_target(self, action):
        q = validate_action(action, self.asset_dir)
        self.data.ctrl[self.actuator_ids] = q[:6]
        self.data.ctrl[self.gripper_actuator] = q[6]

    def step(self):
        warnings = np.array([w.number for w in self.data.warning])
        before = float(self.data.time)
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("simulation became non-finite")
        if (
            np.any(np.array([w.number for w in self.data.warning]) > warnings)
            or self.data.time <= before
        ):
            raise RuntimeError("MuJoCo warning or simulation time reset; inspect model contacts")
        self.record_contacts()

    def record_contacts(self):
        for c in self.data.contact:
            names = sorted((self.model.geom(c.geom1).name, self.model.geom(c.geom2).name))
            key = " / ".join(names)
            self.contact_counts[key] = self.contact_counts.get(key, 0) + 1
            self.max_penetration_m = max(self.max_penetration_m, max(0.0, -float(c.dist)))

    def flange_matrix(self):
        site = self.data.site("flange")
        t = np.eye(4)
        t[:3, :3] = site.xmat.reshape(3, 3)
        t[:3, 3] = site.xpos
        t[:3, 3] -= self.config["base_xyz_m"]
        return t

    def status_text(self):
        pose = self.flange_matrix()
        rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        return (
            "q deg: "
            + np.array2string(np.rad2deg(self.action()[:6]), precision=1)
            + f"\nwidth: {self.action()[6] * 1000:.1f} mm\nflange mm: "
            + np.array2string(pose[:3, 3] * 1000, precision=1)
            + "\nRPY deg: "
            + np.array2string(rpy, precision=1)
        )

    def render(self):
        if self.renderer is None:
            camera = self.config["camera"]
            self.renderer = mujoco.Renderer(
                self.model, height=camera["height"], width=camera["width"]
            )
        self.renderer.update_scene(self.data, camera="d435")
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def visible_robot_pixels(self):
        if self.renderer is None:
            self.render()
        self.renderer.enable_segmentation_rendering()
        try:
            labels = self.renderer.render()
            robot_geoms = [
                i for i in range(self.model.ngeom) if self.model.geom(i).name.endswith("_mesh")
            ]
            return int(
                (
                    np.isin(labels[:, :, 0], robot_geoms)
                    & (labels[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                ).sum()
            )
        finally:
            self.renderer.disable_segmentation_rendering()
