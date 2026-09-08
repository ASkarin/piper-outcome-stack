"""Narrow, reproducible conversion of the pinned PiPER URDF and gripper Xacro."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

SOURCE_COMMIT = "f6642ce0d7872c686f29c99e9e10cd23d1d49313"
ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/piper"
JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGERS = ("gripper_joint1", "gripper_joint2")
ACTION_KEYS = (*tuple(f"joint_{i}.pos" for i in range(1, 7)), "gripper.pos")
# Existing controller readout, not new real-robot safety limits.
CONTROLLER_LOWER = np.deg2rad([-150, 0, -170, -100, -70, -180])
CONTROLLER_UPPER = np.deg2rad([150, 180, 0, 100, 70, 180])


def values(text, size=3):
    a = np.array([float(x) for x in text.split()])
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"expected {size} finite values: {text}")
    return a


def fmt(a):
    return " ".join(format(float(x), ".17g") for x in a)


def origin(node):
    o = node.find("origin")
    xyz = values(o.get("xyz", "0 0 0")) if o is not None else np.zeros(3)
    rpy = values(o.get("rpy", "0 0 0")) if o is not None else np.zeros(3)
    return xyz, Rotation.from_euler("xyz", rpy)


def source_tree(asset_dir=ASSETS):
    arm = ET.parse(asset_dir / "upstream/piper_description.urdf").getroot()
    gripper = ET.parse(asset_dir / "upstream/piper_with_gripper_description.xacro").getroot()
    for element in gripper:
        if element.tag in ("link", "joint"):
            arm.append(element)
    return arm


@lru_cache(maxsize=4)
def limits(asset_dir=ASSETS):
    src = source_tree(asset_dir)
    joints = {x.get("name"): x for x in src.findall("joint")}
    lower = np.array([float(joints[n].find("limit").get("lower")) for n in JOINTS])
    upper = np.array([float(joints[n].find("limit").get("upper")) for n in JOINTS])
    width = float(joints["gripper"].find("limit").get("upper"))
    return np.r_[np.maximum(lower, CONTROLLER_LOWER), 0.0], np.r_[
        np.minimum(upper, CONTROLLER_UPPER), width
    ]


def validate_values(action):
    if isinstance(action, dict):
        if set(action) != set(ACTION_KEYS):
            raise ValueError("action requires exactly six joint positions and gripper.pos")
        action = [action[k] for k in ACTION_KEYS]
    q = np.asarray(action, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("action requires seven finite rad/m values")
    return q.copy()


def validate_action(action, asset_dir=ASSETS):
    q = validate_values(action)
    lo, hi = limits(asset_dir)
    if np.any(q < lo) or np.any(q > hi):
        raise ValueError("action is outside simulation model/controller intersection")
    return q.copy()


def urdf_fk(action, asset_dir=ASSETS):
    """Independent original-URDF traversal, including original mimic declarations."""
    q = validate_action(action, asset_dir)
    position = dict(zip(JOINTS, q[:6], strict=True))
    position["gripper"] = q[6]
    frames = {"world": np.eye(4)}
    pending = list(source_tree(asset_dir).findall("joint"))
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in frames]
        if not ready:
            raise ValueError("URDF joint graph is disconnected")
        for j in ready:
            xyz, rot = origin(j)
            t = np.eye(4)
            t[:3, :3], t[:3, 3] = rot.as_matrix(), xyz
            motion = np.eye(4)
            name, kind = j.get("name"), j.get("type")
            if kind != "fixed":
                mimic = j.find("mimic")
                angle = position.get(name, 0.0)
                if mimic is not None:
                    angle = position[mimic.get("joint")] * float(
                        mimic.get("multiplier", 1)
                    ) + float(mimic.get("offset", 0))
                axis = values(j.find("axis").get("xyz"))
                if kind == "revolute":
                    motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
                elif kind == "prismatic":
                    motion[:3, 3] = axis * angle
                else:
                    raise ValueError(f"unsupported PiPER joint type: {kind}")
            frames[j.find("child").get("link")] = frames[j.find("parent").get("link")] @ t @ motion
            pending.remove(j)
    return frames


def build_xml(asset_dir=ASSETS):
    src = source_tree(asset_dir)
    links = {x.get("name"): x for x in src.findall("link")}
    joints = src.findall("joint")
    lo, hi = limits(asset_dir)
    root = ET.Element("mujoco", model="piper_s0_s1")
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes", fusestatic="false")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    ET.SubElement(root, "size", njmax="1000", nconmax="300")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    contact = ET.SubElement(root, "contact")
    actuator = ET.SubElement(root, "actuator")
    bodies = {"world": world}
    pending = [j for j in joints if j.get("name") != "gripper"]
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in bodies]
        if not ready:
            raise ValueError("unsupported source hierarchy")
        for j in ready:
            name = j.get("name")
            child, parent = j.find("child").get("link"), j.find("parent").get("link")
            xyz, rot = origin(j)
            quat = rot.as_quat()[[3, 0, 1, 2]]
            body = ET.SubElement(bodies[parent], "body", name=child, pos=fmt(xyz), quat=fmt(quat))
            bodies[child] = body
            link = links[child]
            inertial = link.find("inertial")
            if inertial is not None:
                ipos, irot = origin(inertial)
                inertia = inertial.find("inertia")
                tensor = np.array(
                    [
                        [float(inertia.get(k)) for k in row]
                        for row in (
                            ("ixx", "ixy", "ixz"),
                            ("ixy", "iyy", "iyz"),
                            ("ixz", "iyz", "izz"),
                        )
                    ]
                )
                tensor = irot.as_matrix() @ tensor @ irot.as_matrix().T
                ET.SubElement(
                    body,
                    "inertial",
                    pos=fmt(ipos),
                    mass=inertial.find("mass").get("value"),
                    fullinertia=fmt(
                        [
                            tensor[0, 0],
                            tensor[1, 1],
                            tensor[2, 2],
                            tensor[0, 1],
                            tensor[0, 2],
                            tensor[1, 2],
                        ]
                    ),
                )
            collision = link.find("collision")
            if collision is not None:
                mesh = Path(collision.find("geometry/mesh").get("filename")).name
                if not (asset_dir / "meshes" / mesh).is_file():
                    raise ValueError(f"missing source STL: {mesh}")
                ET.SubElement(assets, "mesh", name=child, file=mesh)
                gpos, grot = origin(collision)
                ET.SubElement(
                    body,
                    "geom",
                    name=child + "_mesh",
                    type="mesh",
                    mesh=child,
                    pos=fmt(gpos),
                    quat=fmt(grot.as_quat()[[3, 0, 1, 2]]),
                    rgba="0.25 0.3 0.34 1",
                )
            if j.get("type") != "fixed":
                limit = j.find("limit")
                bounds = [float(limit.get("lower")), float(limit.get("upper"))]
                if name in JOINTS:
                    index = JOINTS.index(name)
                    bounds = [lo[index], hi[index]]
                ET.SubElement(
                    body,
                    "joint",
                    name=name,
                    type="hinge" if j.get("type") == "revolute" else "slide",
                    axis=j.find("axis").get("xyz"),
                    range=fmt(bounds),
                    limited="true",
                )
                if name in JOINTS:
                    force = float(limit.get("effort"))
                    ET.SubElement(
                        actuator,
                        "position",
                        name=name + "_servo",
                        joint=name,
                        kp="1",
                        kv="1",
                        forcerange=fmt([-force, force]),
                        forcelimited="true",
                    )
            if parent != "world":
                # Direct neighbours necessarily touch at their joint surfaces.
                ET.SubElement(contact, "exclude", body1=parent, body2=child)
            pending.remove(j)
    ET.SubElement(bodies["flange_link"], "site", name="flange", size="0.006", rgba="1 0.2 0.1 1")
    tendon = ET.SubElement(root, "tendon")
    fixed = ET.SubElement(tendon, "fixed", name="gripper_width")
    ET.SubElement(fixed, "joint", joint=FINGERS[0], coef="1")
    ET.SubElement(fixed, "joint", joint=FINGERS[1], coef="-1")
    equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "joint", joint1=FINGERS[1], joint2=FINGERS[0], polycoef="0 -1 0 0 0")
    ET.SubElement(
        actuator,
        "position",
        name="gripper_servo",
        tendon="gripper_width",
        kp="1",
        kv="1",
        forcerange="-10 10",
        forcelimited="true",
    )
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


if __name__ == "__main__":
    (ASSETS / "piper.xml").write_text(build_xml(), encoding="utf-8")
