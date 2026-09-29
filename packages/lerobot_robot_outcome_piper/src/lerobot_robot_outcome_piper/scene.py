"""Explicit physical scene provenance; no automatic reuse of installation data."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneContext:
    scene_id: str
    base_installation: str
    camera_view: str
    work_area_notes: str
    camera_to_base_calibration: str | None = None

    def __post_init__(self):
        for name in ("scene_id", "base_installation", "camera_view", "work_area_notes"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"scene {name} must describe the current installation")
