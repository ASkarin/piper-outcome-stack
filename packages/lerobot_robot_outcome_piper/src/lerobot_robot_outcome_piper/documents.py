"""Strict JSON loading for runtime configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import OutcomePiperValidationError


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomePiperValidationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OutcomePiperValidationError(f"expected JSON object in {path}")
    return value
