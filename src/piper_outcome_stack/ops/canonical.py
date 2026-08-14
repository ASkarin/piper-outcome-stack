"""JSON loading and the canonical preregistration digest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .errors import ValidationError

HASH_PREFIX = "sha256:"


def sha256_text_file(path: str | Path) -> str:
    """Hash UTF-8 text after normalizing checkout-specific line endings."""

    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot read text {path}: {exc}") from exc
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return HASH_PREFIX + hashlib.sha256(canonical).hexdigest()


def load_json(path: str | Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc
