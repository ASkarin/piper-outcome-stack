"""Save a closed episode via official Dataset APIs outside the live control process."""

import json
from pathlib import Path
import tempfile
import time
import zipfile


DEPTH_COMPRESSION_LEVEL = 1
STAGE_PREFIX = "PIPER_SAVE_STAGE: "


def _stage(message):
    print(STAGE_PREFIX + message, flush=True)


def _same_array(original, restored):
    import numpy as np

    return (
        original.dtype == restored.dtype
        and original.shape == restored.shape
        and np.array_equal(original, restored, equal_nan=True)
    )


def compress_depth_file(path, kind):
    """Replace one temporary depth file only after lossless read-back verification."""
    import numpy as np
    from PIL import Image

    path = Path(path)
    before = path.stat().st_size
    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False
    ) as output:
        candidate = Path(output.name)
        if kind == "npz":
            with np.load(path, allow_pickle=False) as source:
                arrays = {key: source[key] for key in source.files}
            # Standard NPZ layout, with an explicit fast Deflate level.
            with zipfile.ZipFile(
                output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=DEPTH_COMPRESSION_LEVEL
            ) as archive:
                for key, value in arrays.items():
                    with archive.open(key + ".npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, value, allow_pickle=False)
        elif kind == "tiff":
            with Image.open(path) as source:
                original = np.array(source)
                mode = source.mode
                # libtiff TIFFTAG_ZIPQUALITY sets Deflate effort, not pixel precision.
                source.save(
                    output,
                    format="TIFF",
                    compression="tiff_deflate",
                    tiffinfo={65557: DEPTH_COMPRESSION_LEVEL},
                )
        else:
            raise ValueError(f"unknown depth format: {kind}")
    if kind == "npz":
        with np.load(candidate, allow_pickle=False) as restored:
            valid = set(restored.files) == set(arrays) and all(
                _same_array(value, restored[key]) for key, value in arrays.items()
            )
    else:
        with Image.open(candidate) as restored:
            valid = restored.mode == mode and _same_array(original, np.array(restored))
    if not valid:
        raise RuntimeError(
            f"lossless compression verification failed: {path}; candidate={candidate}"
        )
    after = candidate.stat().st_size
    candidate.replace(path)
    return before, after


def compress_episode_depth(episode, features, raw_paths, report_path):
    """Run sequentially in offline conversion, before TIFF bytes enter Parquet."""
    started = time.monotonic()
    report = {
        "status": "running",
        "compression_level": DEPTH_COMPRESSION_LEVEL,
        "npz": {"files": 0, "before_bytes": 0, "after_bytes": 0},
        "tiff": {"files": 0, "before_bytes": 0, "after_bytes": 0},
    }
    current = None
    try:
        paths = [(Path(p), "npz") for p in raw_paths]
        for key, feature in features.items():
            if feature["dtype"] == "image" and (feature.get("info") or {}).get("is_depth_map"):
                paths.extend((Path(p), "tiff") for p in episode[key])
        previous_kind = None
        for current, kind in paths:
            if kind != previous_kind:
                _stage(f"正在无损压缩并校验深度{kind.upper()}，请等待。")
                previous_kind = kind
            before, after = compress_depth_file(current, kind)
            report[kind]["files"] += 1
            report[kind]["before_bytes"] += before
            report[kind]["after_bytes"] += after
        report["status"] = "verified"
    except Exception as exc:
        report.update(status="failed", path=str(current), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["duration_s"] = time.monotonic() - started
        Path(report_path).write_text(json.dumps(report, indent=2))
