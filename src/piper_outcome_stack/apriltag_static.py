"""Camera-only AprilTag visibility capture; never connects a robot or CAN interface."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image


def make_detector():
    parameters = cv2.aruco.DetectorParameters()
    parameters.errorCorrectionRate = 0.0
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detections_for_image(image, pixel_format, detector):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("tag capture requires uint8 three-channel color")
    if pixel_format not in ("rgb8", "bgr8"):
        raise ValueError("unsupported color pixel format")
    conversion = cv2.COLOR_RGB2GRAY if pixel_format == "rgb8" else cv2.COLOR_BGR2GRAY
    corners, identifiers, _ = detector.detectMarkers(cv2.cvtColor(image, conversion))
    if identifiers is None:
        return []
    result = []
    for identifier, points in zip(identifiers[:, 0], corners, strict=True):
        points = points.reshape(4, 2)
        edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        result.append(
            dict(
                id=int(identifier),
                corners_pixels=points.tolist(),
                centre_pixels=points.mean(axis=0).tolist(),
                edge_lengths_pixels=edges.tolist(),
            )
        )
    return result


def summarize(rows, reference_id, tool_id):
    observed_ids = sorted({tag["id"] for row in rows for tag in row["detections"]})
    statistics = {}
    for identifier in sorted(set(observed_ids) | {reference_id, tool_id}):
        appearances = [tag for row in rows for tag in row["detections"] if tag["id"] == identifier]
        count = sum(any(tag["id"] == identifier for tag in row["detections"]) for row in rows)
        centres = np.asarray([tag["centre_pixels"] for tag in appearances])
        statistics[str(identifier)] = dict(
            frames_detected=count,
            total_frames=len(rows),
            centre_std_pixels=None
            if len(appearances) < 2
            else centres.std(axis=0, ddof=1).tolist(),
            centre_range_pixels=None if not appearances else np.ptp(centres, axis=0).tolist(),
            minimum_edge_pixels=None
            if not appearances
            else min(min(tag["edge_lengths_pixels"]) for tag in appearances),
        )
    together = sum(
        {reference_id, tool_id} <= {tag["id"] for tag in row["detections"]} for row in rows
    )
    return dict(
        frames=len(rows),
        tag_statistics=statistics,
        both_visible_frames=together,
        reference_id=reference_id,
        candidate_tool_id=tool_id,
        pose_estimated=False,
        rigid_mount_verified=False,
    )


def capture(camera, output, samples, hz, reference_id, tool_id):
    detector = make_detector()
    rows = []
    first_number = last_number = None
    first_time = last_time = None
    with (output / "frames.jsonl").open("x") as log:
        for index in range(samples):
            started = time.monotonic()
            images, telemetry = camera.read_with_metadata(0.2)
            meta = telemetry["color"]
            if last_number is not None and meta.frame_number <= last_number:
                raise RuntimeError("camera sample frame number did not advance")
            image = images["color"]
            detections = detections_for_image(image, meta.pixel_format, detector)
            row = dict(
                sample=index,
                acquired_monotonic_s=started,
                camera={name: asdict(value) for name, value in telemetry.items()},
                detections=detections,
                rgb_path=f"frame-{index:03d}.png",
            )
            rgb = image if meta.pixel_format == "rgb8" else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(output / row["rgb_path"])
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            rows.append(row)
            if first_number is None:
                first_number = meta.frame_number
                first_time = meta.device_timestamp_ms
            last_number = meta.frame_number
            last_time = meta.device_timestamp_ms
            time.sleep(max(0.0, 1 / hz - (time.monotonic() - started)))
    return dict(
        **summarize(rows, reference_id, tool_id),
        first_frame_number=first_number,
        last_frame_number=last_number,
        device_duration_ms=last_time - first_time,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-config", type=Path, required=True)
    parser.add_argument("--camera-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-id", type=int, required=True)
    parser.add_argument("--tool-id", type=int, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.samples <= 0 or not np.isfinite(args.sample_hz) or args.sample_hz <= 0:
        raise ValueError("sample count and rate must be positive")
    if args.reference_id == args.tool_id:
        raise ValueError("reference and tool tags must have different IDs")
    record = json.loads(args.record_config.read_text())
    values = dict(record["robot"]["cameras"][args.camera_key])
    if values.pop("type") != "intelrealsense":
        raise ValueError("this capture uses the configured RealSense backend")
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot_robot_outcome_piper.realsense import TimedRealSenseCamera

    config = RealSenseCameraConfig(**values)
    if not config.use_rgb or args.sample_hz > config.fps:
        raise ValueError("RGB and a supported sampling rate are required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="started",
        started_utc=datetime.now(timezone.utc).isoformat(),
        record_config=str(args.record_config),
        camera_key=args.camera_key,
        camera_config=values,
        opencv_version=cv2.__version__,
        family="tag25h9",
        error_correction_rate=0.0,
        camera_io=True,
        can_io=False,
        motor_commands_sent=0,
        streams_saved=["color"],
        metadata_saved="all configured streams",
        scope="static visibility and pixel variation only; no TCP calibration or policy execution",
    )
    camera = TimedRealSenseCamera(config)
    try:
        camera.connect()
        report.update(
            capture(
                camera, args.output, args.samples, args.sample_hz, args.reference_id, args.tool_id
            ),
            status="captured",
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if camera.is_connected:
                camera.disconnect()
        except Exception as exc:
            report.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            report["ended_utc"] = datetime.now(timezone.utc).isoformat()
            (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
