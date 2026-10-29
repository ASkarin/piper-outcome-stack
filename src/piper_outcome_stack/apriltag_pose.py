"""Offline square-tag pose diagnostics; retain both planar solutions and all frames."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def square_points(side_m):
    if not np.isfinite(side_m) or side_m <= 0:
        raise ValueError("tag side must be an explicit positive metric length")
    h = side_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def camera_matrix(metadata):
    intr = metadata["intrinsics"]
    if metadata["rotation"] is not None or metadata["alignment"] != "native":
        raise ValueError("this diagnostic requires unrotated native color coordinates")
    # RealSense inverse Brown coefficients cannot be passed as forward Brown coefficients.
    if any(float(v) != 0 for v in intr["coeffs"]):
        raise ValueError("nonzero distortion requires its calibrated projection model")
    matrix = np.array(
        [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], float
    )
    if not np.isfinite(matrix).all() or intr["fx"] <= 0 or intr["fy"] <= 0:
        raise ValueError("invalid native intrinsics")
    return matrix


def pose_solutions(corners, side_m, matrix):
    points = square_points(side_m)
    corners = np.ascontiguousarray(corners, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("tag pose needs four finite ordered image corners")
    count, rotations, translations, _ = cv2.solvePnPGeneric(
        points, corners, matrix, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if count != 2:
        raise ValueError(f"expected two IPPE square solutions; received {count}")
    result = []
    for index, (rvec, tvec) in enumerate(zip(rotations, translations, strict=True)):
        rotation = cv2.Rodrigues(rvec)[0]
        translation = np.asarray(tvec).reshape(3)
        depths = (points @ rotation.T + translation)[:, 2]
        projected = cv2.projectPoints(points, rvec, tvec, matrix, np.zeros(5))[0].reshape(4, 2)
        rms = float(np.sqrt(np.square(projected - corners).sum(axis=1).mean()))
        if not np.isfinite([*rotation.ravel(), *translation, rms]).all():
            raise ValueError("nonfinite PnP output")
        result.append(
            dict(
                solver_index=index,
                rotation_matrix=rotation.tolist(),
                translation_camera_m=translation.tolist(),
                corner_reprojection_rms_pixels=rms,
                positive_corner_depths=bool((depths > 0).all()),
            )
        )
    result.sort(key=lambda row: row["corner_reprojection_rms_pixels"])
    return result


def relative_pose(reference, tool):
    r_ref, r_tool = (np.asarray(v["rotation_matrix"]) for v in (reference, tool))
    t_ref, t_tool = (np.asarray(v["translation_camera_m"]) for v in (reference, tool))
    return dict(
        rotation_matrix=(r_ref.T @ r_tool).tolist(),
        translation_reference_m=(r_ref.T @ (t_tool - t_ref)).tolist(),
        displacement_camera_m=(t_tool - t_ref).tolist(),
    )


def scalar_stats(values):
    a = np.asarray(values, float)
    return dict(
        minimum=float(a.min()),
        median=float(np.median(a)),
        maximum=float(a.max()),
        mean=float(a.mean()),
    )


def translation_stats(values):
    a = np.asarray(values, float)
    residual = np.linalg.norm(a - a.mean(axis=0), axis=1)
    return dict(
        mean_m=a.mean(axis=0).tolist(),
        std_mm=(a.std(axis=0, ddof=1) * 1000).tolist(),
        peak_to_peak_mm=(np.ptp(a, axis=0) * 1000).tolist(),
        radial_deviation_p95_mm=float(np.quantile(residual, 0.95) * 1000),
        radial_deviation_max_mm=float(residual.max() * 1000),
    )


def rotation_stats(matrices):
    rotations = Rotation.from_matrix(np.asarray(matrices))
    mean = rotations.mean()
    deviations = np.rad2deg((mean.inv() * rotations).magnitude())
    changes = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude())
    return dict(
        mean_rotation_matrix=mean.as_matrix().tolist(),
        deviation_deg=scalar_stats(deviations),
        adjacent_change_deg=scalar_stats(changes),
    )


def analyze(rows, reference_id, tool_id, reference_side_m, tool_side_m):
    frames = []
    for row in rows:
        tags = {tag["id"]: tag for tag in row["detections"]}
        if len(tags) != len(row["detections"]) or not {reference_id, tool_id} <= tags.keys():
            raise ValueError(f"sample {row['sample']} lacks unique requested tag detections")
        matrix = camera_matrix(row["camera"]["color"])
        solutions = {
            str(identity): pose_solutions(tags[identity]["corners_pixels"], side, matrix)
            for identity, side in [(reference_id, reference_side_m), (tool_id, tool_side_m)]
        }
        # Preserve all hypotheses. The following summaries use minimum reprojection error,
        # which is a declared convention, not evidence that a physical branch is correct.
        if not all(s["positive_corner_depths"] for group in solutions.values() for s in group):
            raise ValueError(
                f"sample {row['sample']} has an invalid-depth pose; no silent exclusion"
            )
        combinations = [
            dict(
                reference_rank=i,
                tool_rank=j,
                **relative_pose(solutions[str(reference_id)][i], solutions[str(tool_id)][j]),
            )
            for i in range(2)
            for j in range(2)
        ]
        translations = np.array([c["translation_reference_m"] for c in combinations])
        spread = (
            np.linalg.norm(translations[:, None, :] - translations[None, :, :], axis=2).max() * 1000
        )
        frames.append(
            dict(
                sample=row["sample"],
                frame_number=row["camera"]["color"]["frame_number"],
                solutions=solutions,
                relative_hypotheses=combinations,
                relative_translation_hypothesis_diameter_mm=float(spread),
            )
        )
    if len(frames) < 2:
        raise ValueError("repeatability needs at least two frames")
    summary = dict(
        frames=len(frames),
        selection="minimum reprojection RMS independently per tag; no smoothing or branch deletion",
        tags={},
    )
    for identity in (reference_id, tool_id):
        groups = [f["solutions"][str(identity)] for f in frames]
        best = [g[0] for g in groups]
        separations = [
            np.rad2deg(
                Rotation.from_matrix(
                    np.asarray(g[0]["rotation_matrix"]).T @ np.asarray(g[1]["rotation_matrix"])
                ).magnitude()
            )
            for g in groups
        ]
        summary["tags"][str(identity)] = dict(
            position_camera=translation_stats([v["translation_camera_m"] for v in best]),
            rotation_camera=rotation_stats([v["rotation_matrix"] for v in best]),
            best_reprojection_rms_pixels=scalar_stats(
                [g[0]["corner_reprojection_rms_pixels"] for g in groups]
            ),
            second_minus_best_reprojection_rms_pixels=scalar_stats(
                [
                    g[1]["corner_reprojection_rms_pixels"] - g[0]["corner_reprojection_rms_pixels"]
                    for g in groups
                ]
            ),
            alternative_rotation_separation_deg=scalar_stats(separations),
        )
    selected = [f["relative_hypotheses"][0] for f in frames]
    summary["selected_relative"] = dict(
        position_reference=translation_stats([v["translation_reference_m"] for v in selected]),
        displacement_camera=translation_stats([v["displacement_camera_m"] for v in selected]),
        rotation_reference=rotation_stats([v["rotation_matrix"] for v in selected]),
        tag_centre_distance_mm=scalar_stats(
            [np.linalg.norm(v["displacement_camera_m"]) * 1000 for v in selected]
        ),
    )
    summary["relative_translation_hypothesis_diameter_mm"] = scalar_stats(
        [f["relative_translation_hypothesis_diameter_mm"] for f in frames]
    )
    return dict(summary=summary, frames=frames)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--reference-id", type=int, required=True)
    parser.add_argument("--tool-id", type=int, required=True)
    parser.add_argument("--reference-side-m", type=float, required=True)
    parser.add_argument("--tool-side-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in args.frames.read_text().splitlines()]
    result = analyze(rows, args.reference_id, args.tool_id, args.reference_side_m, args.tool_side_m)
    result.update(
        source_frames=str(args.frames.resolve()),
        reference_id=args.reference_id,
        tool_id=args.tool_id,
        reference_side_m=args.reference_side_m,
        tool_side_m=args.tool_side_m,
        opencv_version=cv2.__version__,
        method="SOLVEPNP_IPPE_SQUARE",
        coordinate_convention="T_camera_tag; T_reference_tool = inverse(T_camera_reference) * T_camera_tool",
        reprojection_definition="RMS Euclidean image distance over four corners",
        hardware_io=False,
        can_io=False,
        motor_commands_sent=0,
        tcp_calibrated=False,
        absolute_accuracy_verified=False,
        analysis_wall_s=time.monotonic() - started,
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    (args.output / "poses.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
