"""Short receive-only teach-mode/RGBD diagnostic; never a training Dataset."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time


from lerobot_robot_outcome_piper.teach_source import deny_transmission, feedback_row, teaching


def collect(arm, receiver, camera, output, attempts, *, seconds=10.0, hz=10.0, wait_seconds=60.0):
    import numpy as np
    from PIL import Image

    output = Path(output)
    started = time.monotonic()
    capture_started = None
    count = 0
    states = set()
    with (output / "samples.jsonl").open("x") as log:
        while True:
            tick = time.monotonic()
            if capture_started is not None and tick - capture_started >= seconds:
                break
            if capture_started is None and tick - started >= wait_seconds:
                raise TimeoutError(
                    "teach_status=1 was not confirmed; sampling ended without changing mode"
                )
            frames, metadata = camera.read_with_metadata(0.2)
            row = feedback_row(arm, receiver)
            if attempts:
                raise RuntimeError("unexpected transmit attempt was blocked")
            if row["feedback_complete"]:
                states.add((row["ctrl_mode"], row["arm_status"], row["teach_status"]))
                if (
                    row["err_code"] != 0
                    or row["arm_status"] not in (0, 11)
                    or any(d["error"] for d in row["drivers"])
                ):
                    raise RuntimeError(f"controller status requires operator inspection: {row}")
            if capture_started is None and teaching(row, 0.2):
                capture_started = time.monotonic()
                print(
                    f"[记录中] 已确认示教状态，记录 {seconds:g} 秒。请扶住臂体，不测试松手悬停。",
                    flush=True,
                )
            row["camera"] = {key: asdict(value) for key, value in metadata.items()}
            row["phase"] = "waiting_for_teach" if capture_started is None else "teach_recording"
            if capture_started is not None:
                if (
                    row["feedback_complete"]
                    and 0 <= row["status_age_s"] <= 0.2
                    and row["teach_status"] == 2
                ):
                    return {
                        "frames": count,
                        "controller_states": sorted(states),
                        "duration_s": time.monotonic() - capture_started,
                        "end_reason": "teach_stopped",
                    }
                if not teaching(row, 0.2):
                    raise RuntimeError(f"teaching feedback missing/stale or mode changed: {row}")
                name = f"frame-{count:04d}"
                Image.fromarray(frames["color"]).save(output / f"{name}.png", compress_level=0)
                with (output / f"{name}.npz").open("xb") as stream:
                    np.savez(stream, depth=frames["depth"])
                row["frame_index"] = count
                row["rgb_path"], row["depth_path"] = name + ".png", name + ".npz"
                count += 1
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            time.sleep(max(0.0, 1 / hz - (time.monotonic() - tick)))
    return {
        "frames": count,
        "controller_states": sorted(states),
        "duration_s": time.monotonic() - capture_started if capture_started is not None else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    from lerobot_robot_outcome_piper.sdk import create_piper
    from lerobot_robot_outcome_piper.timing import FeedbackReceiver
    from lerobot_robot_outcome_piper.realsense import TimedRealSenseCamera
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    report = {
        "scope": "receive-only manual-teach/RGBD feasibility; no action labels or training data",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "transmit_attempts": [],
        "motor_commands_sent": False,
    }
    arm = None
    camera = None
    counts = Counter()
    count_lock = threading.Lock()
    try:
        arm = create_piper(cfg["can_interface"], cfg["firmware"])
        comm = arm.get_context().get_comm()
        if comm is None:
            comm = arm.get_context().init_comm()
        deny_transmission(comm, report["transmit_attempts"])
        arm.connect()
        grip = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        receiver = FeedbackReceiver(arm, grip, time.monotonic)
        original = comm.get_callback()

        def count_packet(packet):
            with count_lock:
                counts[packet.arbitration_id] += 1
            original(packet)

        comm.set_callback(count_packet)
        camera_values = dict(cfg["camera"])
        camera_values.pop("type", None)
        camera = TimedRealSenseCamera(RealSenseCameraConfig(**camera_values))
        print("[准备中] 连接接收端并暖机相机，请先不要切换示教。", flush=True)
        camera.connect()
        if not receiver.wait_ready(2.0):
            raise RuntimeError("initial robot feedback is incomplete; do not start teaching")
        initial = feedback_row(arm, receiver)
        report["initial_feedback"] = initial
        if (
            not initial["feedback_complete"]
            or initial["feedback_age_s"] > 0.2
            or initial["err_code"] != 0
            or initial["arm_status"] not in (0, 11)
            or any(d["error"] for d in initial["drivers"])
        ):
            raise RuntimeError(f"initial feedback requires inspection: {initial}")
        if teaching(initial, 0.2):
            print("[准备完成] 已在示教模式，请不要再次按按钮；即将记录。", flush=True)
        else:
            print("[准备完成] 现在可单击示教按钮；绿灯常亮后缓慢、小幅拖动。", flush=True)
        print("程序只接收数据，不控制机械臂，也不响应Xbox停止键；不要双击回放。", flush=True)
        report.update(collect(arm, receiver, camera, args.output, report["transmit_attempts"]))
        report["status"] = "observed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for name, resource in [("camera", camera), ("SDK", arm)]:
            if resource is not None:
                try:
                    if name != "camera" or resource.is_connected:
                        resource.disconnect()
                except Exception as exc:
                    report.setdefault("cleanup_errors", []).append(
                        f"{name}: {type(exc).__name__}: {exc}"
                    )
                    report["status"] = "failed"
        with count_lock:
            report["can_id_counts"] = {hex(k): v for k, v in counts.items()}
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            "[采样结束] 程序没有改变机械臂模式。若绿灯仍常亮，请扶稳并回到低位停放姿态，再单击结束示教；灯已灭则不要再按。",
            flush=True,
        )
        print(
            "不要双击回放；暂不切回CAN控制。报告：" + str(args.output / "summary.json"), flush=True
        )
    return 0 if report["status"] == "observed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
