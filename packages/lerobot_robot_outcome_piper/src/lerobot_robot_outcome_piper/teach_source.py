"""Receive-only official SDK feedback, shared by teach recording and diagnostics."""

import math
import gc
import time


def deny_transmission(comm, attempts):
    def blocked(message, *args, **kwargs):
        attempts.append(
            {"at_monotonic_s": time.monotonic(), "can_id": getattr(message, "arbitration_id", None)}
        )
        raise RuntimeError("receive-only teach observer blocked a CAN transmission")

    comm.send = blocked


def feedback_row(arm, receiver):
    began = time.monotonic()
    row = {"sampled_monotonic_s": began, "feedback_complete": False}
    with receiver.condition:
        row["received_ids"] = {hex(k): v for k, v in receiver.received.items()}
    if arm.has_comm_error():
        raise RuntimeError(f"CAN receive error: {arm.get_comm_error()}")
    snapshot_started = time.monotonic()
    try:
        feedback = receiver.snapshot()
    except RuntimeError as exc:
        row["feedback_error"] = str(exc)
        return row
    snapshot_finished = time.monotonic()
    status = feedback.status.msg
    joints = [float(q) for q in feedback.joints.msg]
    width = float(feedback.gripper.msg.value)
    grip_mode = feedback.gripper.msg.mode
    if len(joints) != 6 or not all(math.isfinite(v) for v in [*joints, width]):
        raise RuntimeError("non-finite or incomplete robot feedback")
    row.update(
        feedback_complete=True,
        joint_rad=joints,
        gripper_value=width,
        gripper_mode=grip_mode,
        gripper_m=width if grip_mode == "width" else None,
        received_monotonic_s=list(feedback.received_s),
        ctrl_mode=int(status.ctrl_mode),
        arm_status=int(status.arm_status),
        teach_status=int(status.teach_status),
        mode_feedback=int(status.mode_feedback),
        err_code=int(status.err_code),
    )
    row["feedback_age_s"] = time.monotonic() - min(feedback.received_s)
    row["status_age_s"] = time.monotonic() - feedback.received_s[3]
    row["sdk_timestamps_s"] = [
        getattr(obj, "timestamp", None)
        for obj in (feedback.joints, feedback.status, feedback.gripper)
    ]
    row["component_ages_s"] = [time.monotonic() - t for t in feedback.received_s]
    drivers_started = time.monotonic()
    row["drivers"] = [
        dict(
            joint=i,
            received_monotonic_s=stamp,
            enabled=None if state is None else bool(state.msg.foc_status.driver_enable_status),
            error=None if state is None else bool(state.msg.foc_status.driver_error_status),
        )
        for i, (state, stamp) in enumerate(receiver.driver_states(), 1)
    ]
    row["feedback_read_timing"] = dict(
        initial_s=snapshot_started - began,
        snapshot_s=snapshot_finished - snapshot_started,
        assembly_s=drivers_started - snapshot_finished,
        drivers_s=time.monotonic() - drivers_started,
        automatic_gc_enabled=gc.isenabled(),
    )
    return row


def teaching(row, max_age):
    # Official teach_status 1 means drag trajectory recording; not leader mode.
    return (
        row["feedback_complete"]
        and 0 <= row["status_age_s"] <= max_age
        and row["teach_status"] == 1
    )


class TeachSource:
    """One receive-only session. Construction/import performs no device I/O."""

    def __init__(self, config):
        self.config = config
        self.arm = None
        self.cameras = {}
        self.transmit_attempts = []

    def connect(self, *, require_teach=True):
        from .sdk import create_piper
        from .timing import FeedbackReceiver
        from .camera import make_timed_cameras

        self.arm = create_piper(self.config.can_interface, self.config.firmware)
        comm = self.arm.get_context().get_comm()
        if comm is None:
            comm = self.arm.get_context().init_comm()
        deny_transmission(comm, self.transmit_attempts)
        # Official parser and host timestamps are installed before receive starts.
        grip = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.receiver = FeedbackReceiver(self.arm, grip, time.monotonic)
        if self.config.capture_timing is not None:
            self.receiver.joint_max_skew_s = self.config.capture_timing.joint_max_skew_s
            self.receiver.snapshot_wait_s = min(
                self.config.feedback_timeout_s, self.config.capture_timing.joint_max_skew_s
            )
        self.arm.connect()
        if not self.receiver.wait_ready(2.0):
            raise RuntimeError("initial robot feedback incomplete")
        self.cameras = make_timed_cameras(self.config.cameras)
        for camera in self.cameras.values():
            camera.max_frame_age_s = self.config.capture_timing.camera_max_age_s
            camera.connect()
        if require_teach:
            self.feedback()
        else:
            self.feedback(require_teach=False)

    def feedback(self, *, require_teach=True):
        row = feedback_row(self.arm, self.receiver)
        row["sampled_monotonic_s"] = time.monotonic()
        check_feedback(row, self.config.feedback_timeout_s, require_teach=require_teach)
        if self.transmit_attempts:
            raise RuntimeError("receive-only session blocked a transmit attempt")
        return row

    def read(self):
        from dataclasses import asdict

        frames, metadata = {}, {}
        read_timing = {}
        for name, camera in self.cameras.items():
            read_started = time.monotonic()
            images, meta = camera.read_with_metadata(self.config.capture_timing.camera_max_age_s)
            read_timing[name] = dict(
                read_started_s=read_started,
                read_finished_s=time.monotonic(),
                diagnostics=dict(getattr(camera, "last_read_diagnostics", {})),
            )
            for kind, pixels in images.items():
                key = name if kind == "color" else f"{name}.depth"
                frames[key] = pixels
                metadata[key] = asdict(meta[kind])
        feedback_started = time.monotonic()
        row = self.feedback()
        row["sampling_read_timing"] = dict(
            cameras=read_timing,
            feedback_started_s=feedback_started,
            feedback_finished_s=time.monotonic(),
            feedback_stages=row.get("feedback_read_timing"),
        )
        row["camera"] = metadata
        return frames, row

    def disconnect(self):
        failures = []
        for resource in [*(c for c in self.cameras.values() if c.is_connected), self.arm]:
            if resource is not None:
                try:
                    resource.disconnect()
                except Exception as exc:
                    failures.append(str(exc))
        if failures:
            raise RuntimeError("teach receive cleanup failed: " + "; ".join(failures))


def check_feedback(row, timeout, *, require_teach=True):
    if not row["feedback_complete"]:
        raise RuntimeError("robot feedback incomplete")
    now = row["sampled_monotonic_s"]
    stamps = row["received_monotonic_s"]
    if len(stamps) != 5 or any(not math.isfinite(t) or not 0 <= now - t <= timeout for t in stamps):
        raise RuntimeError("robot feedback stale or invalid receive time")
    drivers = row["drivers"]
    if len(drivers) != 6 or any(
        d["received_monotonic_s"] is None
        or d["enabled"] is None
        or d["error"]
        or not 0 <= now - d["received_monotonic_s"] <= timeout
        for d in drivers
    ):
        raise RuntimeError("driver feedback missing, stale or faulted")
    if row["err_code"] or row["arm_status"] not in (0, 11):
        raise RuntimeError(
            f"controller fault: arm_status={row['arm_status']}, err_code={row['err_code']}"
        )
    allowed_mode = (
        (row["ctrl_mode"] == 2 and row["teach_status"] in (1, 2))
        if require_teach
        else (row["ctrl_mode"] in (0, 1, 2) and row["teach_status"] in (0, 1, 2))
    )
    if not allowed_mode:
        raise RuntimeError(
            f"unexpected teach mode: ctrl={row['ctrl_mode']}, teach={row['teach_status']}"
        )
    values = row["joint_rad"] + [row["gripper_m"]]
    if (
        len(values) != 7
        or row["gripper_mode"] != "width"
        or any(v is None or not math.isfinite(v) for v in values)
    ):
        raise RuntimeError("teach recording requires six joint radians and metric gripper width")
