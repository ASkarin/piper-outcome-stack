"""Operator Enter-confirmed A pose, using the existing Xbox pose processor."""

import math
import time
from copy import deepcopy

from lerobot_robot_outcome_piper.teleoperator import AXIS_KEYS, CONTROL_KEYS
from lerobot_robot_outcome_piper.teleop_control import TeleopState
from lerobot_robot_outcome_piper.safety import ACTION_KEYS
from lerobot_robot_outcome_piper.workflows import _processor

from .policy_execution import CONTROL_PERIOD_S
from .policy_motion import finish_hold


def confirm_work_pose(config, input_fn=input):
    if config.work_joint_rad is None or config.work_gripper_m is None:
        raise ValueError("A work pose is not configured")
    print(
        "A工作姿态（度）：" + str([round(math.degrees(q), 3) for q in config.work_joint_rad]),
        flush=True,
    )
    print(f"夹爪目标：{config.work_gripper_m * 1000:.2f} mm。", flush=True)
    print(
        "[准备] 请先松开LB，让摇杆和扳机完全回中。\n确认机械臂到A工作姿态的路径无遮挡。", flush=True
    )
    answer = input_fn(
        "按回车：使能、保持并自动移动到A工作姿态。\n到位后还需重新按住LB，才会开始模型控制。\n输入其他内容再回车：取消本次测试。\n> "
    )
    return answer == ""


class WorkPoseInput:
    def __init__(self, teleop, robot, control):
        self.teleop, self.robot, self.control = teleop, robot, control
        self.cancelled = False

    def poll(self):
        raw = self.teleop.get_action()
        if raw["emergency_stop"]:
            self.robot.request_emergency_stop("Xbox B during startup A")
            raise RuntimeError("Xbox B: startup A stopped")
        conflict = not raw["neutral"] or any(
            raw[k] for k in ("hold", "home", "work", "mode_switch", "translation_switch")
        )
        if conflict and not self.cancelled:
            self.cancelled = True
            self.control.request_hold("startup_A_operator_cancelled")
        return raw

    def camera_poll(self):
        raw = self.poll()
        # Enter authorizes only this pose segment. Camera waits must use that
        # same permission, after checking physical B/buttons for cancellation.
        return {**raw, "hold": not self.cancelled and self.control.state is TeleopState.POSE_MOVING}


def prepare_work_pose(
    robot, teleop, control, config, rows, *, clock=time.monotonic, sleep=time.sleep
):
    """Called only after terminal confirmation and operator-started enable."""
    from lerobot_robot_outcome_piper.robot import PiperState

    inputs = WorkPoseInput(teleop, robot, control)
    previous_poll = robot.camera_input_poll
    previous_phase = control.recording_phase
    robot.camera_input_poll = inputs.camera_poll
    control.recording_phase = "preparing"
    pipeline = _processor(robot.config, config)
    pipeline.steps[0].control = control
    stage = "settle"
    try:
        print(
            "[回A] 正在自动移动到A工作姿态。\n此阶段请保持LB松开，摇杆和扳机回中。\n按LB、推杆或其他动作键会取消回A并保持；B急停。",
            flush=True,
        )
        while True:
            started = clock()
            inputs.poll()
            observation = robot.get_observation()
            inputs.poll()
            if inputs.cancelled:
                finish_hold(robot, control, inputs, observation, rows, clock, sleep)
                return {"status": "cancelled_held", "hold_confirmed": control.hold_confirmed}
            raw = {
                **dict.fromkeys(AXIS_KEYS, 0.0),
                **dict.fromkeys(CONTROL_KEYS, False),
                "neutral": True,
                "hold": stage == "move",
                "work": stage == "select",
            }
            action = pipeline((raw, observation))
            inputs.poll()  # Recheck after planning, before dispatching its waypoint.
            if inputs.cancelled:
                finish_hold(robot, control, inputs, observation, rows, clock, sleep)
                return {"status": "cancelled_held", "hold_confirmed": control.hold_confirmed}
            if action.rejection_reason:
                raise ValueError("A preparation rejected: " + action.rejection_reason)
            action.generated_monotonic_s = clock()
            robot.send_action(action)
            rows.append(
                {
                    "phase": "startup_A",
                    "startup_stage": stage,
                    "input_telemetry": deepcopy(robot.last_observation_telemetry),
                    "sdk": deepcopy(robot.last_action_telemetry),
                }
            )
            if control.pose_event == "completed":
                finish_hold(robot, control, inputs, observation, rows, clock, sleep)
                if inputs.cancelled:
                    return {"status": "cancelled_held", "hold_confirmed": control.hold_confirmed}
                # The final poll verified physical LB released and neutral. End
                # synthetic pose hold so the next physical press is a fresh edge.
                control.recording_phase = previous_phase
                control.observe(False, True)
                print(
                    "[回A完成] A工作姿态已到位，并已确认保持。\n接下来等待策略就绪提示，再按住LB开始测试。",
                    flush=True,
                )
                return {
                    "status": "arrived_held",
                    "hold_confirmed": control.hold_confirmed,
                    "target_joints": list(config.work_joint_rad),
                    "target_gripper_m": config.work_gripper_m,
                    "observed_at_sequence_completion": {
                        k: float(observation[k]) for k in ACTION_KEYS
                    },
                }
            if stage == "settle" and control.hold_confirmed:
                stage = "select"
            elif stage == "select":
                if control.state is not TeleopState.POSE_READY:
                    raise ValueError("A pose request was not accepted")
                stage = "plan"
            elif (
                stage == "plan"
                and control.hold_confirmed
                and control.pose_sequence is not None
                and control.pose_sequence.planning_complete
            ):
                stage = "move"
            elif stage == "move" and control.state is not TeleopState.POSE_MOVING:
                raise ValueError("A pose execution cancelled before arrival")
            sleep(max(0, CONTROL_PERIOD_S - (clock() - started)))
    except (Exception, KeyboardInterrupt) as exc:
        if robot.state is PiperState.ACTIVE:
            robot.request_input_fault(exc)
        raise
    finally:
        robot.camera_input_poll = previous_poll
        control.recording_phase = previous_phase
