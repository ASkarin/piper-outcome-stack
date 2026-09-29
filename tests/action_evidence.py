from lerobot_robot_outcome_piper.safety import ACTION_KEYS


def dispatched(values, sequence=1, time_s=1.0):
    values = dict(values)

    def command(name, target):
        return dict(
            name=name,
            target=target,
            result="sdk_returned",
            started_monotonic_s=time_s,
            ended_monotonic_s=time_s + 0.001,
        )

    return dict(
        observation_sequence=sequence,
        result="sdk_returned",
        values=values,
        commands=[
            command("move_j", [values[k] for k in ACTION_KEYS[:6]]),
            command("move_gripper_m", values[ACTION_KEYS[6]]),
        ],
    )
