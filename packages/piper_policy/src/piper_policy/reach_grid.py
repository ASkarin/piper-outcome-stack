"""Fixed trilinear inference over neural predictions at the 27 training goals."""

import numpy as np

from .reach_bc import FEEDBACK_GAIN


class ReachEndpointGrid:
    def __init__(self, axes_m, endpoints_rad, fixed_gripper_m):
        self.axes = np.asarray(axes_m, dtype=float).copy()
        self.values = np.asarray(endpoints_rad, dtype=float).copy()
        self.gripper = float(fixed_gripper_m)
        if self.axes.shape != (3, 3) or self.values.shape != (3, 3, 3, 6):
            raise ValueError("reach grid requires three 3-node axes and 27 joint6 endpoints")
        if not np.isfinite(self.axes).all() or not np.isfinite(self.values).all():
            raise ValueError("reach grid must be finite")
        if not np.all(np.diff(self.axes, axis=1) > 0):
            raise ValueError("reach grid axes must be strictly increasing")
        if not np.isfinite(self.gripper) or self.gripper < 0:
            raise ValueError("reach grid gripper must be finite and nonnegative")

    def endpoint(self, goal):
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("reach grid requires finite goal XYZ3")
        if np.any(goal < self.axes[:, 0]) or np.any(goal > self.axes[:, -1]):
            raise ValueError("goal outside training grid; extrapolation is not supported")
        cells, fractions = [], []
        for axis, value in zip(self.axes, goal, strict=True):
            index = int(np.searchsorted(axis, value, side="right") - 1)
            if index == 2:  # The upper boundary belongs to the final closed cell.
                index = 1
            cells.append(index)
            fractions.append((value - axis[index]) / (axis[index + 1] - axis[index]))
        output = np.zeros(6)
        for x in (0, 1):
            for y in (0, 1):
                for z in (0, 1):
                    weight = 1.0
                    for bit, fraction in zip((x, y, z), fractions, strict=True):
                        weight *= fraction if bit else 1 - fraction
                    output += weight * self.values[cells[0] + x, cells[1] + y, cells[2] + z]
        return output

    def action(self, state, goal):
        state = np.asarray(state, dtype=float)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("reach grid requires finite physical state7")
        endpoint = self.endpoint(goal)
        return np.r_[state[:6] + FEEDBACK_GAIN * (endpoint - state[:6]), self.gripper]
