# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from typing import Any

import warp as wp

from ..kernels import pid_force_kernel, pid_integral_state_kernel
from .base import Controller


class PIDController(Controller):
    """Stateful PID controller.

    Force law: f = constant + act + Kp*e + Ki*∫e·dt + Kd*de

    Maintains an integral term with anti-windup clamping.
    Produces raw (unclamped) forces — pair with ``Clamp`` for force limiting.
    """

    @dataclass
    class State:
        """Integral state for PID controller."""

        integral: wp.array = None  # Shape (N,)

        def reset(self) -> None:
            self.integral.zero_()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "kp": args.get("kp", 0.0),
            "ki": args.get("ki", 0.0),
            "kd": args.get("kd", 0.0),
            "integral_max": args.get("integral_max", math.inf),
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        kp: wp.array,
        ki: wp.array,
        kd: wp.array,
        integral_max: wp.array,
        constant_force: wp.array | None = None,
    ):
        """Initialize PID controller.

        Args:
            kp: Proportional gains. Shape (N,).
            ki: Integral gains. Shape (N,).
            kd: Derivative gains. Shape (N,).
            integral_max: Anti-windup limits. Shape (N,).
            constant_force: Constant force offsets. Shape (N,). None to skip.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.constant_force = constant_force

    def is_stateful(self) -> bool:
        return True

    def state(self, num_actuators: int, device: wp.Device) -> "PIDController.State":
        return PIDController.State(
            integral=wp.zeros(num_actuators, dtype=wp.float32, device=device),
        )

    def compute(
        self,
        positions: wp.array,
        velocities: wp.array,
        target_pos: wp.array,
        target_vel: wp.array,
        act_input: wp.array | None,
        input_indices: wp.array,
        target_indices: wp.array,
        forces: wp.array,
        force_indices: wp.array,
        num_actuators: int,
        state: "PIDController.State",
        dt: float,
    ) -> None:
        wp.launch(
            kernel=pid_force_kernel,
            dim=num_actuators,
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                act_input,
                input_indices,
                target_indices,
                force_indices,
                self.kp,
                self.ki,
                self.kd,
                self.integral_max,
                self.constant_force,
                dt,
                state.integral,
            ],
            outputs=[forces],
        )

    def update_state(
        self,
        positions: wp.array,
        velocities: wp.array,
        target_pos: wp.array,
        target_vel: wp.array,
        input_indices: wp.array,
        target_indices: wp.array,
        num_actuators: int,
        current_state: "PIDController.State",
        next_state: "PIDController.State",
        dt: float,
    ) -> None:
        wp.launch(
            kernel=pid_integral_state_kernel,
            dim=num_actuators,
            inputs=[
                positions,
                target_pos,
                input_indices,
                target_indices,
                self.integral_max,
                dt,
                current_state.integral,
            ],
            outputs=[next_state.integral],
        )
