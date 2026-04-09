# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import warp as wp

from ..kernels import dc_motor_clamp_kernel
from .base import Dynamic


class DCMotorSaturation(Dynamic):
    """DC motor velocity-dependent torque saturation.

    Clips controller output using the torque–speed characteristic:
        τ_max(v) = clamp(τ_sat*(1 - v/v_max),  0,  effort_limit)
        τ_min(v) = clamp(τ_sat*(-1 - v/v_max), -effort_limit, 0)

    At zero velocity the motor can produce up to ±τ_sat (capped by
    effort_limit). As velocity approaches v_max, available torque in
    the direction of motion drops to zero.

    This is a post-controller dynamic.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "velocity_limit" not in args:
            raise ValueError("DCMotorSaturation requires 'velocity_limit' argument")
        return {
            "saturation_effort": args.get("saturation_effort", math.inf),
            "velocity_limit": args["velocity_limit"],
            "max_force": args.get("max_force", math.inf),
        }

    def __init__(
        self,
        saturation_effort: wp.array,
        velocity_limit: wp.array,
        max_force: wp.array,
    ):
        """Initialize DC motor saturation dynamic.

        Args:
            saturation_effort: Peak motor torque at stall. Shape (N,).
            velocity_limit: Maximum joint velocity for torque-speed curve. Shape (N,).
            max_force: Absolute effort limits (continuous-rated). Shape (N,).
        """
        self.saturation_effort = saturation_effort
        self.velocity_limit = velocity_limit
        self.max_force = max_force

    def modify_forces(
        self,
        forces: wp.array,
        positions: wp.array,
        velocities: wp.array,
        input_indices: wp.array,
        num_actuators: int,
    ) -> None:
        wp.launch(
            kernel=dc_motor_clamp_kernel,
            dim=num_actuators,
            inputs=[
                velocities,
                input_indices,
                self.saturation_effort,
                self.velocity_limit,
                self.max_force,
            ],
            outputs=[forces],
        )
