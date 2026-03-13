# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DC motor actuator with velocity-dependent torque saturation."""

import math
from typing import Any

import warp as wp

from ..kernels import pd_controller_kernel
from .base import Actuator


class ActuatorDCMotor(Actuator):
    """DC motor actuator with velocity-dependent torque saturation.

    Uses the same PD control law as ActuatorPD, but clips torques using the DC motor
    torque-speed characteristic instead of a fixed box limit:

        τ_max(v) = clamp(τ_sat·(1 - v/v_max),  0,  effort_limit)
        τ_min(v) = clamp(τ_sat·(-1 - v/v_max), -effort_limit, 0)
        τ_applied = clamp(τ_computed, τ_min(v), τ_max(v))

    At zero velocity the motor can produce up to ±τ_sat (capped by effort_limit).
    As velocity approaches v_max, available torque in the direction of motion drops to zero.
    Beyond v_max, no torque can be produced in the direction of motion (back-EMF).

    Stateless: no internal memory.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve arguments with defaults.

        Args:
            args (dict): User-provided arguments.

        Returns:
            dict: Arguments with defaults.

        Raises:
            ValueError: If 'velocity_limit' not provided.
        """
        if "velocity_limit" not in args:
            raise ValueError("ActuatorDCMotor requires 'velocity_limit' argument")
        return {
            "kp": args.get("kp", 0.0),
            "kd": args.get("kd", 0.0),
            "max_force": args.get("max_force", math.inf),
            "saturation_effort": args.get("saturation_effort", math.inf),
            "velocity_limit": args["velocity_limit"],
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        input_indices: wp.array,
        output_indices: wp.array,
        kp: wp.array,
        kd: wp.array,
        max_force: wp.array,
        saturation_effort: wp.array,
        velocity_limit: wp.array,
        constant_force: wp.array = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_target_vel_attr: str = "joint_target_vel",
        control_input_attr: str = "joint_act",
        control_output_attr: str = "joint_f",
    ):
        """Initialize DC motor actuator.

        Args:
            input_indices (wp.array): DOF indices for reading state and targets. Shape (N,).
            output_indices (wp.array): DOF indices for writing output. Shape (N,).
            kp (wp.array): Proportional gains. Shape (N,).
            kd (wp.array): Derivative gains. Shape (N,).
            max_force (wp.array): Absolute effort limits (continuous-rated). Shape (N,).
            saturation_effort (wp.array): Peak motor torque at stall. Shape (N,).
            velocity_limit (wp.array): Maximum joint velocity for torque-speed curve. Shape (N,).
            constant_force (wp.array, optional): Constant offsets. Shape (N,). None to skip.
            state_pos_attr (str): Attribute on sim_state for positions.
            state_vel_attr (str): Attribute on sim_state for velocities.
            control_target_pos_attr (str): Attribute on sim_control for target positions.
            control_target_vel_attr (str): Attribute on sim_control for target velocities.
            control_input_attr (str): Attribute on sim_control for control input. None to skip.
            control_output_attr (str): Attribute on sim_control for output forces.
        """
        super().__init__(input_indices, output_indices, control_output_attr)

        for name, arr in [
            ("kp", kp),
            ("kd", kd),
            ("max_force", max_force),
            ("saturation_effort", saturation_effort),
            ("velocity_limit", velocity_limit),
        ]:
            if len(arr) != self.num_actuators:
                raise ValueError(f"{name} length ({len(arr)}) must match num_actuators ({self.num_actuators})")

        if constant_force is not None and len(constant_force) != self.num_actuators:
            raise ValueError(
                f"constant_force length ({len(constant_force)}) must match num_actuators ({self.num_actuators})"
            )

        self.kp = kp
        self.kd = kd
        self.max_force = max_force
        self.saturation_effort = saturation_effort
        self.velocity_limit = velocity_limit
        self.constant_force = constant_force

        self.state_pos_attr = state_pos_attr
        self.state_vel_attr = state_vel_attr
        self.control_target_pos_attr = control_target_pos_attr
        self.control_target_vel_attr = control_target_vel_attr
        self.control_input_attr = control_input_attr

    def _run_controller(
        self,
        sim_state: Any,
        sim_control: Any,
        controller_output: wp.array,
        output_indices: wp.array,
        current_state: Any,
        dt: float,
    ) -> None:
        """Compute DC motor PD control forces with velocity-dependent saturation."""
        control_input = None
        if self.control_input_attr is not None:
            control_input = getattr(sim_control, self.control_input_attr, None)

        wp.launch(
            kernel=pd_controller_kernel,
            dim=self.num_actuators,
            inputs=[
                getattr(sim_state, self.state_pos_attr),
                getattr(sim_state, self.state_vel_attr),
                getattr(sim_control, self.control_target_pos_attr),
                getattr(sim_control, self.control_target_vel_attr),
                control_input,
                self.input_indices,
                self.input_indices,
                output_indices,
                self.kp,
                self.kd,
                self.max_force,
                self.constant_force,
                self.saturation_effort,
                self.velocity_limit,
                None,  # lookup_angles (remotized)
                None,  # lookup_torques (remotized)
                0,  # lookup_size (remotized)
            ],
            outputs=[controller_output],
        )
