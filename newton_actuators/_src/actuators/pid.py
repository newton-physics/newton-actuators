# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

"""Stateful PID controller actuator."""

import math
from dataclasses import dataclass
from typing import Any

import warp as wp

from ..kernels import pid_controller_kernel, pid_integral_state_kernel
from .base import Actuator


class ActuatorPID(Actuator):
    """Stateful PID controller.

    Control law: τ = clamp(G·[constant + act + Kp·(target_pos - G·q) + Ki·∫e·dt + Kd·(target_vel - G·v)], ±max_force)

    Stateful: maintains integral term with anti-windup clamping.
    """

    @dataclass
    class State:
        """Integral state for PID actuators."""

        integral: wp.array = None  # Shape (N,)

    def is_stateful(self) -> bool:
        return True

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve arguments with defaults.

        Args:
            args (dict): User-provided arguments.

        Returns:
            dict: Arguments with defaults (kp=0, ki=0, kd=0, max_force=inf, integral_max=inf, gear=1, constant_force=0).
        """
        return {
            "kp": args.get("kp", 0.0),
            "ki": args.get("ki", 0.0),
            "kd": args.get("kd", 0.0),
            "max_force": args.get("max_force", math.inf),
            "integral_max": args.get("integral_max", math.inf),
            "gear": args.get("gear", 1.0),
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        input_indices: wp.array,
        output_indices: wp.array,
        kp: wp.array,
        ki: wp.array,
        kd: wp.array,
        max_force: wp.array,
        integral_max: wp.array,
        gear: wp.array,
        constant_force: wp.array = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_target_vel_attr: str = "joint_target_vel",
        control_input_attr: str = "joint_act",
        control_output_attr: str = "joint_f",
    ):
        """Initialize PID actuator.

        Args:
            input_indices (wp.array): DOF indices for reading state and targets. Shape (N,).
            output_indices (wp.array): DOF indices for writing output. Shape (N,).
            kp (wp.array): Proportional gains. Shape (N,).
            ki (wp.array): Integral gains. Shape (N,).
            kd (wp.array): Derivative gains. Shape (N,).
            max_force (wp.array): Force limits. Shape (N,).
            integral_max (wp.array): Anti-windup limits. Shape (N,).
            gear (wp.array): Gear ratios. Shape (N,).
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
            ("ki", ki),
            ("kd", kd),
            ("max_force", max_force),
            ("integral_max", integral_max),
            ("gear", gear),
        ]:
            if len(arr) != self.num_actuators:
                raise ValueError(f"{name} length ({len(arr)}) must match num_actuators ({self.num_actuators})")

        if constant_force is not None and len(constant_force) != self.num_actuators:
            raise ValueError(
                f"constant_force length ({len(constant_force)}) must match num_actuators ({self.num_actuators})"
            )

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_force = max_force
        self.integral_max = integral_max
        self.gear = gear
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
        current_state: "ActuatorPID.State",
        dt: float,
    ) -> None:
        """Compute PID control forces."""
        control_input = None
        if self.control_input_attr is not None:
            control_input = getattr(sim_control, self.control_input_attr, None)

        wp.launch(
            kernel=pid_controller_kernel,
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
                self.ki,
                self.kd,
                self.max_force,
                self.integral_max,
                self.gear,
                self.constant_force,
                dt,
                current_state.integral,
            ],
            outputs=[controller_output],
        )

    def _run_state_manager(
        self,
        sim_state: Any,
        sim_control: Any,
        current_state: "ActuatorPID.State",
        next_state: "ActuatorPID.State",
        dt: float,
    ) -> None:
        """Update integral state."""
        wp.launch(
            kernel=pid_integral_state_kernel,
            dim=self.num_actuators,
            inputs=[
                getattr(sim_state, self.state_pos_attr),
                getattr(sim_control, self.control_target_pos_attr),
                self.input_indices,
                self.input_indices,
                self.integral_max,
                self.gear,
                dt,
                current_state.integral,
            ],
            outputs=[next_state.integral],
        )

    def state(self) -> "ActuatorPID.State":
        """Return a new state with zero-initialized integral."""
        device = self.input_indices.device
        return ActuatorPID.State(
            integral=wp.zeros(self.num_actuators, dtype=wp.float32, device=device),
        )
