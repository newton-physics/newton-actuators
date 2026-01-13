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

"""Stateless PD controller actuator."""

import math
from typing import Any

import warp as wp

from ..kernels import pd_controller_kernel
from .base import Actuator


class ActuatorPD(Actuator):
    """Stateless PD controller.

    Control law: τ = clamp(constant + gear*act + Kp*e_pos + Kd*e_vel, ±max_force)

    Gains (Kp, Kd) operate in joint space. Gear scales feedforward input independently.
    Stateless: no internal memory, computes torques directly from current state.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve arguments with defaults.

        Args:
            args (dict): User-provided arguments.

        Returns:
            dict: Arguments with defaults (kp=0, kd=0, max_force=inf, gear=1, constant_force=0).
        """
        return {
            "kp": args.get("kp", 0.0),
            "kd": args.get("kd", 0.0),
            "max_force": args.get("max_force", math.inf),
            "gear": args.get("gear", 1.0),
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        input_indices: wp.array,
        output_indices: wp.array,
        kp: wp.array,
        kd: wp.array,
        max_force: wp.array,
        gear: wp.array,
        constant_force: wp.array,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_target_vel_attr: str = "joint_target_vel",
        control_input_attr: str = "joint_act",
        control_output_attr: str = "joint_f",
    ):
        """Initialize PD actuator.

        Args:
            input_indices (wp.array): DOF indices for reading state and targets. Shape (N,).
            output_indices (wp.array): DOF indices for writing output. Shape (N,).
            kp (wp.array): Proportional gains. Shape (N,).
            kd (wp.array): Derivative gains. Shape (N,).
            max_force (wp.array): Force limits. Shape (N,).
            gear (wp.array): Gear ratios. Shape (N,).
            constant_force (wp.array): Constant offsets. Shape (N,).
            state_pos_attr (str): Attribute on sim_state for positions.
            state_vel_attr (str): Attribute on sim_state for velocities.
            control_target_pos_attr (str): Attribute on sim_control for target positions.
            control_target_vel_attr (str): Attribute on sim_control for target velocities.
            control_input_attr (str): Attribute on sim_control for control input.
            control_output_attr (str): Attribute on sim_control for output forces.
        """
        super().__init__(input_indices, output_indices, control_output_attr)

        for name, arr in [("kp", kp), ("kd", kd), ("max_force", max_force), ("gear", gear), ("constant_force", constant_force)]:
            if len(arr) != self.num_actuators:
                raise ValueError(f"{name} length ({len(arr)}) must match num_actuators ({self.num_actuators})")

        self.kp = kp
        self.kd = kd
        self.max_force = max_force
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
        current_state: Any,
        dt: float,
    ) -> None:
        """Compute PD control forces."""
        wp.launch(
            kernel=pd_controller_kernel,
            dim=self.num_actuators,
            inputs=[
                getattr(sim_state, self.state_pos_attr),
                getattr(sim_state, self.state_vel_attr),
                getattr(sim_control, self.control_target_pos_attr),
                getattr(sim_control, self.control_target_vel_attr),
                getattr(sim_control, self.control_input_attr),
                self.input_indices,
                self.input_indices,
                output_indices,
                self.kp,
                self.kd,
                self.max_force,
                self.gear,
                self.constant_force,
            ],
            outputs=[controller_output],
        )
