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

"""PD controller with input delay."""

import math
from dataclasses import dataclass
from typing import Any

import warp as wp

from ..kernels import delay_buffer_state_kernel, pd_controller_kernel
from .base import Actuator


class ActuatorDelayedPD(Actuator):
    """PD controller with input delay.

    Control law: τ = clamp(constant + gear*act_delayed + Kp*e_pos_delayed + Kd*e_vel_delayed, ±max_force)

    Gains (Kp, Kd) operate in joint space. Gear scales feedforward input independently.
    Stateful: delays targets by N timesteps using circular buffer to model actuator lag.
    """

    SCALAR_PARAMS = {"delay"}

    @dataclass
    class State:
        """Circular buffer state for delayed actuators."""

        buffer_pos: wp.array = None  # Shape (delay, N)
        buffer_vel: wp.array = None  # Shape (delay, N)
        buffer_act: wp.array = None  # Shape (delay, N)
        write_idx: int = 0           # Last write position
        is_filled: bool = False      # Buffer filled at least once

    def is_stateful(self) -> bool:
        return True

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve arguments with defaults. Requires 'delay'.

        Args:
            args (dict): User-provided arguments.

        Returns:
            dict: Arguments with defaults.

        Raises:
            ValueError: If 'delay' not provided.
        """
        if "delay" not in args:
            raise ValueError("ActuatorDelayedPD requires 'delay' argument")
        return {
            "kp": args.get("kp", 0.0),
            "kd": args.get("kd", 0.0),
            "delay": args["delay"],
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
        delay: int,
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
        """Initialize delayed PD actuator.

        Args:
            input_indices (wp.array): DOF indices for reading state and targets. Shape (N,).
            output_indices (wp.array): DOF indices for writing output. Shape (N,).
            kp (wp.array): Proportional gains. Shape (N,).
            kd (wp.array): Derivative gains. Shape (N,).
            delay (int): Number of timesteps to delay inputs.
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
        self.delay = delay

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
        current_state: "ActuatorDelayedPD.State",
        dt: float,
    ) -> None:
        """Compute delayed PD control forces."""
        if current_state is None or not current_state.is_filled:
            return

        read_idx = (current_state.write_idx + 1) % self.delay
        delayed_pos = current_state.buffer_pos[read_idx]
        delayed_vel = current_state.buffer_vel[read_idx]
        delayed_act = current_state.buffer_act[read_idx]

        wp.launch(
            kernel=pd_controller_kernel,
            dim=self.num_actuators,
            inputs=[
                getattr(sim_state, self.state_pos_attr),
                getattr(sim_state, self.state_vel_attr),
                delayed_pos,
                delayed_vel,
                delayed_act,
                self.input_indices,
                self._sequential_indices,
                output_indices,
                self.kp,
                self.kd,
                self.max_force,
                self.gear,
                self.constant_force,
            ],
            outputs=[controller_output],
        )

    def _run_state_manager(
        self,
        sim_state: Any,
        sim_control: Any,
        current_state: "ActuatorDelayedPD.State",
        next_state: "ActuatorDelayedPD.State",
        dt: float,
    ) -> None:
        """Update circular delay buffer."""
        if next_state is None:
            return

        copy_idx = current_state.write_idx
        write_idx = (current_state.write_idx + 1) % self.delay

        wp.launch(
            kernel=delay_buffer_state_kernel,
            dim=self.num_actuators,
            inputs=[
                getattr(sim_control, self.control_target_pos_attr),
                getattr(sim_control, self.control_target_vel_attr),
                getattr(sim_control, self.control_input_attr),
                self.input_indices,
                copy_idx,
                write_idx,
                current_state.buffer_pos,
                current_state.buffer_vel,
                current_state.buffer_act,
            ],
            outputs=[
                next_state.buffer_pos,
                next_state.buffer_vel,
                next_state.buffer_act,
            ],
        )

        next_state.write_idx = write_idx
        next_state.is_filled = current_state.is_filled or (write_idx == self.delay - 1)

    def state(self) -> "ActuatorDelayedPD.State":
        """Return a new state with allocated circular buffers."""
        device = self.input_indices.device
        return ActuatorDelayedPD.State(
            buffer_pos=wp.zeros((self.delay, self.num_actuators), dtype=wp.float32, device=device),
            buffer_vel=wp.zeros((self.delay, self.num_actuators), dtype=wp.float32, device=device),
            buffer_act=wp.zeros((self.delay, self.num_actuators), dtype=wp.float32, device=device),
            write_idx=self.delay - 1,
            is_filled=False,
        )
