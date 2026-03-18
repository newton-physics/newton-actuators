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

"""LSTM-based neural network actuator."""

from typing import Any

import torch
import warp as wp

from ..kernels import nn_output_kernel
from .base import Actuator


class ActuatorNetLSTM(Actuator):
    """LSTM-based neural network actuator.

    Uses a pre-trained LSTM network to compute joint torques from position
    error and velocity. The network maintains hidden and cell state across
    timesteps to capture temporal dynamics of the actuator.

    The network must be callable as:
        torques, (h_new, c_new) = network(input, (h, c))

    where input has shape (batch, 1, 2) with features [pos_error, velocity],
    and h/c have shape (num_layers, batch, hidden_size).

    The network is expected to have a `lstm` attribute (torch.nn.LSTM) so that
    num_layers and hidden_size can be inferred automatically.

    Output: torque = network_output, clamped to ±max_force.
    """

    def is_graphable(self) -> bool:
        return False

    def __init__(
        self,
        input_indices: wp.array,
        output_indices: wp.array,
        network: Any,
        max_force: wp.array,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_output_attr: str = "joint_f",
    ):
        """Initialize LSTM actuator.

        Args:
            input_indices: DOF indices for reading state and targets. Shape (N,).
            output_indices: DOF indices for writing output. Shape (N,).
            network: Pre-trained LSTM network (torch.nn.Module) or path to TorchScript file.
                Must be callable as network(input, (h, c)) -> (output, (h_new, c_new)) and
                expose a `lstm` attribute for dimension inference.
            max_force: Per-actuator force limits. Shape (N,).
            state_pos_attr: Attribute on sim_state for joint positions.
            state_vel_attr: Attribute on sim_state for joint velocities.
            control_target_pos_attr: Attribute on sim_control for target positions.
            control_output_attr: Attribute on sim_control for output forces.
        """
        super().__init__(input_indices, output_indices, control_output_attr)

        if len(max_force) != self.num_actuators:
            raise ValueError(f"max_force length ({len(max_force)}) must match num_actuators ({self.num_actuators})")

        self.max_force = max_force

        device = input_indices.device
        self._torch_device = torch.device(f"cuda:{device.ordinal}" if device.is_cuda else "cpu")

        if isinstance(network, str):
            self.network = torch.jit.load(network, map_location=self._torch_device).eval()
        else:
            self.network = network.to(self._torch_device).eval()

        num_layers = self.network.lstm.num_layers
        hidden_size = self.network.lstm.hidden_size

        self._torch_indices = torch.tensor(input_indices.numpy(), dtype=torch.long, device=self._torch_device)

        self.state_pos_attr = state_pos_attr
        self.state_vel_attr = state_vel_attr
        self.control_target_pos_attr = control_target_pos_attr

        self.hidden = torch.zeros(num_layers, self.num_actuators, hidden_size, device=self._torch_device)
        self.cell = torch.zeros(num_layers, self.num_actuators, hidden_size, device=self._torch_device)

    def _run_controller(
        self,
        sim_state: Any,
        sim_control: Any,
        controller_output: wp.array,
        output_indices: wp.array,
        current_state: Any,
        dt: float,
    ) -> None:
        """Compute LSTM network torques."""
        current_pos = wp.to_torch(getattr(sim_state, self.state_pos_attr))
        current_vel = wp.to_torch(getattr(sim_state, self.state_vel_attr))
        target_pos = wp.to_torch(getattr(sim_control, self.control_target_pos_attr))

        pos_error = target_pos[self._torch_indices] - current_pos[self._torch_indices]
        vel = current_vel[self._torch_indices]

        # (num_actuators, 1, 2): seq_len=1, features=[pos_error, velocity]
        net_input = torch.stack([pos_error, vel], dim=1).unsqueeze(1)

        with torch.inference_mode():
            torques, (self.hidden, self.cell) = self.network(net_input, (self.hidden, self.cell))

        torques = torques.reshape(-1)
        torques_wp = wp.from_torch(torques.contiguous(), dtype=wp.float32)

        wp.launch(
            kernel=nn_output_kernel,
            dim=self.num_actuators,
            inputs=[torques_wp, self.max_force, output_indices],
            outputs=[controller_output],
        )
