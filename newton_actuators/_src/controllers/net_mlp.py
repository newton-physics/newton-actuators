# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from typing import Any

import warp as wp

from .base import Controller


class NetMLPController(Controller):
    """MLP-based neural network controller.

    Uses a pre-trained MLP to compute joint torques from position error
    and velocity history. Produces raw (unclamped) forces — pair with
    ``Clamp`` for force limiting.

    Input preparation (per actuator):
        pos_input = [pos_error[t] for t in input_idx] * pos_scale
        vel_input = [velocity[t]  for t in input_idx] * vel_scale
        network_input = cat(pos_input, vel_input)   # or vel, pos if input_order="vel_pos"

    Output: torque = network(input) * torque_scale
    """

    SHARED_PARAMS = {"network_path"}

    @dataclass
    class State:
        """History buffers for MLP controller."""

        pos_error_history: Any = None
        vel_history: Any = None

        def reset(self) -> None:
            self.pos_error_history.zero_()
            self.vel_history.zero_()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "network_path" not in args:
            raise ValueError("NetMLPController requires 'network_path' argument")
        return {
            "network_path": args["network_path"],
            "pos_scale": args.get("pos_scale", 1.0),
            "vel_scale": args.get("vel_scale", 1.0),
            "torque_scale": args.get("torque_scale", 1.0),
            "input_order": args.get("input_order", "pos_vel"),
            "input_idx": args.get("input_idx", None),
        }

    def __init__(
        self,
        pos_scale: float = 1.0,
        vel_scale: float = 1.0,
        torque_scale: float = 1.0,
        input_order: str = "pos_vel",
        input_idx: list[int] | None = None,
        network: Any = None,
        network_path: str | None = None,
        device: wp.Device | str | None = None,
    ):
        """Initialize MLP controller.

        Args:
            pos_scale: Scaling factor for position error inputs.
            vel_scale: Scaling factor for velocity inputs.
            torque_scale: Scaling factor for network output torques.
            input_order: Concatenation order, "pos_vel" or "vel_pos".
            input_idx: History timestep indices to feed the network. 0 = current,
                1 = one step ago, etc. Default [0].
            network: Pre-trained network (torch.nn.Module). If None, loaded from network_path.
            network_path: Path to a TorchScript model file.
            device: Warp device or device string (e.g. "cuda:0"). Required when
                using network_path; inferred from network parameters when omitted.
        """
        import torch

        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.torque_scale = torque_scale
        if input_order not in ("pos_vel", "vel_pos"):
            raise ValueError(f"input_order must be 'pos_vel' or 'vel_pos'; got '{input_order}'")
        self.input_order = input_order
        self.input_idx = input_idx if input_idx else [0]
        if any(i < 0 for i in self.input_idx):
            raise ValueError(f"input_idx must contain non-negative integers; got {self.input_idx}")
        self.history_length = max(self.input_idx) + 1

        self.network_path = network_path

        if device is not None:
            wp_device = wp.get_device(device)
            self._torch_device = torch.device(f"cuda:{wp_device.ordinal}" if wp_device.is_cuda else "cpu")
        elif network is not None:
            params = list(network.parameters())
            self._torch_device = params[0].device if params else torch.device("cpu")
        else:
            self._torch_device = torch.device("cpu")

        if network is not None:
            self.network = network.to(self._torch_device).eval()
        elif network_path is not None:
            self.network = torch.jit.load(network_path, map_location=self._torch_device).eval()
        else:
            raise ValueError("Either 'network' or 'network_path' must be provided")

        self._torch_input_indices: Any = None
        self._torch_sequential_indices: Any = None
        self._warp_sequential_indices: wp.array | None = None

    def set_indices(self, input_indices: wp.array, sequential_indices: wp.array) -> None:
        import torch

        self._torch_input_indices = torch.tensor(input_indices.numpy(), dtype=torch.long, device=self._torch_device)
        self._torch_sequential_indices = torch.arange(len(input_indices), dtype=torch.long, device=self._torch_device)
        self._warp_sequential_indices = sequential_indices

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return False

    def state(self, num_actuators: int, device: wp.Device) -> "NetMLPController.State":
        import torch

        return NetMLPController.State(
            pos_error_history=torch.zeros(self.history_length, num_actuators, device=self._torch_device),
            vel_history=torch.zeros(self.history_length, num_actuators, device=self._torch_device),
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
        state: "NetMLPController.State",
        dt: float,
    ) -> None:
        import torch

        current_pos = wp.to_torch(positions)
        current_vel = wp.to_torch(velocities)
        target = wp.to_torch(target_pos)

        torch_target_idx = (
            self._torch_sequential_indices
            if target_indices is self._warp_sequential_indices
            else self._torch_input_indices
        )

        pos_error = target[torch_target_idx] - current_pos[self._torch_input_indices]
        vel = current_vel[self._torch_input_indices]

        state.pos_error_history[0] = pos_error
        state.vel_history[0] = vel

        pos_input = torch.stack([state.pos_error_history[i] for i in self.input_idx], dim=1)
        vel_input = torch.stack([state.vel_history[i] for i in self.input_idx], dim=1)

        if self.input_order == "pos_vel":
            net_input = torch.cat([pos_input * self.pos_scale, vel_input * self.vel_scale], dim=1)
        else:
            net_input = torch.cat([vel_input * self.vel_scale, pos_input * self.pos_scale], dim=1)

        with torch.inference_mode():
            torques = self.network(net_input)

        torques = torques.reshape(num_actuators) * self.torque_scale
        torques_wp = wp.from_torch(torques.contiguous(), dtype=wp.float32)
        wp.copy(forces, torques_wp)

    def update_state(
        self,
        positions: wp.array,
        velocities: wp.array,
        target_pos: wp.array,
        target_vel: wp.array,
        input_indices: wp.array,
        target_indices: wp.array,
        num_actuators: int,
        current_state: "NetMLPController.State",
        next_state: "NetMLPController.State",
        dt: float,
    ) -> None:
        if next_state is None:
            return
        next_state.pos_error_history = current_state.pos_error_history.roll(1, 0)
        next_state.vel_history = current_state.vel_history.roll(1, 0)
