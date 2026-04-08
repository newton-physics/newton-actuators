# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import warp as wp

from ..kernels import delay_buffer_state_kernel
from .base import Dynamic


class Delay(Dynamic):
    """Input delay dynamic.

    Delays targets by N timesteps using a circular buffer, modelling
    actuator communication or processing lag. When the buffer is not
    yet filled, the controller is skipped (no force output).

    This is a pre-controller dynamic: it replaces targets with delayed
    versions before the controller sees them.
    """

    SCALAR_PARAMS = {"delay"}

    @dataclass
    class State:
        """Circular buffer state for delayed targets."""

        buffer_pos: wp.array = None  # Shape (delay, N)
        buffer_vel: wp.array = None  # Shape (delay, N)
        buffer_act: wp.array = None  # Shape (delay, N)
        write_idx: int = 0
        is_filled: bool = False

        def reset(self) -> None:
            self.buffer_pos.zero_()
            self.buffer_vel.zero_()
            self.buffer_act.zero_()
            self.write_idx = self.buffer_pos.shape[0] - 1
            self.is_filled = False

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "delay" not in args:
            raise ValueError("Delay requires 'delay' argument")
        return {"delay": args["delay"]}

    def __init__(self, delay: int):
        """Initialize delay dynamic.

        Args:
            delay: Number of timesteps to delay inputs.
        """
        self.delay = delay
        self._sequential_indices: wp.array | None = None

    def bind(self, num_actuators: int, sequential_indices: wp.array, device: wp.Device) -> None:
        self._sequential_indices = sequential_indices

    def is_stateful(self) -> bool:
        return True

    def state(self, num_actuators: int, device: wp.Device) -> "Delay.State":
        return Delay.State(
            buffer_pos=wp.zeros((self.delay, num_actuators), dtype=wp.float32, device=device),
            buffer_vel=wp.zeros((self.delay, num_actuators), dtype=wp.float32, device=device),
            buffer_act=wp.zeros((self.delay, num_actuators), dtype=wp.float32, device=device),
            write_idx=self.delay - 1,
            is_filled=False,
        )

    def modify_targets(
        self,
        target_pos: wp.array,
        target_vel: wp.array,
        act_input: wp.array | None,
        target_indices: wp.array,
        current_state: "Delay.State",
    ) -> tuple[wp.array, wp.array, wp.array | None, wp.array] | None:
        if current_state is None or not current_state.is_filled:
            return None

        read_idx = (current_state.write_idx + 1) % self.delay
        delayed_pos = current_state.buffer_pos[read_idx]
        delayed_vel = current_state.buffer_vel[read_idx]
        delayed_act = current_state.buffer_act[read_idx] if act_input is not None else None

        return delayed_pos, delayed_vel, delayed_act, self._sequential_indices

    def update_state(
        self,
        target_pos: wp.array,
        target_vel: wp.array,
        act_input: wp.array | None,
        input_indices: wp.array,
        num_actuators: int,
        current_state: "Delay.State",
        next_state: "Delay.State",
        dt: float,
    ) -> None:
        if next_state is None:
            return

        copy_idx = current_state.write_idx
        write_idx = (current_state.write_idx + 1) % self.delay

        wp.launch(
            kernel=delay_buffer_state_kernel,
            dim=num_actuators,
            inputs=[
                target_pos,
                target_vel,
                act_input,
                input_indices,
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
