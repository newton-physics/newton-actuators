# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from .controllers.base import Controller
from .delay import Delay
from .dynamics.base import Dynamic


# TODO: replace with a Transmission class that applies gear ratios / linkage
# transforms before accumulating into the output array.
@wp.kernel
def _scatter_add_kernel(
    forces: wp.array(dtype=float),
    output_indices: wp.array(dtype=wp.uint32),
    output: wp.array(dtype=float),
):
    """Accumulate forces into output at specified indices."""
    i = wp.tid()
    out_idx = output_indices[i]
    output[out_idx] = output[out_idx] + forces[i]


@dataclass
class ActuatorState:
    """Composed state for an Actuator.

    Holds the controller state and, if a delay is present, the delay
    state. Dynamics are stateless.
    """

    controller_state: Any = None
    delay_state: Any = None

    def reset(self) -> None:
        if self.controller_state is not None:
            self.controller_state.reset()
        if self.delay_state is not None:
            self.delay_state.reset()


class Actuator:
    """Composed actuator: controller + optional delay + dynamics.

    An actuator reads from simulation state/control arrays, computes
    forces via a controller, applies dynamics (clamping, saturation, etc.),
    and writes the result to the output array.

    Delay is handled separately from dynamics because it is the only
    pre-controller modifier (it replaces targets with delayed versions).
    All dynamics are post-controller (they modify forces).

    Usage::

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(kp=kp, kd=kd),
            delay=Delay(delay=5),
            dynamics=[Clamp(max_force=max_f)],
        )

        # Simulation loop
        actuator.step(sim_state, sim_control, state_a, state_b, dt=0.01)

    Args:
        input_indices: DOF indices for reading state and targets. Shape (N,)
            for single-input or (N, M) for multi-input actuators.
        output_indices: DOF indices for writing output forces. Shape (N,)
            for single-output or (N, M) for multi-output actuators.
        controller: Controller that computes raw forces.
        delay: Optional Delay instance for input delay.
        dynamics: List of Dynamic objects (post-controller force modifiers).
        state_pos_attr: Attribute on sim_state for positions.
        state_vel_attr: Attribute on sim_state for velocities.
        control_target_pos_attr: Attribute on sim_control for target positions.
        control_target_vel_attr: Attribute on sim_control for target velocities.
        control_input_attr: Attribute on sim_control for control input. None to skip.
        control_output_attr: Attribute on sim_control for output forces.
    """

    def __init__(
        self,
        input_indices: wp.array,
        output_indices: wp.array,
        controller: Controller,
        delay: Delay | None = None,
        dynamics: list[Dynamic] | None = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_target_vel_attr: str = "joint_target_vel",
        control_input_attr: str | None = "joint_act",
        control_output_attr: str = "joint_f",
    ):
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.controller = controller
        self.delay = delay
        self.dynamics = dynamics or []
        self.num_actuators = len(input_indices)

        if len(output_indices) != self.num_actuators:
            raise ValueError(
                f"output_indices length ({len(output_indices)}) must match "
                f"input_indices length ({self.num_actuators})"
            )

        self.state_pos_attr = state_pos_attr
        self.state_vel_attr = state_vel_attr
        self.control_target_pos_attr = control_target_pos_attr
        self.control_target_vel_attr = control_target_vel_attr
        self.control_input_attr = control_input_attr
        self.control_output_attr = control_output_attr

        device = input_indices.device
        self._sequential_indices = wp.array(
            np.arange(self.num_actuators, dtype=np.uint32), device=device
        )
        self._forces = wp.zeros(self.num_actuators, dtype=wp.float32, device=device)

        controller.set_device(device)
        controller.set_indices(input_indices, self._sequential_indices)
        for dyn in self.dynamics:
            dyn.set_device(device)
        if self.delay is not None:
            self.delay.set_indices(self.num_actuators, self._sequential_indices)

    @property
    def SHARED_PARAMS(self) -> set[str]:
        params: set[str] = set()
        params |= self.controller.SHARED_PARAMS
        if self.delay is not None:
            params |= self.delay.SHARED_PARAMS
        for d in self.dynamics:
            params |= d.SHARED_PARAMS
        return params

    def is_stateful(self) -> bool:
        """Return True if controller or delay maintains internal state."""
        return self.controller.is_stateful() or self.delay is not None

    def is_graphable(self) -> bool:
        """Return True if all components can be captured in a CUDA graph."""
        return self.controller.is_graphable() and all(d.is_graphable() for d in self.dynamics)

    def has_transmission(self) -> bool:
        """Return True if this actuator applies a transmission transform."""
        return False

    def state(self) -> ActuatorState | None:
        """Return a new composed state, or None if fully stateless."""
        if not self.is_stateful():
            return None
        device = self.input_indices.device
        return ActuatorState(
            controller_state=(
                self.controller.state(self.num_actuators, device)
                if self.controller.is_stateful()
                else None
            ),
            delay_state=(
                self.delay.state(self.num_actuators, device)
                if self.delay is not None
                else None
            ),
        )

    def step(
        self,
        sim_state: Any,
        sim_control: Any,
        current_act_state: ActuatorState | None = None,
        next_act_state: ActuatorState | None = None,
        dt: float = None,
    ) -> None:
        """Execute one control step.

        1. **Delay** — read delayed targets from buffer.
        2. **Controller** — compute raw forces.
        3. **Dynamics** — modify forces and scatter-add to output.
        4. **State updates** — update delay buffer and controller state.

        If the delay buffer is still filling, steps 2–3 are skipped
        (no forces produced) but the buffer keeps accumulating.

        Args:
            sim_state: Simulation state with position/velocity arrays.
            sim_control: Control structure with target/output arrays.
            current_act_state: Current composed state (None if stateless).
            next_act_state: Next composed state (None if stateless).
            dt: Timestep in seconds.
        """
        has_states = current_act_state is not None and next_act_state is not None

        positions = getattr(sim_state, self.state_pos_attr)
        velocities = getattr(sim_state, self.state_vel_attr)

        orig_target_pos = getattr(sim_control, self.control_target_pos_attr)
        orig_target_vel = getattr(sim_control, self.control_target_vel_attr)
        orig_act_input = None
        if self.control_input_attr is not None:
            orig_act_input = getattr(sim_control, self.control_input_attr, None)

        target_pos = orig_target_pos
        target_vel = orig_target_vel
        act_input = orig_act_input
        target_indices = self.input_indices

        # --- 1. Delay: read delayed targets ---
        skip_compute = False
        if self.delay is not None:
            delay_state = current_act_state.delay_state if current_act_state else None

            if self.delay.is_ready(delay_state):
                target_pos, target_vel, act_input, target_indices = (
                    self.delay.get_delayed_targets(act_input, delay_state)
                )
            else:
                skip_compute = True

        if not skip_compute:
            # --- 2. Controller: compute forces ---
            ctrl_state = current_act_state.controller_state if current_act_state else None
            self.controller.compute(
                positions,
                velocities,
                target_pos,
                target_vel,
                act_input,
                self.input_indices,
                target_indices,
                self._forces,
                self._sequential_indices,
                self.num_actuators,
                ctrl_state,
                dt,
            )

            # --- 3. Dynamics: modify forces + write output ---
            for dyn in self.dynamics:
                dyn.modify_forces(
                    self._forces, positions, velocities,
                    self.input_indices, self.num_actuators,
                )

            output = getattr(sim_control, self.control_output_attr)
            wp.launch(
                kernel=_scatter_add_kernel,
                dim=self.num_actuators,
                inputs=[self._forces, self.output_indices],
                outputs=[output],
            )

        # --- 4. State updates ---
        if has_states:
            if self.controller.is_stateful() and not skip_compute:
                self.controller.update_state(
                    current_act_state.controller_state,
                    next_act_state.controller_state,
                )

            if self.delay is not None:
                self.delay.update_state(
                    orig_target_pos, orig_target_vel, orig_act_input,
                    self.input_indices, self.num_actuators,
                    current_act_state.delay_state,
                    next_act_state.delay_state,
                    dt,
                )
