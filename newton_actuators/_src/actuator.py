# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from .controllers.base import Controller
from .dynamics.base import Dynamic
from .kernels import scatter_add_kernel


@dataclass
class ActuatorState:
    """Composed state for an Actuator.

    Holds the controller state and a list of dynamic states, one per
    dynamic in the actuator's dynamics list.
    """

    controller_state: Any = None
    dynamic_states: list = field(default_factory=list)

    def reset(self) -> None:
        """Reset all sub-states in-place."""
        if self.controller_state is not None:
            self.controller_state.reset()
        for s in self.dynamic_states:
            if s is not None:
                s.reset()


class Actuator:
    """Composed actuator: controller + dynamics.

    An actuator reads from simulation state/control arrays, computes
    forces via a controller, applies dynamics (delay, clamping, etc.),
    and writes the result to the output array.

    Usage::

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(kp=kp, kd=kd),
            dynamics=[Delay(delay=5), Clamp(max_force=max_f)],
        )

        # Simulation loop
        actuator.step(sim_state, sim_control, state_a, state_b, dt=0.01)

    Args:
        input_indices: DOF indices for reading state and targets. Shape (N,).
        output_indices: DOF indices for writing output forces. Shape (N,).
        controller: Controller that computes raw forces.
        dynamics: List of Dynamic objects applied in order.
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

        controller.bind(input_indices, self._sequential_indices, device)
        for dyn in self.dynamics:
            dyn.bind(self.num_actuators, self._sequential_indices, device)

    @property
    def SCALAR_PARAMS(self) -> set[str]:
        params: set[str] = set()
        params |= self.controller.SCALAR_PARAMS
        for d in self.dynamics:
            params |= d.SCALAR_PARAMS
        return params

    def is_stateful(self) -> bool:
        """Return True if any component maintains internal state."""
        return self.controller.is_stateful() or any(d.is_stateful() for d in self.dynamics)

    def is_graphable(self) -> bool:
        """Return True if all components can be captured in a CUDA graph."""
        return self.controller.is_graphable() and all(d.is_graphable() for d in self.dynamics)

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
            dynamic_states=[
                d.state(self.num_actuators, device) if d.is_stateful() else None
                for d in self.dynamics
            ],
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

        1. Extract arrays from sim_state / sim_control.
        2. Apply pre-controller dynamics (modify targets).
        3. Run controller (compute raw forces).
        4. Apply post-controller dynamics (modify forces).
        5. Scatter-add forces to output.
        6. Update states.

        Args:
            sim_state: Simulation state with position/velocity arrays.
            sim_control: Control structure with target/output arrays.
            current_act_state: Current composed state (None if stateless).
            next_act_state: Next composed state (None if stateless).
            dt: Timestep in seconds.
        """
        positions = getattr(sim_state, self.state_pos_attr)
        velocities = getattr(sim_state, self.state_vel_attr)

        orig_target_pos = getattr(sim_control, self.control_target_pos_attr)
        orig_target_vel = getattr(sim_control, self.control_target_vel_attr)
        orig_act_input = None
        if self.control_input_attr is not None:
            orig_act_input = getattr(sim_control, self.control_input_attr, None)

        # --- Pre-controller dynamics (modify targets) ---
        target_pos = orig_target_pos
        target_vel = orig_target_vel
        act_input = orig_act_input
        target_indices = self.input_indices

        skip_compute = False
        for i, dyn in enumerate(self.dynamics):
            dyn_state = current_act_state.dynamic_states[i] if current_act_state else None
            result = dyn.modify_targets(target_pos, target_vel, act_input, target_indices, dyn_state)
            if result is None:
                skip_compute = True
                break
            target_pos, target_vel, act_input, target_indices = result

        if not skip_compute:
            # --- Controller ---
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

            # --- Post-controller dynamics (modify forces) ---
            for i, dyn in enumerate(self.dynamics):
                dyn_state = current_act_state.dynamic_states[i] if current_act_state else None
                dyn.modify_forces(
                    self._forces, positions, velocities,
                    self.input_indices, self.num_actuators, dyn_state,
                )

            # --- Scatter-add to output ---
            output = getattr(sim_control, self.control_output_attr)
            wp.launch(
                kernel=scatter_add_kernel,
                dim=self.num_actuators,
                inputs=[self._forces, self.output_indices],
                outputs=[output],
            )

        # --- State updates ---
        if self.is_stateful() and current_act_state is not None and next_act_state is not None:
            if self.controller.is_stateful() and not skip_compute:
                self.controller.update_state(
                    positions, velocities,
                    target_pos, target_vel,
                    self.input_indices, target_indices,
                    self.num_actuators,
                    current_act_state.controller_state,
                    next_act_state.controller_state,
                    dt,
                )

            for i, dyn in enumerate(self.dynamics):
                if dyn.is_stateful():
                    dyn.update_state(
                        orig_target_pos, orig_target_vel, orig_act_input,
                        self.input_indices, self.num_actuators,
                        current_act_state.dynamic_states[i],
                        next_act_state.dynamic_states[i],
                        dt,
                    )
