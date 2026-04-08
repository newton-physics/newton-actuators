# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp


class Dynamic:
    """Base class for composable dynamics that modify actuator behavior.

    Dynamics are stacked on top of a controller to add behaviors such as
    delay, clamping, or velocity-dependent saturation. Each dynamic can
    modify targets before the controller runs (``modify_targets``) and/or
    modify forces after the controller runs (``modify_forces``).

    Class Attributes:
        SCALAR_PARAMS: Parameter names that are instance-level (shared across
            all DOFs). Different values require separate actuator instances.
    """

    SCALAR_PARAMS: set[str] = set()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    def bind(self, num_actuators: int, sequential_indices: wp.array, device: wp.Device) -> None:
        """Called by Actuator to provide sizing and device info.

        Args:
            num_actuators: Number of actuators N.
            sequential_indices: Sequential indices [0..N). Shape (N,).
            device: Warp device.
        """
        pass

    def modify_targets(
        self,
        target_pos: wp.array,
        target_vel: wp.array,
        act_input: wp.array | None,
        target_indices: wp.array,
        current_state: Any,
    ) -> tuple[wp.array, wp.array, wp.array | None, wp.array] | None:
        """Pre-controller: modify or replace targets.

        Return (target_pos, target_vel, act_input, target_indices) with modified
        arrays, or ``None`` to signal the controller should be skipped this step
        (e.g. delay buffer not yet filled).

        Args:
            target_pos: Current target positions.
            target_vel: Current target velocities.
            act_input: Feedforward control input (may be None).
            target_indices: Indices into target arrays.
            current_state: Dynamic state (None if stateless).

        Returns:
            Modified (target_pos, target_vel, act_input, target_indices) or None to skip.
        """
        return target_pos, target_vel, act_input, target_indices

    def modify_forces(
        self,
        forces: wp.array,
        positions: wp.array,
        velocities: wp.array,
        input_indices: wp.array,
        num_actuators: int,
        current_state: Any,
    ) -> None:
        """Post-controller: modify forces in-place.

        Args:
            forces: Force buffer to modify. Shape (N,).
            positions: Joint positions (global array).
            velocities: Joint velocities (global array).
            input_indices: Indices into positions/velocities.
            num_actuators: Number of actuators N.
            current_state: Dynamic state (None if stateless).
        """
        pass

    def update_state(
        self,
        target_pos: wp.array,
        target_vel: wp.array,
        act_input: wp.array | None,
        input_indices: wp.array,
        num_actuators: int,
        current_state: Any,
        next_state: Any,
        dt: float,
    ) -> None:
        """Update internal state.

        Called with the **original** (pre-dynamic) targets so that state
        updates record what the simulation actually commanded.

        Args:
            target_pos: Original target positions (from sim_control).
            target_vel: Original target velocities (from sim_control).
            act_input: Original feedforward control input (may be None).
            input_indices: Original DOF indices.
            num_actuators: Number of actuators N.
            current_state: Current dynamic state.
            next_state: Next dynamic state to write.
            dt: Timestep in seconds.
        """
        pass

    def is_stateful(self) -> bool:
        """Return True if this dynamic maintains internal state."""
        return False

    def is_graphable(self) -> bool:
        """Return True if all operations can be captured in a CUDA graph."""
        return True

    def state(self, num_actuators: int, device: wp.Device) -> Any:
        """Create and return a new state object, or None if stateless."""
        return None
