# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import warp as wp

from ..kernels import remotized_clamp_kernel
from .base import Dynamic


class RemotizedClamp(Dynamic):
    """Angle-dependent torque clamping via lookup table.

    Replaces a fixed ±max_force box clamp with angle-dependent torque
    limits interpolated from a lookup table. Models remotized actuators
    (e.g., linkage-driven joints) where the transmission ratio and thus
    maximum output torque vary with joint angle.

    This is a post-controller dynamic.
    """

    SHARED_PARAMS = {"lookup_angles", "lookup_torques"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "lookup_angles" not in args or "lookup_torques" not in args:
            raise ValueError("RemotizedClamp requires 'lookup_angles' and 'lookup_torques' arguments")
        return {
            "lookup_angles": tuple(args["lookup_angles"]),
            "lookup_torques": tuple(args["lookup_torques"]),
        }

    def __init__(
        self,
        lookup_angles: wp.array | tuple[float, ...] | list[float],
        lookup_torques: wp.array | tuple[float, ...] | list[float],
        device: wp.Device | str | None = None,
    ):
        """Initialize remotized clamp dynamic.

        Args:
            lookup_angles: Sorted joint angles for the torque lookup table. Shape (K,).
            lookup_torques: Max output torques corresponding to lookup_angles. Shape (K,).
            device: Warp device. Required when passing plain lists/tuples;
                    ignored when passing wp.array (device is inferred).
        """
        if len(lookup_angles) != len(lookup_torques):
            raise ValueError(
                f"lookup_angles length ({len(lookup_angles)}) must match "
                f"lookup_torques length ({len(lookup_torques)})"
            )
        self.lookup_size = len(lookup_angles)

        if isinstance(lookup_angles, wp.array):
            self.lookup_angles = lookup_angles
        else:
            self.lookup_angles = wp.array(
                np.array(lookup_angles, dtype=np.float32), device=device
            )

        if isinstance(lookup_torques, wp.array):
            self.lookup_torques = lookup_torques
        else:
            self.lookup_torques = wp.array(
                np.array(lookup_torques, dtype=np.float32), device=device
            )

    def modify_forces(
        self,
        forces: wp.array,
        positions: wp.array,
        velocities: wp.array,
        input_indices: wp.array,
        num_actuators: int,
    ) -> None:
        wp.launch(
            kernel=remotized_clamp_kernel,
            dim=num_actuators,
            inputs=[
                positions,
                input_indices,
                self.lookup_angles,
                self.lookup_torques,
                self.lookup_size,
            ],
            outputs=[forces],
        )
