# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import warp as wp

from ..kernels import box_clamp_kernel
from .base import Dynamic


class Clamp(Dynamic):
    """Box-clamp dynamic.

    Clamps controller output forces to ±max_force per actuator.
    This is a post-controller dynamic.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        return {"max_force": args.get("max_force", math.inf)}

    def __init__(self, max_force: wp.array):
        """Initialize clamp dynamic.

        Args:
            max_force: Per-actuator force limits. Shape (N,).
        """
        self.max_force = max_force

    def modify_forces(
        self,
        forces: wp.array,
        positions: wp.array,
        velocities: wp.array,
        input_indices: wp.array,
        num_actuators: int,
        current_state: Any,
    ) -> None:
        wp.launch(
            kernel=box_clamp_kernel,
            dim=num_actuators,
            inputs=[self.max_force],
            outputs=[forces],
        )
