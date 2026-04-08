# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp

from ..kernels import pd_force_kernel
from .base import Controller


class PDController(Controller):
    """Stateless PD controller.

    Force law: f = constant + act + Kp*(target_pos - q) + Kd*(target_vel - v)

    Produces raw (unclamped) forces. Pair with ``Clamp`` or other dynamics
    for force limiting.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "kp": args.get("kp", 0.0),
            "kd": args.get("kd", 0.0),
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        kp: wp.array,
        kd: wp.array,
        constant_force: wp.array | None = None,
    ):
        """Initialize PD controller.

        Args:
            kp: Proportional gains. Shape (N,).
            kd: Derivative gains. Shape (N,).
            constant_force: Constant force offsets. Shape (N,). None to skip.
        """
        self.kp = kp
        self.kd = kd
        self.constant_force = constant_force

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
        state: Any,
        dt: float,
    ) -> None:
        wp.launch(
            kernel=pd_force_kernel,
            dim=num_actuators,
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                act_input,
                input_indices,
                target_indices,
                force_indices,
                self.kp,
                self.kd,
                self.constant_force,
            ],
            outputs=[forces],
        )
