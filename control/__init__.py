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

"""
Control library for actuator-based control in physics simulations.

This library provides a collection of actuator implementations that integrate with
physics simulation pipelines. Actuators read from simulation state arrays and write
computed forces/torques back to control arrays.

Available Actuators:
    - :class:`~control.Actuator`: Abstract base class for all actuators
    - :class:`~control.PDActuator`: Stateless PD controller
    - :class:`~control.DelayedPDActuator`: PD controller with input delay
    - :class:`~control.PIDActuator`: Stateful PID controller with integral term

State Classes:
    - :class:`~control.DelayedActuatorState`: State for delayed actuators (circular buffer)
    - :class:`~control.PIDActuatorState`: State for PID actuators (integral term)

Typical workflow:

1. Create actuators with appropriate parameters
2. Initialize double-buffered states for stateful actuators
3. In the simulation loop, call actuator.step() to compute forces
4. Swap state buffers for stateful actuators

Example:
    ```python
    import warp as wp
    from control import PDActuator, DelayedPDActuator

    # Create a PD actuator for 3 DOFs
    pd_actuator = PDActuator(
        indices=wp.array([0, 1, 2], dtype=wp.uint32),
        kp=wp.array([100.0, 100.0, 100.0], dtype=wp.float32),
        kd=wp.array([10.0, 10.0, 10.0], dtype=wp.float32),
        max_force=wp.array([50.0, 50.0, 50.0], dtype=wp.float32),
        gear=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
        constant_force=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
    )

    # In simulation loop - stateless actuators don't need state management
    pd_actuator.step(sim_state, sim_control, None, None)
    ```
"""

# Actuator classes
from ._src.actuators import (
    Actuator,
    DelayedPDActuator,
    PDActuator,
    PIDActuator,
)

# State classes
from ._src.types import (
    DelayedActuatorState,
    PIDActuatorState,
)

# USD parsing
from ._src.usd_parser import (
    ParsedActuator,
    parse_actuator_prim,
)

# Version
from ._version import __version__

__all__ = [
    # Version
    "__version__",
    # Actuator classes
    "Actuator",
    "DelayedPDActuator",
    "PDActuator",
    "PIDActuator",
    # State classes
    "DelayedActuatorState",
    "PIDActuatorState",
    # USD parsing
    "ParsedActuator",
    "parse_actuator_prim",
]

