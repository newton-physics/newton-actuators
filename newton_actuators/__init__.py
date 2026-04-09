# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton Actuators - GPU-accelerated actuator library for physics simulations."""

from ._src.actuator import Actuator, ActuatorState
from ._src.controllers import (
    Controller,
    NetLSTMController,
    NetMLPController,
    PDController,
    PIDController,
)
from ._src.delay import Delay
from ._src.clamping import (
    Clamp,
    DCMotorSaturation,
    Clamping,
    RemotizedClamp,
)
from ._src.usd_parser import (
    ParsedActuator,
    parse_actuator_prim,
)
from ._version import __version__

__all__ = [
    "__version__",
    # Composer
    "Actuator",
    "ActuatorState",
    # Controllers
    "Controller",
    "PDController",
    "PIDController",
    "NetMLPController",
    "NetLSTMController",
    # Clamping
    "Clamping",
    "Clamp",
    "DCMotorSaturation",
    "Delay",
    "RemotizedClamp",
    # USD
    "ParsedActuator",
    "parse_actuator_prim",
]
