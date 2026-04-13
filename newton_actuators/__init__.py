# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton Actuators - GPU-accelerated actuator library for physics simulations."""

from ._src.actuator import Actuator, StateActuator
from ._src.clamping import (
    Clamping,
    ClampingMaxForce,
    ClampingPositionBased,
    ClampingVelocityBased,
)
from ._src.controllers import (
    Controller,
    ControllerNetLSTM,
    ControllerNetMLP,
    ControllerPD,
    ControllerPID,
)
from ._src.delay import Delay
from ._src.usd_parser import (
    ParsedActuator,
    parse_actuator_prim,
)
from ._version import __version__

__all__ = [
    "__version__",
    # Composer
    "Actuator",
    "StateActuator",
    # Controllers
    "Controller",
    "ControllerPD",
    "ControllerPID",
    "ControllerNetMLP",
    "ControllerNetLSTM",
    # Delay
    "Delay",
    # Clamping
    "Clamping",
    "ClampingMaxForce",
    "ClampingPositionBased",
    "ClampingVelocityBased",
    # USD
    "ParsedActuator",
    "parse_actuator_prim",
]
