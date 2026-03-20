# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton Actuators - GPU-accelerated actuator library for physics simulations."""

from ._src.actuators import (
    Actuator,
    ActuatorDCMotor,
    ActuatorDelayedPD,
    ActuatorNetLSTM,
    ActuatorNetMLP,
    ActuatorPD,
    ActuatorPID,
    ActuatorRemotizedPD,
)
from ._src.usd_parser import (
    ParsedActuator,
    parse_actuator_prim,
)
from ._version import __version__

__all__ = [
    "__version__",
    "Actuator",
    "ActuatorDCMotor",
    "ActuatorDelayedPD",
    "ActuatorNetLSTM",
    "ActuatorNetMLP",
    "ActuatorPD",
    "ActuatorPID",
    "ActuatorRemotizedPD",
    "ParsedActuator",
    "parse_actuator_prim",
]
