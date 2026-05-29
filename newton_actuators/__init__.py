# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton Actuators - GPU-accelerated actuator library for physics simulations.

.. deprecated::
    This standalone ``newton-actuators`` package is deprecated and will no longer
    be maintained. Starting with Newton 1.3, actuators are available exclusively as
    ``newton.actuators`` from the ``newton`` package. Please migrate:

    - Concepts guide: https://newton-physics.github.io/newton/latest/concepts/actuators.html
    - API reference: https://newton-physics.github.io/newton/latest/api/newton_actuators.html
"""

import warnings as _warnings

_warnings.warn(
    "The 'newton-actuators' package is deprecated and will no longer be maintained. "
    "Starting with Newton 1.3, use 'newton.actuators' from the 'newton' package instead. "
    "See https://newton-physics.github.io/newton/latest/concepts/actuators.html for migration details.",
    DeprecationWarning,
    stacklevel=2,
)

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
