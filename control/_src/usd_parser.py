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

"""USD parser for actuator prims."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actuators import PDActuator, PIDActuator, DelayedPDActuator


@dataclass
class ParsedActuator:
    """Result of parsing a USD actuator prim."""

    actuator_class: type
    target_paths: list[str]
    kwargs: dict[str, Any] = field(default_factory=dict)
    transmission: list[float] | None = None


API_SCHEMA_HANDLERS: dict[str, dict[str, str]] = {
    "PDControllerAPI": {
        "kp": "kp",
        "kd": "kd",
        "maxForce": "max_force",
        "constForce": "constant_force",
        "gear": "gear",
    },
    "PIDControllerAPI": {
        "kp": "kp",
        "ki": "ki",
        "kd": "kd",
        "maxForce": "max_force",
        "integralMax": "integral_max",
        "constForce": "constant_force",
        "gear": "gear",
    },
    "DelayAPI": {
        "delay": "delay",
    },
}


def get_attribute(prim, name: str, default: Any = None) -> Any:
    """Get attribute value from a USD prim, returning default if not found."""
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    return attr.Get()


def get_relationship_targets(prim, name: str) -> list[str]:
    """Get relationship target paths from a USD prim."""
    rel = prim.GetRelationship(name)
    if not rel:
        return []
    return [str(t) for t in rel.GetTargets()]


def get_attribute_prefixes(prim) -> set[str]:
    """Get all attribute namespace prefixes from a prim."""
    prefixes = set()
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if ":" in name:
            prefixes.add(name.split(":")[0])
    return prefixes


def infer_schemas_from_attributes(prim) -> list[str]:
    """Infer API schemas from attribute prefixes."""
    prefixes = get_attribute_prefixes(prim)
    schemas = []
    if "pdcontroller" in prefixes:
        schemas.append("PDControllerAPI")
    if "pidcontroller" in prefixes:
        schemas.append("PIDControllerAPI")
    if "delay" in prefixes:
        schemas.append("DelayAPI")
    return schemas


def determine_actuator_class(schemas: list[str]) -> type:
    """Determine actuator class from inferred schemas."""
    has_delay = "DelayAPI" in schemas
    has_pid = "PIDControllerAPI" in schemas
    has_pd = "PDControllerAPI" in schemas

    if has_delay and has_pd:
        return DelayedPDActuator
    elif has_pid:
        return PIDActuator
    elif has_pd:
        return PDActuator
    else:
        return PDActuator


def extract_kwargs_from_prim(prim, schemas: list[str]) -> dict[str, Any]:
    """Extract actuator parameters from prim attributes."""
    kwargs = {}
    for schema_name in schemas:
        if schema_name not in API_SCHEMA_HANDLERS:
            continue
        param_map = API_SCHEMA_HANDLERS[schema_name]
        schema_prefix = schema_name.replace("API", "").lower()
        for usd_name, kwarg_name in param_map.items():
            for attr_name in [f"{schema_prefix}:{usd_name}", usd_name]:
                value = get_attribute(prim, attr_name)
                if value is not None:
                    kwargs[kwarg_name] = value
                    break
    return kwargs


def parse_actuator_prim(prim) -> ParsedActuator | None:
    """Parse a USD Actuator prim. Returns None if not a valid actuator."""
    if prim.GetTypeName() != "Actuator":
        return None

    target_paths = get_relationship_targets(prim, "target")
    if not target_paths:
        return None

    schemas = infer_schemas_from_attributes(prim)
    transmission = get_attribute(prim, "transmission")

    return ParsedActuator(
        actuator_class=determine_actuator_class(schemas),
        target_paths=target_paths,
        kwargs=extract_kwargs_from_prim(prim, schemas),
        transmission=list(transmission) if transmission else None,
    )

