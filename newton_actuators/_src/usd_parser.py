# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .controllers import PDController, PIDController
from .delay import Delay
from .dynamics import Clamp, DCMotorSaturation


@dataclass
class SchemaEntry:
    """Maps an API schema to a component class and its USD→kwarg param names."""

    component_class: type
    param_map: dict[str, str]
    is_controller: bool = False
    validate: Any = None


def _validate_dc_motor(kwargs: dict[str, Any]) -> None:
    vel_lim = kwargs.get("velocity_limit")
    if vel_lim is not None and vel_lim <= 0.0:
        raise ValueError(
            f"DCMotorAPI requires velocity_limit > 0 (division by velocity_limit "
            f"in saturation computation); got {vel_lim}"
        )


# Temporary registry until the actual USD schema is merged.
SCHEMA_REGISTRY: dict[str, SchemaEntry] = {
    "PDControllerAPI": SchemaEntry(
        component_class=PDController,
        param_map={"kp": "kp", "kd": "kd", "constForce": "constant_force"},
        is_controller=True,
    ),
    "PIDControllerAPI": SchemaEntry(
        component_class=PIDController,
        param_map={"kp": "kp", "ki": "ki", "kd": "kd", "integralMax": "integral_max", "constForce": "constant_force"},
        is_controller=True,
    ),
    "ClampAPI": SchemaEntry(
        component_class=Clamp,
        param_map={"maxForce": "max_force"},
    ),
    "DelayAPI": SchemaEntry(
        component_class=Delay,
        param_map={"delay": "delay"},
    ),
    "DCMotorAPI": SchemaEntry(
        component_class=DCMotorSaturation,
        param_map={"saturationEffort": "saturation_effort", "velocityLimit": "velocity_limit", "maxForce": "max_force"},
        validate=_validate_dc_motor,
    ),
}


@dataclass
class ParsedActuator:
    """Result of parsing a USD actuator prim.

    Each detected API schema produces a (class, kwargs) entry.
    The controller is separated out; everything else goes into
    component_specs (delay, dynamics, etc.).
    """

    controller_class: type
    controller_kwargs: dict[str, Any] = field(default_factory=dict)
    component_specs: list[tuple[type, dict[str, Any]]] = field(default_factory=list)
    target_paths: list[str] = field(default_factory=list)
    transmission: list[float] | None = None


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


def get_schemas_from_prim(prim) -> list[str]:
    """Get applied schemas that match the registry.

    Uses prim.GetAppliedSchemas() and matches against SCHEMA_REGISTRY keys.
    """
    # TODO: replace string matching with proper USD schema type checks
    applied = prim.GetAppliedSchemas()
    return [s for s in applied if s in SCHEMA_REGISTRY]


def parse_actuator_prim(prim) -> ParsedActuator | None:
    """Parse a USD Actuator prim into a composed actuator specification.

    Each detected schema directly maps to a component class with its
    extracted params. Returns None if not a valid actuator prim.
    """
    if prim.GetTypeName() != "Actuator":
        return None

    target_paths = get_relationship_targets(prim, "newton:actuator:target")
    if not target_paths:
        return None

    schemas = get_schemas_from_prim(prim)
    transmission = get_attribute(prim, "newton:actuator:transmission")

    controller_class = None
    controller_kwargs: dict[str, Any] = {}
    component_specs: list[tuple[type, dict[str, Any]]] = []

    for schema_name in schemas:
        entry = SCHEMA_REGISTRY.get(schema_name)
        if entry is None:
            continue

        kwargs: dict[str, Any] = {}
        for usd_name, kwarg_name in entry.param_map.items():
            value = get_attribute(prim, f"newton:actuator:{usd_name}")
            if value is not None:
                kwargs[kwarg_name] = value

        if entry.validate is not None:
            entry.validate(kwargs)

        if entry.is_controller:
            controller_class = entry.component_class
            controller_kwargs = kwargs
        else:
            component_specs.append((entry.component_class, kwargs))

    if controller_class is None:
        return None

    return ParsedActuator(
        controller_class=controller_class,
        controller_kwargs=controller_kwargs,
        component_specs=component_specs,
        target_paths=target_paths,
        transmission=list(transmission) if transmission else None,
    )
