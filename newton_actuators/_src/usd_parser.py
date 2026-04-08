# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .controllers import PDController, PIDController
from .dynamics import Clamp, DCMotorSaturation, Delay


@dataclass
class ParsedActuator:
    """Result of parsing a USD actuator prim.

    Contains the controller class, dynamics classes, and their respective
    kwargs so that the caller can construct an ``Actuator`` by composing them.
    """

    controller_class: type
    controller_kwargs: dict[str, Any] = field(default_factory=dict)
    dynamics_specs: list[tuple[type, dict[str, Any]]] = field(default_factory=list)
    target_paths: list[str] = field(default_factory=list)
    transmission: list[float] | None = None


API_SCHEMA_HANDLERS: dict[str, dict[str, str]] = {
    "PDControllerAPI": {
        "kp": "kp",
        "kd": "kd",
        "constForce": "constant_force",
    },
    "PIDControllerAPI": {
        "kp": "kp",
        "ki": "ki",
        "kd": "kd",
        "integralMax": "integral_max",
        "constForce": "constant_force",
    },
    "ClampAPI": {
        "maxForce": "max_force",
    },
    "DelayAPI": {
        "delay": "delay",
    },
    "DCMotorAPI": {
        "saturationEffort": "saturation_effort",
        "velocityLimit": "velocity_limit",
        "maxForce": "max_force",
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


def get_actuator_attribute_names(prim) -> set[str]:
    """Get all newton:actuator attribute names from a prim."""
    names = set()
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        if attr_name.startswith("newton:actuator:"):
            names.add(attr_name.split(":")[-1])
    return names


def infer_schemas_from_prim(prim) -> list[str]:
    """Infer actuator schemas from attribute names."""
    attr_names = get_actuator_attribute_names(prim)
    schemas = []

    if "ki" in attr_names:
        schemas.append("PIDControllerAPI")
    elif "kp" in attr_names or "kd" in attr_names:
        schemas.append("PDControllerAPI")
    if "maxForce" in attr_names:
        schemas.append("ClampAPI")
    if "delay" in attr_names:
        schemas.append("DelayAPI")
    if "saturationEffort" in attr_names and "velocityLimit" in attr_names:
        schemas.append("DCMotorAPI")

    return schemas


def determine_controller_and_dynamics(schemas: list[str]) -> tuple[type, list[type]]:
    """Determine controller class and dynamics classes from inferred schemas."""
    has_pid = "PIDControllerAPI" in schemas
    has_delay = "DelayAPI" in schemas
    has_dc_motor = "DCMotorAPI" in schemas
    has_clamp = "ClampAPI" in schemas

    controller_cls = PIDController if has_pid else PDController

    dynamics_classes: list[type] = []
    if has_delay:
        dynamics_classes.append(Delay)
    if has_dc_motor:
        dynamics_classes.append(DCMotorSaturation)
    elif has_clamp:
        dynamics_classes.append(Clamp)

    return controller_cls, dynamics_classes


def validate_kwargs(schemas: list[str], kwargs: dict[str, Any]) -> None:
    """Validate extracted kwargs. Raises ValueError on invalid values."""
    if "DCMotorAPI" in schemas:
        vel_lim = kwargs.get("velocity_limit")
        if vel_lim is not None and vel_lim <= 0.0:
            raise ValueError(
                f"DCMotorAPI requires velocity_limit > 0 (division by velocity_limit "
                f"in saturation computation); got {vel_lim}"
            )


def extract_kwargs_from_prim(prim, schemas: list[str]) -> dict[str, Any]:
    """Extract actuator parameters from prim attributes."""
    kwargs: dict[str, Any] = {}
    for schema_name in schemas:
        if schema_name not in API_SCHEMA_HANDLERS:
            continue
        param_map = API_SCHEMA_HANDLERS[schema_name]
        for usd_name, kwarg_name in param_map.items():
            value = get_attribute(prim, f"newton:actuator:{usd_name}")
            if value is not None:
                kwargs[kwarg_name] = value
    validate_kwargs(schemas, kwargs)
    return kwargs


def _split_kwargs(
    kwargs: dict[str, Any],
    controller_cls: type,
    dynamics_classes: list[type],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split flat kwargs into controller kwargs and per-dynamic kwargs."""
    ctrl_keys = set(controller_cls.resolve_arguments({}).keys())
    ctrl_kwargs = {k: v for k, v in kwargs.items() if k in ctrl_keys}

    dynamics_kwargs_list = []
    for dyn_cls in dynamics_classes:
        try:
            dyn_defaults = dyn_cls.resolve_arguments(kwargs)
        except (ValueError, KeyError):
            dyn_defaults = {}
        dyn_kwargs = {k: v for k, v in kwargs.items() if k in dyn_defaults}
        dynamics_kwargs_list.append(dyn_kwargs)

    return ctrl_kwargs, dynamics_kwargs_list


def parse_actuator_prim(prim) -> ParsedActuator | None:
    """Parse a USD Actuator prim into a composed actuator specification.

    Returns None if not a valid actuator prim.
    """
    if prim.GetTypeName() != "Actuator":
        return None

    target_paths = get_relationship_targets(prim, "newton:actuator:target")
    if not target_paths:
        return None

    schemas = infer_schemas_from_prim(prim)
    all_kwargs = extract_kwargs_from_prim(prim, schemas)
    transmission = get_attribute(prim, "newton:actuator:transmission")

    controller_cls, dynamics_classes = determine_controller_and_dynamics(schemas)
    ctrl_kwargs, dynamics_kwargs_list = _split_kwargs(all_kwargs, controller_cls, dynamics_classes)

    dynamics_specs = list(zip(dynamics_classes, dynamics_kwargs_list))

    return ParsedActuator(
        controller_class=controller_cls,
        controller_kwargs=ctrl_kwargs,
        dynamics_specs=dynamics_specs,
        target_paths=target_paths,
        transmission=list(transmission) if transmission else None,
    )
