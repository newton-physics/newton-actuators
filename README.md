[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton-actuators/main)

**This project is in active beta development.** This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

# Newton Actuators

GPU-accelerated actuator library for physics simulations.

This library provides a collection of actuator implementations that integrate with physics simulation pipelines. Actuators read from simulation state arrays and write computed forces/torques back to control arrays.

## Installation

```bash
pip install newton-actuators
```

Or install from source:

```bash
cd newton-actuators
pip install -e .
```

## API Reference

### Actuator Classes

| Actuator | Description | Stateful | Transmission |
|----------|-------------|----------|--------------|
| `ActuatorPD` | Stateless PD controller | No | No |
| `ActuatorPID` | PID controller with integral term | Yes | No |
| `ActuatorDelayedPD` | PD controller with input delay | Yes | No |

#### Control Laws

- **ActuatorPD**: `τ = clamp(G·[constant + act + Kp·(target_pos - G·q) + Kd·(target_vel - G·v)], ±max_force)`
- **ActuatorPID**: `τ = clamp(G·[constant + act + Kp·(target_pos - G·q) + Ki·∫e·dt + Kd·(target_vel - G·v)], ±max_force)`
- **ActuatorDelayedPD**: Same as PD but with delayed targets (circular buffer)

### Base Class Methods

All actuators inherit from `Actuator` and provide these methods:

- `resolve_arguments(args) -> dict`: (classmethod) Resolve user-provided arguments with defaults
- `is_stateful() -> bool`: Returns True if the actuator maintains internal state
- `has_transmission() -> bool`: Returns True if the actuator has a transmission phase
- `state() -> State | None`: Returns a new state instance (None for stateless actuators)
- `step(sim_state, sim_control, current_state, next_state, dt)`: Execute one control step

### State Classes

Stateful actuators use nested State classes:

- `ActuatorPID.State` - Contains the integral term for PID control
- `ActuatorDelayedPD.State` - Contains circular buffers for delayed targets

## Workflow

1. **Create actuators** with appropriate parameters
2. **Check statefulness**: Call `actuator.is_stateful()` to determine if state management is needed
3. **Initialize states**: For stateful actuators, create double-buffered states with `actuator.state()`
4. **Simulation loop**: Call `actuator.step()` to compute forces
5. **Swap buffers**: For stateful actuators, swap state buffers after each step

## Examples

### Stateless Actuator (ActuatorPD)

```python
import warp as wp
from newton_actuators import ActuatorPD

# Create a PD actuator for 3 DOFs
indices = wp.array([0, 1, 2], dtype=wp.uint32)
pd_actuator = ActuatorPD(
    input_indices=indices,
    output_indices=indices,
    kp=wp.array([100.0, 100.0, 100.0], dtype=wp.float32),
    kd=wp.array([10.0, 10.0, 10.0], dtype=wp.float32),
    max_force=wp.array([50.0, 50.0, 50.0], dtype=wp.float32),
    gear=wp.array([1.0, 1.0, 1.0], dtype=wp.float32),
    constant_force=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
)

# In simulation loop - stateless actuators don't need state management
pd_actuator.step(sim_state, sim_control, None, None, dt=0.01)
```

### Stateful Actuator (ActuatorPID)

```python
import warp as wp
from newton_actuators import ActuatorPID

indices = wp.array([0, 1], dtype=wp.uint32)
pid_actuator = ActuatorPID(
    input_indices=indices,
    output_indices=indices,
    kp=wp.array([100.0, 100.0], dtype=wp.float32),
    ki=wp.array([10.0, 10.0], dtype=wp.float32),
    kd=wp.array([5.0, 5.0], dtype=wp.float32),
    max_force=wp.array([50.0, 50.0], dtype=wp.float32),
    integral_max=wp.array([10.0, 10.0], dtype=wp.float32),
    gear=wp.array([1.0, 1.0], dtype=wp.float32),
    constant_force=wp.array([0.0, 0.0], dtype=wp.float32),
)

# Check if actuator needs state management
if pid_actuator.is_stateful():
    # Create double-buffered states
    state_a = pid_actuator.state()
    state_b = pid_actuator.state()

# Simulation loop with state swapping
current_state, next_state = state_a, state_b
for step in range(num_steps):
    pid_actuator.step(sim_state, sim_control, current_state, next_state, dt=0.01)
    current_state, next_state = next_state, current_state  # Swap buffers
```

## USD Parsing

The library includes utilities for parsing actuator definitions from USD files:

```python
from newton_actuators import parse_actuator_prim

# Parse a USD prim with actuator attributes
result = parse_actuator_prim(prim)
if result is not None:
    actuator_class = result.actuator_class  # e.g., ActuatorPD
    target_paths = result.target_paths      # e.g., ["/World/Robot/Joint1"]
    kwargs = result.kwargs                   # e.g., {"kp": 100.0, "kd": 10.0}
```

## License

Apache-2.0
