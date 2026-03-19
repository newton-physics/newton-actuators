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
| `ActuatorDCMotor` | PD with DC motor velocity-dependent saturation | No | No |
| `ActuatorRemotizedPD` | Delayed PD with angle-dependent torque limits | Yes | No |
| `ActuatorNetMLP` | MLP network actuator with position/velocity history | Yes | No |
| `ActuatorNetLSTM` | LSTM network actuator with recurrent hidden state | Yes | No |

#### Control Laws

- **ActuatorPD**: `τ = clamp(constant + act + Kp·(target_pos - q) + Kd·(target_vel - v), ±max_force)`
- **ActuatorPID**: `τ = clamp(constant + act + Kp·(target_pos - q) + Ki·∫e·dt + Kd·(target_vel - v), ±max_force)`
- **ActuatorDelayedPD**: Same as PD but with delayed targets (circular buffer)
- **ActuatorDCMotor**: Same PD force computation, but torque is clamped to velocity-dependent bounds from the motor torque-speed curve: `τ_max(v) = clamp(τ_sat·(1 - v/v_max), 0, effort_limit)`, `τ_min(v) = clamp(τ_sat·(-1 - v/v_max), -effort_limit, 0)`, `τ = clamp(τ, τ_min(v), τ_max(v))`
- **ActuatorRemotizedPD**: Same as DelayedPD, but torque limits are interpolated from an angle-dependent lookup table: `τ_limit = interp(q, lookup_table)`
- **ActuatorNetMLP**: `τ = clamp(network(cat(pos_error_history * pos_scale, vel_history * vel_scale)) * torque_scale, ±max_force)` — history is maintained internally
- **ActuatorNetLSTM**: `τ = clamp(network(input, (h, c)), ±max_force)` — hidden and cell state maintained internally

### Base Class Methods

All actuators inherit from `Actuator` and provide these methods:

- `resolve_arguments(args) -> dict`: (classmethod) Resolve user-provided arguments with defaults
- `is_stateful() -> bool`: Returns True if the actuator maintains internal state
- `is_graphable() -> bool`: Returns True if `step()` can be captured in a CUDA graph (False for torch-based NN actuators)
- `has_transmission() -> bool`: Returns True if the actuator has a transmission phase
- `state() -> State | None`: Returns a new state instance (None for stateless actuators)
- `step(sim_state, sim_control, current_state, next_state, dt)`: Execute one control step

### State Classes

Stateful actuators use nested State classes:

- `ActuatorPID.State` - Contains the integral term for PID control
- `ActuatorDelayedPD.State` - Contains circular buffers for delayed targets
- `ActuatorRemotizedPD.State` - Inherits `ActuatorDelayedPD.State` (same delay buffers)

- `ActuatorNetMLP.State` - Contains position error and velocity history buffers
- `ActuatorNetLSTM.State` - Contains LSTM hidden and cell state tensors

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

### Non-Graphable Stateful Actuator (ActuatorNetLSTM)

Network actuators (`ActuatorNetMLP`, `ActuatorNetLSTM`) are stateful but not
CUDA-graphable due to Warp-PyTorch interop. Because their `step()` cannot be
captured in a CUDA graph, double-buffering is not strictly required — you can
pass the **same state object** as both `current_state` and `next_state` to
simplify your code:

```python
# Simple: single state object (fine when not using CUDA graphs)
state = lstm_actuator.state()
for step in range(num_steps):
    lstm_actuator.step(sim_state, sim_control, state, state, dt=0.01)
```

To reset state between episodes, call `state.reset()`:

```python
state.reset()
```

### DC Motor Actuator

```python
import warp as wp
from newton_actuators import ActuatorDCMotor

indices = wp.array([0, 1], dtype=wp.uint32)
dc_motor = ActuatorDCMotor(
    input_indices=indices,
    output_indices=indices,
    kp=wp.array([200.0, 200.0], dtype=wp.float32),
    kd=wp.array([20.0, 20.0], dtype=wp.float32),
    max_force=wp.array([50.0, 50.0], dtype=wp.float32),
    saturation_effort=wp.array([80.0, 80.0], dtype=wp.float32),
    velocity_limit=wp.array([10.0, 10.0], dtype=wp.float32),
)

# Stateless - no state management needed
dc_motor.step(sim_state, sim_control, None, None, dt=0.01)
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
