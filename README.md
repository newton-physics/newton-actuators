[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton-actuators/main)

**This project is in active beta development.** This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

# Newton Actuators

GPU-accelerated actuator library for physics simulations.

This library provides composable actuator implementations that integrate with
physics simulation pipelines. Actuators read from simulation state arrays and
write computed forces/torques back to control arrays.

## Architecture

An actuator is composed from three building blocks:

- **Controller** — computes raw forces from state error (PD, PID, neural network).
- **Delay** — optional pre-controller modifier that delays targets by N timesteps.
- **Clamping** — post-controller force bounds (symmetric limits,
  velocity-dependent saturation, angle-dependent torque curves, …).
  Order does not matter — each clamping intersects its limits independently.

The `Actuator` class wires them together:

```
Actuator
├── Controller  (compute raw forces)
├── Delay       (optional: delays targets by N timesteps)
└── Clamping[]  (post-controller force bounds)
    ├── Clamp              (±max_force)
    ├── DCMotorSaturation  (velocity-dependent saturation)
    └── RemotizedClamp     (angle-dependent lookup)
```

## Installation

```bash
pip install newton-actuators
```

Or install from source:

```bash
cd newton-actuators
pip install -e .
```

### With PyTorch (for neural network controllers)

The `NetMLPController` and `NetLSTMController` require PyTorch. Install the
extra matching your CUDA version:

**Using uv** (index routing is automatic):

```bash
uv pip install "newton-actuators[torch-cu12]"   # CUDA 12.x
uv pip install "newton-actuators[torch-cu13]"   # CUDA 13.x
```

**Using pip** (requires manual `--extra-index-url`):

```bash
pip install "newton-actuators[torch-cu12]" --extra-index-url https://download.pytorch.org/whl/cu128
pip install "newton-actuators[torch-cu13]" --extra-index-url https://download.pytorch.org/whl/cu130
```

## API Reference

### Controllers

| Controller | Description | Stateful |
|---|---|---|
| `PDController` | Proportional-derivative controller | No |
| `PIDController` | PID controller with anti-windup integral | Yes |
| `NetMLPController` | MLP network with position/velocity history | Yes |
| `NetLSTMController` | LSTM network with recurrent hidden state | Yes |

#### Force Laws

- **PDController**: `f = constant + act + Kp·(target_pos - q) + Kd·(target_vel - v)`
- **PIDController**: `f = constant + act + Kp·e + Ki·∫e·dt + Kd·de`
- **NetMLPController**: `f = network(cat(pos_error_history, vel_history))`
- **NetLSTMController**: `f, (h', c') = network(input, (h, c))`

### Delay

| Component | Description | Stateful |
|---|---|---|
| `Delay` | Delays targets by N timesteps (circular buffer) | Yes |

Passed to `Actuator` via the `delay=` parameter (not in the clamping list).

### Clamping

| Clamping | Description | Stateful |
|---|---|---|
| `Clamp` | Box-clamp to ±max_force | No |
| `DCMotorSaturation` | Velocity-dependent torque–speed saturation | No |
| `RemotizedClamp` | Angle-dependent torque limits via lookup table | No |

### Actuator (Composer)

`Actuator(input_indices, output_indices, controller, delay=None, clamping=[...])`
composes a controller with an optional delay and zero or more clamping objects.
The `step()` method runs:

1. **Delay** — read delayed targets from buffer (skipped if no delay or buffer still filling)
2. **Controller** — compute raw forces
3. **Clamping** — bound forces (e.g. `Clamp`, `DCMotorSaturation`)
4. **Scatter-add** — accumulate forces into the output array
5. **State updates** — update delay buffer and controller state

### Common Methods

- `actuator.is_stateful()` — True if any component maintains internal state
- `actuator.is_graphable()` — True if `step()` can be captured in a CUDA graph
- `actuator.state()` — return a new `ActuatorState` (None if stateless)
- `actuator.step(sim_state, sim_control, current_state, next_state, dt)` — one control step
- `state.reset()` — zero all internal buffers without reallocating

## Workflow

1. **Create an actuator** by composing a controller with clamping
2. **Check statefulness**: call `actuator.is_stateful()`
3. **Initialize states**: for stateful actuators, create double-buffered states with `actuator.state()`
4. **Simulation loop**: call `actuator.step()` each timestep
5. **Swap buffers**: for stateful actuators, swap state buffers after each step
6. **Reset between episodes**: call `state.reset()` to zero internal buffers

## Examples

### Stateless: PD + Clamp

```python
import warp as wp
from newton_actuators import Actuator, PDController, Clamp

indices = wp.array([0, 1, 2], dtype=wp.uint32)
actuator = Actuator(
    input_indices=indices,
    output_indices=indices,
    controller=PDController(
        kp=wp.array([100.0, 100.0, 100.0], dtype=wp.float32),
        kd=wp.array([10.0, 10.0, 10.0], dtype=wp.float32),
    ),
    clamping=[
        Clamp(max_force=wp.array([50.0, 50.0, 50.0], dtype=wp.float32)),
    ],
)

# Stateless — no state management needed
actuator.step(sim_state, sim_control, None, None, dt=0.01)
```

### Stateful: PID + Clamp

```python
from newton_actuators import PIDController

actuator = Actuator(
    input_indices=indices,
    output_indices=indices,
    controller=PIDController(
        kp=wp.array([100.0, 100.0], dtype=wp.float32),
        ki=wp.array([10.0, 10.0], dtype=wp.float32),
        kd=wp.array([5.0, 5.0], dtype=wp.float32),
        integral_max=wp.array([10.0, 10.0], dtype=wp.float32),
    ),
    clamping=[
        Clamp(max_force=wp.array([50.0, 50.0], dtype=wp.float32)),
    ],
)

state_a = actuator.state()
state_b = actuator.state()

current, nxt = state_a, state_b
for step in range(num_steps):
    actuator.step(sim_state, sim_control, current, nxt, dt=0.01)
    current, nxt = nxt, current
```

### Composing: PD + Delay + DC Motor Saturation

The composer pattern lets you freely combine controllers and clamping:

```python
from newton_actuators import Delay, DCMotorSaturation

actuator = Actuator(
    input_indices=indices,
    output_indices=indices,
    controller=PDController(kp=kp, kd=kd),
    delay=Delay(delay=5),
    clamping=[
        DCMotorSaturation(
            saturation_effort=sat_effort,
            velocity_limit=vel_limit,
            max_force=max_force,
        ),
    ],
)
```

### LSTM Controller

```python
from newton_actuators import NetLSTMController

actuator = Actuator(
    input_indices=indices,
    output_indices=indices,
    controller=NetLSTMController(network=my_lstm_model),
    clamping=[Clamp(max_force=max_force)],
)

state = actuator.state()
for step in range(num_steps):
    actuator.step(sim_state, sim_control, state, state, dt=0.01)
```

## USD Parsing

```python
from newton_actuators import parse_actuator_prim

result = parse_actuator_prim(prim)
if result is not None:
    controller_cls = result.controller_class    # e.g. PDController
    ctrl_kwargs = result.controller_kwargs      # e.g. {"kp": 100.0, "kd": 10.0}
    component_specs = result.component_specs    # e.g. [(Delay, {"delay": 5}), (Clamp, {"max_force": 50.0})]
    target_paths = result.target_paths          # e.g. ["/World/Robot/Joint1"]
```

## License

Apache-2.0
