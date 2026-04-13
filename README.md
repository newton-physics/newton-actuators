[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton-actuators/main)

**This project is in active beta development.** This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

# Newton Actuators

GPU-accelerated actuator library for physics simulations.

This library provides composable actuator implementations that read physics simulation state, compute actuator forces, and write the forces back to control arrays for application to the simulation. The simulator does not need to be part of Newton: the library is designed to be reusable anywhere the caller can provide state arrays and consume forces. Each actuator instance is vectorized: a single actuator object operates on a batch of indices in global state and control arrays, allowing efficient integration into RL workflows. The goal is to provide canonical actuator models with support for differentiability and graphable execution where the underlying controller implementation supports it. The library is designed to be easy for users to customize and extend for their specific actuator models.

**Current limitations (v1):** no transmission support, SISO only, no dynamics component.

## Architecture

An actuator is composed from three building blocks:

- **Controller** — control law that computes raw forces or torques from the current simulator state and control targets (PD, PID, neural-network-based control).
- **Delay** — optionally delays the control targets (e.g. position or velocity) by N timesteps before they reach the controller, allowing the actuator to model communication or processing latency.
- **Clamping** — applies post-controller output limits to the computed forces or torques to model motor limits, such as saturation, back-EMF losses, performance envelopes, or angle-dependent torque limits. Multiple clamping stages can be combined; order does not matter as each stage applies its limits independently.

The `Actuator` class wires them together:

```
Actuator
├── Controller      (control law that computes raw forces)
├── Delay           (optional: delays control targets by N timesteps)
└── Clamping[]          (post-controller force bounds)
    ├── ClampingMaxForce       (±max_force box clamp)
    ├── ClampingDCMotor   (velocity-dependent saturation)
    └── ClampingPositionBased  (angle-dependent lookup)
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

The `ControllerNetMLP` and `ControllerNetLSTM` require PyTorch. Install the
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
| `ControllerPD` | Proportional-derivative controller | No |
| `ControllerPID` | PID controller with integral clamping | Yes |
| `ControllerNetMLP` | MLP network with position/velocity history | Yes |
| `ControllerNetLSTM` | LSTM network with recurrent hidden state | Yes |

#### Force Laws

- **ControllerPD**: `f = bias + act + Kp·(target_pos - q) + Kd·(target_vel - v)`
- **ControllerPID**: `f = bias + act + Kp·e + Ki·∫e·dt + Kd·ė` where `e = target_pos - q`
- **ControllerNetMLP**: `f = network(input)` where input includes position-error and velocity history
- **ControllerNetLSTM**: `f, (h', c') = network(input, (h, c))`

### Delay

| Component | Description | Stateful |
|---|---|---|
| `Delay` | Delays control targets by N timesteps (circular buffer) | Yes |

Passed to `Actuator` via the `delay=` parameter.

### Clamping

| Clamping | Description | Stateful |
|---|---|---|
| `ClampingMaxForce` | Box-clamp raw forces to ±max_force | No |
| `ClampingDCMotor` | Velocity-dependent torque–speed saturation | No |
| `ClampingPositionBased` | Angle-dependent torque limits via lookup table | No |

Multiple clamping stages can be combined freely; each applies its own limits independently (intersection semantics).

### Actuator (Composer)

`Actuator(indices, controller, delay=None, clamping=[...])`
composes a controller with an optional delay and zero or more clamping objects.
The `step()` method runs:

1. **Delay** — read delayed targets from buffer (zero force output while buffer is still filling)
2. **Controller** — compute raw forces
3. **Clamping** — clamp raw forces (e.g. `ClampingMaxForce`, `ClampingDCMotor`)
4. **Scatter-add** — accumulate forces into the output array at the specified DOF indices
5. **State updates** — update delay buffer and controller state

### Common Methods

- `actuator.is_stateful()` — True if any component maintains internal state
- `actuator.is_graphable()` — True if `step()` can be captured in a CUDA graph
- `actuator.state()` — return a new `StateActuator` (None if stateless)
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

### Stateless: PD + ClampingMaxForce

```python
import warp as wp
from newton_actuators import Actuator, ControllerPD, ClampingMaxForce

indices = wp.array([0, 1, 2], dtype=wp.uint32)
actuator = Actuator(
    indices=indices,
    controller=ControllerPD(
        kp=wp.array([100.0, 100.0, 100.0], dtype=wp.float32),
        kd=wp.array([10.0, 10.0, 10.0], dtype=wp.float32),
    ),
    clamping=[
        ClampingMaxForce(max_force=wp.array([50.0, 50.0, 50.0], dtype=wp.float32)),
    ],
)

# Stateless — no state management needed
actuator.step(sim_state, sim_control, None, None, dt=0.01)
```

### Stateful: PID + ClampingMaxForce

```python
from newton_actuators import ControllerPID

actuator = Actuator(
    indices=indices,
    controller=ControllerPID(
        kp=wp.array([100.0, 100.0], dtype=wp.float32),
        ki=wp.array([10.0, 10.0], dtype=wp.float32),
        kd=wp.array([5.0, 5.0], dtype=wp.float32),
        integral_max=wp.array([10.0, 10.0], dtype=wp.float32),
    ),
    clamping=[
        ClampingMaxForce(max_force=wp.array([50.0, 50.0], dtype=wp.float32)),
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
from newton_actuators import Delay, ClampingDCMotor

actuator = Actuator(
    indices=indices,
    controller=ControllerPD(kp=kp, kd=kd),
    delay=Delay(delay=5),
    clamping=[
        ClampingDCMotor(
            saturation_effort=sat_effort,
            velocity_limit=vel_limit,
            max_force=max_force,
        ),
    ],
)
```

### LSTM Controller

```python
from newton_actuators import ControllerNetLSTM

actuator = Actuator(
    indices=indices,
    controller=ControllerNetLSTM(network=my_lstm_model),
    clamping=[ClampingMaxForce(max_force=max_force)],
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
    controller_cls = result.controller_class    # e.g. ControllerPD
    ctrl_kwargs = result.controller_kwargs      # e.g. {"kp": 100.0, "kd": 10.0}
    component_specs = result.component_specs    # e.g. [(Delay, {"delay": 5}), (ClampingMaxForce, {"max_force": 50.0})]
    target_paths = result.target_paths          # e.g. ["/World/Robot/Joint1"]
```

## License

Apache-2.0
