# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton_actuators import (
    Actuator,
    Clamp,
    DCMotorSaturation,
    Delay,
    PDController,
    PIDController,
    RemotizedClamp,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None


@dataclass
class MockSimState:
    """Mock simulation state for testing."""

    joint_q: wp.array
    joint_qd: wp.array
    tendon_length: wp.array = None
    tendon_vel: wp.array = None


@dataclass
class MockSimControl:
    """Mock simulation control for testing."""

    joint_target_pos: wp.array
    joint_target_vel: wp.array
    joint_act: wp.array
    joint_f: wp.array
    tendon_target_length: wp.array = None
    tendon_target_vel: wp.array = None
    tendon_force: wp.array = None


# ---------------------------------------------------------------------------
# PD Controller + Clamp
# ---------------------------------------------------------------------------


class TestPDWithClamp(unittest.TestCase):
    """Tests for PDController + Clamp (replaces old ActuatorPD)."""

    def setUp(self):
        wp.init()

    def test_creation(self):
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
        self.assertIsInstance(actuator, Actuator)
        self.assertIsNone(actuator.state())
        self.assertFalse(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

    def test_step_computes_correct_forces(self):
        num_dofs = 3
        indices = wp.array([0, 1, 2], dtype=wp.uint32)

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([100.0, 100.0, 100.0], dtype=wp.float32),
                kd=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
            ),
            clamping=[
                Clamp(max_force=wp.array([1000.0, 1000.0, 1000.0], dtype=wp.float32)),
            ],
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
            joint_qd=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0, 2.0, 3.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
            joint_act=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
            joint_f=wp.zeros(num_dofs, dtype=wp.float32),
        )

        actuator.step(sim_state, sim_control, None, None)
        forces = sim_control.joint_f.numpy()
        np.testing.assert_allclose(forces, [100.0, 200.0, 300.0], rtol=1e-5)

    def test_pd_without_clamping(self):
        """PD controller without any clamping produces unclamped forces."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([100.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
        )
        self.assertFalse(actuator.is_stateful())

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([0.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([10.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )
        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 1000.0, places=3)

    def test_resolve_arguments(self):
        resolved = PDController.resolve_arguments({"kp": 50.0})
        self.assertEqual(resolved["kp"], 50.0)
        self.assertEqual(resolved["kd"], 0.0)
        self.assertEqual(resolved["constant_force"], 0.0)


# ---------------------------------------------------------------------------
# PD Controller + Delay + Clamp
# ---------------------------------------------------------------------------


class TestPDWithDelay(unittest.TestCase):
    """Tests for PDController + Delay + Clamp (replaces old ActuatorDelayedPD)."""

    def setUp(self):
        wp.init()

    def test_creation(self):
        indices = wp.array([0, 1], dtype=wp.uint32)
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([100.0, 100.0], dtype=wp.float32),
                kd=wp.array([10.0, 10.0], dtype=wp.float32),
            ),
            delay=Delay(delay=5),
            clamping=[
                Clamp(max_force=wp.array([50.0, 50.0], dtype=wp.float32)),
            ],
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

    def test_state_shape(self):
        delay = 5
        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32)
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([100.0, 100.0], dtype=wp.float32),
                kd=wp.array([10.0, 10.0], dtype=wp.float32),
            ),
            delay=Delay(delay=delay),
        )

        state = actuator.state()
        self.assertIsNotNone(state)
        delay_state = state.delay_state
        self.assertEqual(delay_state.write_idx, delay - 1)
        self.assertFalse(delay_state.is_filled)
        self.assertEqual(delay_state.buffer_pos.shape, (delay, num_dofs))
        self.assertEqual(delay_state.buffer_vel.shape, (delay, num_dofs))
        self.assertEqual(delay_state.buffer_act.shape, (delay, num_dofs))

    def test_delay_requires_argument(self):
        with self.assertRaises(ValueError):
            Delay.resolve_arguments({"kp": 50.0})

    def test_delay_behavior(self):
        """Targets are delayed by N steps."""
        delay = 3
        num_dofs = 1
        indices = wp.array([0], dtype=wp.uint32)

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([1.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
            delay=Delay(delay=delay),
            clamping=[
                Clamp(max_force=wp.array([1000.0], dtype=wp.float32)),
            ],
        )

        stateA = actuator.state()
        stateB = actuator.state()

        target_history = []
        force_history = []

        for step in range(delay + 3):
            target_value = float(step + 1) * 10.0
            target_history.append(target_value)

            sim_state = MockSimState(
                joint_q=wp.array([0.0], dtype=wp.float32),
                joint_qd=wp.array([0.0], dtype=wp.float32),
            )
            sim_control = MockSimControl(
                joint_target_pos=wp.array([target_value], dtype=wp.float32),
                joint_target_vel=wp.array([0.0], dtype=wp.float32),
                joint_act=wp.array([0.0], dtype=wp.float32),
                joint_f=wp.zeros(num_dofs, dtype=wp.float32),
            )

            if step % 2 == 0:
                current, next_state = stateA, stateB
            else:
                current, next_state = stateB, stateA

            actuator.step(sim_state, sim_control, current, next_state, dt=0.01)
            force_history.append(sim_control.joint_f.numpy()[0])

        for i in range(delay):
            self.assertEqual(force_history[i], 0.0, f"Step {i}: expected 0 during fill phase")

        self.assertAlmostEqual(force_history[3], target_history[0], places=5)
        self.assertAlmostEqual(force_history[4], target_history[1], places=5)
        self.assertAlmostEqual(force_history[5], target_history[2], places=5)


# ---------------------------------------------------------------------------
# PID Controller + Clamp
# ---------------------------------------------------------------------------


class TestPIDWithClamp(unittest.TestCase):
    """Tests for PIDController + Clamp (replaces old ActuatorPID)."""

    def setUp(self):
        wp.init()

    def test_creation(self):
        indices = wp.array([0, 1], dtype=wp.uint32)
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
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

    def test_state(self):
        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32)
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

        state = actuator.state()
        self.assertIsNotNone(state)
        self.assertIsNotNone(state.controller_state)
        self.assertEqual(state.controller_state.integral.shape[0], num_dofs)
        np.testing.assert_array_equal(state.controller_state.integral.numpy(), [0.0, 0.0])


# ---------------------------------------------------------------------------
# PD Controller + DCMotorSaturation
# ---------------------------------------------------------------------------


class TestPDWithDCMotor(unittest.TestCase):
    """Tests for PDController + DCMotorSaturation (replaces old ActuatorDCMotor)."""

    def setUp(self):
        wp.init()

    def test_creation(self):
        indices = wp.array([0, 1], dtype=wp.uint32)
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([100.0, 100.0], dtype=wp.float32),
                kd=wp.array([10.0, 10.0], dtype=wp.float32),
            ),
            clamping=[
                DCMotorSaturation(
                    saturation_effort=wp.array([80.0, 80.0], dtype=wp.float32),
                    velocity_limit=wp.array([10.0, 10.0], dtype=wp.float32),
                    max_force=wp.array([50.0, 50.0], dtype=wp.float32),
                ),
            ],
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertFalse(actuator.is_stateful())
        self.assertIsNone(actuator.state())
        self.assertTrue(actuator.is_graphable())

    def test_requires_velocity_limit(self):
        with self.assertRaises(ValueError):
            DCMotorSaturation.resolve_arguments({"kp": 50.0})

    def test_resolve_arguments(self):
        resolved = DCMotorSaturation.resolve_arguments({"kp": 50.0, "velocity_limit": 10.0})
        self.assertEqual(resolved["velocity_limit"], 10.0)

    def test_zero_velocity_full_torque(self):
        indices = wp.array([0], dtype=wp.uint32)
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([100.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
            clamping=[
                DCMotorSaturation(
                    saturation_effort=wp.array([150.0], dtype=wp.float32),
                    velocity_limit=wp.array([10.0], dtype=wp.float32),
                    max_force=wp.array([200.0], dtype=wp.float32),
                ),
            ],
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([0.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )
        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 100.0, places=3)

    def test_velocity_reduces_max_torque(self):
        indices = wp.array([0], dtype=wp.uint32)
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([1000.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
            clamping=[
                DCMotorSaturation(
                    saturation_effort=wp.array([100.0], dtype=wp.float32),
                    velocity_limit=wp.array([10.0], dtype=wp.float32),
                    max_force=wp.array([200.0], dtype=wp.float32),
                ),
            ],
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([5.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )
        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 50.0, places=3)

    def test_at_velocity_limit(self):
        indices = wp.array([0], dtype=wp.uint32)
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([1000.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
            clamping=[
                DCMotorSaturation(
                    saturation_effort=wp.array([100.0], dtype=wp.float32),
                    velocity_limit=wp.array([10.0], dtype=wp.float32),
                    max_force=wp.array([200.0], dtype=wp.float32),
                ),
            ],
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([10.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )
        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 0.0, places=3)

    def test_negative_velocity_increases_positive_limit(self):
        indices = wp.array([0], dtype=wp.uint32)
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([1000.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
            clamping=[
                DCMotorSaturation(
                    saturation_effort=wp.array([100.0], dtype=wp.float32),
                    velocity_limit=wp.array([10.0], dtype=wp.float32),
                    max_force=wp.array([200.0], dtype=wp.float32),
                ),
            ],
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([-5.0], dtype=wp.float32),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )
        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 150.0, places=3)


# ---------------------------------------------------------------------------
# PD Controller + Delay + RemotizedClamp
# ---------------------------------------------------------------------------


class TestPDWithRemotizedClamp(unittest.TestCase):
    """Tests for PDController + Delay + RemotizedClamp (replaces old ActuatorRemotizedPD)."""

    def setUp(self):
        wp.init()

    def _make_lookup(self):
        angles = wp.array([-1.0, 0.0, 1.0], dtype=wp.float32)
        torques = wp.array([10.0, 30.0, 50.0], dtype=wp.float32)
        return angles, torques

    def test_creation(self):
        indices = wp.array([0, 1], dtype=wp.uint32)
        angles, torques = self._make_lookup()
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([100.0, 100.0], dtype=wp.float32),
                kd=wp.array([10.0, 10.0], dtype=wp.float32),
            ),
            delay=Delay(delay=3),
            clamping=[
                RemotizedClamp(lookup_angles=angles, lookup_torques=torques),
            ],
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

    def test_requires_lookup(self):
        with self.assertRaises(ValueError):
            RemotizedClamp.resolve_arguments({"kp": 50.0})

    def test_angle_dependent_clipping(self):
        delay = 2
        indices = wp.array([0], dtype=wp.uint32)
        angles, torques = self._make_lookup()

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([1000.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
            delay=Delay(delay=delay),
            clamping=[
                RemotizedClamp(lookup_angles=angles, lookup_torques=torques),
            ],
        )

        stateA = actuator.state()
        stateB = actuator.state()

        for step in range(delay + 1):
            sim_state = MockSimState(
                joint_q=wp.array([0.0], dtype=wp.float32),
                joint_qd=wp.array([0.0], dtype=wp.float32),
            )
            sim_control = MockSimControl(
                joint_target_pos=wp.array([1.0], dtype=wp.float32),
                joint_target_vel=wp.array([0.0], dtype=wp.float32),
                joint_act=wp.array([0.0], dtype=wp.float32),
                joint_f=wp.zeros(1, dtype=wp.float32),
            )
            if step % 2 == 0:
                current, next_s = stateA, stateB
            else:
                current, next_s = stateB, stateA
            actuator.step(sim_state, sim_control, current, next_s, dt=0.01)

        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 30.0, places=3)

    def test_different_angles(self):
        delay = 2
        indices = wp.array([0], dtype=wp.uint32)
        angles, torques = self._make_lookup()

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([1000.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
            delay=Delay(delay=delay),
            clamping=[
                RemotizedClamp(lookup_angles=angles, lookup_torques=torques),
            ],
        )

        for test_angle, expected_limit in [(-1.0, 10.0), (-0.5, 20.0), (0.5, 40.0), (1.0, 50.0)]:
            stateA = actuator.state()
            stateB = actuator.state()

            for step in range(delay + 1):
                sim_state = MockSimState(
                    joint_q=wp.array([test_angle], dtype=wp.float32),
                    joint_qd=wp.array([0.0], dtype=wp.float32),
                )
                sim_control = MockSimControl(
                    joint_target_pos=wp.array([test_angle + 10.0], dtype=wp.float32),
                    joint_target_vel=wp.array([0.0], dtype=wp.float32),
                    joint_act=wp.array([0.0], dtype=wp.float32),
                    joint_f=wp.zeros(1, dtype=wp.float32),
                )
                if step % 2 == 0:
                    current, next_s = stateA, stateB
                else:
                    current, next_s = stateB, stateA
                actuator.step(sim_state, sim_control, current, next_s, dt=0.01)

            force = sim_control.joint_f.numpy()[0]
            self.assertAlmostEqual(
                force, expected_limit, places=2, msg=f"At angle={test_angle}, expected limit={expected_limit}"
            )

    def test_no_force_during_fill(self):
        delay = 3
        indices = wp.array([0], dtype=wp.uint32)
        angles, torques = self._make_lookup()

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([100.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
            delay=Delay(delay=delay),
            clamping=[
                RemotizedClamp(lookup_angles=angles, lookup_torques=torques),
            ],
        )

        stateA = actuator.state()
        stateB = actuator.state()

        for step in range(delay):
            sim_state = MockSimState(
                joint_q=wp.array([0.0], dtype=wp.float32),
                joint_qd=wp.array([0.0], dtype=wp.float32),
            )
            sim_control = MockSimControl(
                joint_target_pos=wp.array([1.0], dtype=wp.float32),
                joint_target_vel=wp.array([0.0], dtype=wp.float32),
                joint_act=wp.array([0.0], dtype=wp.float32),
                joint_f=wp.zeros(1, dtype=wp.float32),
            )
            if step % 2 == 0:
                current, next_s = stateA, stateB
            else:
                current, next_s = stateB, stateA
            actuator.step(sim_state, sim_control, current, next_s, dt=0.01)
            force = sim_control.joint_f.numpy()[0]
            self.assertEqual(force, 0.0, f"Step {step}: expected 0 during fill phase")


# ---------------------------------------------------------------------------
# Composition tests — novel combinations
# ---------------------------------------------------------------------------


class TestComposition(unittest.TestCase):
    """Test novel combinations enabled by the composer pattern."""

    def setUp(self):
        wp.init()

    def test_pd_with_delay_and_dc_motor(self):
        """PD + Delay + DCMotorSaturation — a combination not possible before."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PDController(
                kp=wp.array([1000.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
            ),
            delay=Delay(delay=2),
            clamping=[
                DCMotorSaturation(
                    saturation_effort=wp.array([100.0], dtype=wp.float32),
                    velocity_limit=wp.array([10.0], dtype=wp.float32),
                    max_force=wp.array([200.0], dtype=wp.float32),
                ),
            ],
        )
        self.assertTrue(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

        stateA = actuator.state()
        stateB = actuator.state()

        for step in range(3):
            sim_state = MockSimState(
                joint_q=wp.array([0.0], dtype=wp.float32),
                joint_qd=wp.array([0.0], dtype=wp.float32),
            )
            sim_control = MockSimControl(
                joint_target_pos=wp.array([1.0], dtype=wp.float32),
                joint_target_vel=wp.array([0.0], dtype=wp.float32),
                joint_act=wp.array([0.0], dtype=wp.float32),
                joint_f=wp.zeros(1, dtype=wp.float32),
            )
            if step % 2 == 0:
                current, next_s = stateA, stateB
            else:
                current, next_s = stateB, stateA
            actuator.step(sim_state, sim_control, current, next_s, dt=0.01)

        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 100.0, places=3)

    def test_pid_with_delay_and_clamp(self):
        """PID + Delay + Clamp — another novel combination."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=PIDController(
                kp=wp.array([100.0], dtype=wp.float32),
                ki=wp.array([10.0], dtype=wp.float32),
                kd=wp.array([0.0], dtype=wp.float32),
                integral_max=wp.array([100.0], dtype=wp.float32),
            ),
            delay=Delay(delay=2),
            clamping=[
                Clamp(max_force=wp.array([50.0], dtype=wp.float32)),
            ],
        )
        self.assertTrue(actuator.is_stateful())

        stateA = actuator.state()
        stateB = actuator.state()
        self.assertIsNotNone(stateA.controller_state)
        self.assertIsNotNone(stateA.delay_state)

        for step in range(5):
            sim_state = MockSimState(
                joint_q=wp.array([0.0], dtype=wp.float32),
                joint_qd=wp.array([0.0], dtype=wp.float32),
            )
            sim_control = MockSimControl(
                joint_target_pos=wp.array([1.0], dtype=wp.float32),
                joint_target_vel=wp.array([0.0], dtype=wp.float32),
                joint_act=wp.array([0.0], dtype=wp.float32),
                joint_f=wp.zeros(1, dtype=wp.float32),
            )
            if step % 2 == 0:
                actuator.step(sim_state, sim_control, stateA, stateB, dt=0.01)
            else:
                actuator.step(sim_state, sim_control, stateB, stateA, dt=0.01)


# ---------------------------------------------------------------------------
# USD Parser
# ---------------------------------------------------------------------------


class MockAttribute:
    def __init__(self, value=None, name=""):
        self._value = value
        self._name = name

    def HasAuthoredValue(self):
        return self._value is not None

    def Get(self):
        return self._value

    def GetName(self):
        return self._name


class MockRelationship:
    def __init__(self, targets=None):
        self._targets = targets or []

    def GetTargets(self):
        return self._targets


class MockPrim:
    def __init__(self, type_name="", attributes=None, relationships=None, schemas=None):
        self._type_name = type_name
        self._attributes = attributes or {}
        self._relationships = relationships or {}
        self._schemas = schemas or []

    def GetTypeName(self):
        return self._type_name

    def GetAttribute(self, name):
        return self._attributes.get(name)

    def GetAttributes(self):
        return [MockAttribute(attr._value, name) for name, attr in self._attributes.items()]

    def GetRelationship(self, name):
        return self._relationships.get(name)

    def GetAppliedSchemas(self):
        return self._schemas


class TestActuatorParser(unittest.TestCase):
    """Tests for USD parsing with the new composed model."""

    def test_parse_pd_actuator_prim(self):
        from newton_actuators import PDController, parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:kd": MockAttribute(10.0, "newton:actuator:kd"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
            schemas=["PDControllerAPI"],
        )

        result = parse_actuator_prim(prim)

        self.assertIsNotNone(result)
        self.assertEqual(result.controller_class, PDController)
        self.assertEqual(result.target_paths, ["/World/Robot/Joint1"])
        self.assertEqual(result.controller_kwargs.get("kp"), 100.0)
        self.assertEqual(result.controller_kwargs.get("kd"), 10.0)
        self.assertEqual(len(result.component_specs), 0)

    def test_parse_delayed_pd_actuator_prim(self):
        from newton_actuators import PDController, parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(50.0, "newton:actuator:kp"),
                "newton:actuator:delay": MockAttribute(5, "newton:actuator:delay"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
            schemas=["PDControllerAPI", "DelayAPI"],
        )

        result = parse_actuator_prim(prim)
        self.assertIsNotNone(result)
        self.assertEqual(result.controller_class, PDController)
        self.assertEqual(result.controller_kwargs.get("kp"), 50.0)
        self.assertEqual(len(result.component_specs), 1)
        delay_cls, delay_kwargs = result.component_specs[0]
        from newton_actuators import Delay
        self.assertEqual(delay_cls, Delay)
        self.assertEqual(delay_kwargs.get("delay"), 5)

    def test_parse_pid_actuator_prim(self):
        from newton_actuators import PIDController, parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:ki": MockAttribute(5.0, "newton:actuator:ki"),
                "newton:actuator:kd": MockAttribute(10.0, "newton:actuator:kd"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
            schemas=["PIDControllerAPI"],
        )

        result = parse_actuator_prim(prim)
        self.assertIsNotNone(result)
        self.assertEqual(result.controller_class, PIDController)
        self.assertEqual(result.controller_kwargs.get("kp"), 100.0)
        self.assertEqual(result.controller_kwargs.get("ki"), 5.0)

    def test_parse_multi_target_actuator(self):
        from newton_actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:transmission": MockAttribute([0.5, 0.3, 0.2], "newton:actuator:transmission"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(
                    ["/World/Robot/Joint1", "/World/Robot/Joint2", "/World/Robot/Joint3"]
                ),
            },
            schemas=["PDControllerAPI"],
        )

        result = parse_actuator_prim(prim)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.target_paths), 3)
        self.assertEqual(result.transmission, [0.5, 0.3, 0.2])

    def test_parse_non_actuator_prim_returns_none(self):
        from newton_actuators import parse_actuator_prim

        prim = MockPrim(type_name="Mesh", attributes={}, relationships={})
        result = parse_actuator_prim(prim)
        self.assertIsNone(result)

    def test_parse_actuator_without_targets_returns_none(self):
        from newton_actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={"newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp")},
            relationships={},
            schemas=["PDControllerAPI"],
        )
        result = parse_actuator_prim(prim)
        self.assertIsNone(result)

    def test_parse_dc_motor_actuator_prim(self):
        from newton_actuators import PDController, parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:kd": MockAttribute(10.0, "newton:actuator:kd"),
                "newton:actuator:maxForce": MockAttribute(200.0, "newton:actuator:maxForce"),
                "newton:actuator:saturationEffort": MockAttribute(150.0, "newton:actuator:saturationEffort"),
                "newton:actuator:velocityLimit": MockAttribute(10.0, "newton:actuator:velocityLimit"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
            schemas=["PDControllerAPI", "DCMotorAPI"],
        )

        result = parse_actuator_prim(prim)
        self.assertIsNotNone(result)
        self.assertEqual(result.controller_class, PDController)
        self.assertEqual(result.controller_kwargs.get("kp"), 100.0)
        self.assertEqual(result.controller_kwargs.get("kd"), 10.0)
        self.assertTrue(len(result.component_specs) > 0)

    def test_parse_dc_motor_velocity_limit_zero_raises(self):
        from newton_actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:saturationEffort": MockAttribute(150.0, "newton:actuator:saturationEffort"),
                "newton:actuator:velocityLimit": MockAttribute(0.0, "newton:actuator:velocityLimit"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
            schemas=["PDControllerAPI", "DCMotorAPI"],
        )

        with self.assertRaises(ValueError):
            parse_actuator_prim(prim)

    def test_parse_dc_motor_velocity_limit_negative_raises(self):
        from newton_actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:saturationEffort": MockAttribute(150.0, "newton:actuator:saturationEffort"),
                "newton:actuator:velocityLimit": MockAttribute(-5.0, "newton:actuator:velocityLimit"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(["/World/Robot/Joint1"]),
            },
            schemas=["PDControllerAPI", "DCMotorAPI"],
        )

        with self.assertRaises(ValueError):
            parse_actuator_prim(prim)


# ---------------------------------------------------------------------------
# Neural Network Controllers (torch-dependent)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestNetMLPController(unittest.TestCase):
    """Tests for NetMLPController + Clamp (replaces old ActuatorNetMLP)."""

    def setUp(self):
        wp.init()
        import torch

        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.wp_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _make_mlp(self, input_dim, hidden=32):
        return self.torch.nn.Sequential(
            self.torch.nn.Linear(input_dim, hidden),
            self.torch.nn.ELU(),
            self.torch.nn.Linear(hidden, 1),
        )

    def test_creation(self):
        from newton_actuators import Clamp, NetMLPController

        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=2)

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetMLPController(network=network),
            clamping=[Clamp(max_force=wp.array([50.0, 50.0], dtype=wp.float32, device=self.wp_device))],
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertFalse(actuator.is_graphable())

    def test_state(self):
        from newton_actuators import NetMLPController

        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=6)

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetMLPController(network=network, input_idx=[0, 1, 2]),
        )

        state = actuator.state()
        self.assertIsNotNone(state)
        ctrl_state = state.controller_state
        self.assertEqual(ctrl_state.pos_error_history.shape, (3, 3))
        self.assertEqual(ctrl_state.vel_history.shape, (3, 3))

    def test_step_runs(self):
        from newton_actuators import Clamp, NetMLPController

        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=2)

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetMLPController(network=network),
            clamping=[Clamp(max_force=wp.array([1000.0, 1000.0], dtype=wp.float32, device=self.wp_device))],
        )

        stateA = actuator.state()
        stateB = actuator.state()

        sim_state = MockSimState(
            joint_q=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_qd=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0, 2.0], dtype=wp.float32, device=self.wp_device),
            joint_target_vel=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_act=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_f=wp.zeros(num_dofs, dtype=wp.float32, device=self.wp_device),
        )

        actuator.step(sim_state, sim_control, stateA, stateB)
        forces = sim_control.joint_f.numpy()
        self.assertEqual(forces.shape, (2,))

    def test_clamping(self):
        from newton_actuators import Clamp, NetMLPController

        num_dofs = 1
        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)

        network = self.torch.nn.Sequential(self.torch.nn.Linear(2, 1, bias=True))
        with self.torch.no_grad():
            network[0].weight.fill_(0.0)
            network[0].bias.fill_(999.0)

        max_force_val = 10.0
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetMLPController(network=network),
            clamping=[Clamp(max_force=wp.array([max_force_val], dtype=wp.float32, device=self.wp_device))],
        )

        stateA = actuator.state()
        stateB = actuator.state()

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_qd=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32, device=self.wp_device),
            joint_target_vel=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_act=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_f=wp.zeros(num_dofs, dtype=wp.float32, device=self.wp_device),
        )

        actuator.step(sim_state, sim_control, stateA, stateB)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, max_force_val, places=3)

    def test_raw_output(self):
        """Network output is passed through without scaling."""
        from newton_actuators import NetMLPController

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)

        network = self.torch.nn.Sequential(self.torch.nn.Linear(2, 1, bias=True))
        with self.torch.no_grad():
            network[0].weight.fill_(0.0)
            network[0].bias.fill_(5.0)

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetMLPController(network=network),
        )

        stateA = actuator.state()
        stateB = actuator.state()

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_qd=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_target_vel=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_act=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_f=wp.zeros(1, dtype=wp.float32, device=self.wp_device),
        )

        actuator.step(sim_state, sim_control, stateA, stateB)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 5.0, places=3)

    def test_history_persistence(self):
        from newton_actuators import NetMLPController

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=4)

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetMLPController(network=network, input_idx=[0, 1]),
        )

        stateA = actuator.state()
        stateB = actuator.state()

        for step in range(3):
            sim_state = MockSimState(
                joint_q=wp.array([float(step)], dtype=wp.float32, device=self.wp_device),
                joint_qd=wp.array([float(step) * 0.1], dtype=wp.float32, device=self.wp_device),
            )
            sim_control = MockSimControl(
                joint_target_pos=wp.array([float(step + 1)], dtype=wp.float32, device=self.wp_device),
                joint_target_vel=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
                joint_act=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
                joint_f=wp.zeros(1, dtype=wp.float32, device=self.wp_device),
            )
            if step % 2 == 0:
                actuator.step(sim_state, sim_control, stateA, stateB)
                active = stateB
            else:
                actuator.step(sim_state, sim_control, stateB, stateA)
                active = stateA

        ctrl_state = active.controller_state
        self.assertFalse(self.torch.all(ctrl_state.pos_error_history == 0.0).item())
        self.assertFalse(self.torch.all(ctrl_state.vel_history == 0.0).item())

    def test_invalid_input_order(self):
        from newton_actuators import NetMLPController

        with self.assertRaises(ValueError):
            NetMLPController(network=self._make_mlp(2), input_order="invalid")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestNetLSTMController(unittest.TestCase):
    """Tests for NetLSTMController + Clamp (replaces old ActuatorNetLSTM)."""

    def setUp(self):
        wp.init()
        import torch

        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.wp_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _make_lstm(self, hidden_size=8, num_layers=1):
        import torch

        class _SimpleLSTMNet(torch.nn.Module):
            def __init__(self, input_size=2, hidden_size=8, output_size=1, num_layers=1):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.decoder = torch.nn.Linear(hidden_size, output_size)

            def forward(self, x, hc):
                lstm_out, (h_new, c_new) = self.lstm(x, hc)
                output = self.decoder(lstm_out[:, -1, :])
                return output, (h_new, c_new)

        return _SimpleLSTMNet(input_size=2, hidden_size=hidden_size, num_layers=num_layers)

    def test_creation(self):
        from newton_actuators import Clamp, NetLSTMController

        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm()

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetLSTMController(network=network),
            clamping=[Clamp(max_force=wp.array([50.0, 50.0], dtype=wp.float32, device=self.wp_device))],
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertFalse(actuator.is_graphable())

    def test_state(self):
        from newton_actuators import NetLSTMController

        hidden_size = 16
        num_layers = 2
        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm(hidden_size=hidden_size, num_layers=num_layers)

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetLSTMController(network=network),
        )

        state = actuator.state()
        ctrl_state = state.controller_state
        self.assertEqual(ctrl_state.hidden.shape, (num_layers, 3, hidden_size))
        self.assertEqual(ctrl_state.cell.shape, (num_layers, 3, hidden_size))

    def test_step_runs(self):
        from newton_actuators import Clamp, NetLSTMController

        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm()

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetLSTMController(network=network),
            clamping=[Clamp(max_force=wp.array([1000.0, 1000.0], dtype=wp.float32, device=self.wp_device))],
        )

        stateA = actuator.state()
        stateB = actuator.state()

        sim_state = MockSimState(
            joint_q=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_qd=wp.array([1.0, -1.0], dtype=wp.float32, device=self.wp_device),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0, 2.0], dtype=wp.float32, device=self.wp_device),
            joint_target_vel=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_act=wp.array([0.0, 0.0], dtype=wp.float32, device=self.wp_device),
            joint_f=wp.zeros(num_dofs, dtype=wp.float32, device=self.wp_device),
        )

        actuator.step(sim_state, sim_control, stateA, stateB)
        forces = sim_control.joint_f.numpy()
        self.assertEqual(forces.shape, (2,))

    def test_clamping(self):
        from newton_actuators import Clamp, NetLSTMController

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)

        network = self._make_lstm(hidden_size=4)
        with self.torch.no_grad():
            network.decoder.weight.fill_(0.0)
            network.decoder.bias.fill_(500.0)

        max_force_val = 10.0
        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetLSTMController(network=network),
            clamping=[Clamp(max_force=wp.array([max_force_val], dtype=wp.float32, device=self.wp_device))],
        )

        stateA = actuator.state()
        stateB = actuator.state()

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_qd=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32, device=self.wp_device),
            joint_target_vel=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_act=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_f=wp.zeros(1, dtype=wp.float32, device=self.wp_device),
        )

        actuator.step(sim_state, sim_control, stateA, stateB)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, max_force_val, places=3)

    def test_state_evolves(self):
        from newton_actuators import NetLSTMController

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm()

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetLSTMController(network=network),
        )

        stateA = actuator.state()
        stateB = actuator.state()

        self.assertTrue(self.torch.all(stateA.controller_state.hidden == 0.0).item())
        self.assertTrue(self.torch.all(stateA.controller_state.cell == 0.0).item())

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_qd=wp.array([1.0], dtype=wp.float32, device=self.wp_device),
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32, device=self.wp_device),
            joint_target_vel=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_act=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            joint_f=wp.zeros(1, dtype=wp.float32, device=self.wp_device),
        )

        actuator.step(sim_state, sim_control, stateA, stateB)

        self.assertFalse(self.torch.all(stateB.controller_state.hidden == 0.0).item())
        self.assertFalse(self.torch.all(stateB.controller_state.cell == 0.0).item())

    def test_multi_step_different_outputs(self):
        from newton_actuators import NetLSTMController

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm()

        actuator = Actuator(
            input_indices=indices,
            output_indices=indices,
            controller=NetLSTMController(network=network),
        )

        stateA = actuator.state()
        stateB = actuator.state()
        forces = []

        for step in range(5):
            sim_state = MockSimState(
                joint_q=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
                joint_qd=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
            )
            sim_control = MockSimControl(
                joint_target_pos=wp.array([1.0], dtype=wp.float32, device=self.wp_device),
                joint_target_vel=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
                joint_act=wp.array([0.0], dtype=wp.float32, device=self.wp_device),
                joint_f=wp.zeros(1, dtype=wp.float32, device=self.wp_device),
            )

            if step % 2 == 0:
                actuator.step(sim_state, sim_control, stateA, stateB)
            else:
                actuator.step(sim_state, sim_control, stateB, stateA)
            forces.append(sim_control.joint_f.numpy()[0])

        self.assertFalse(
            all(abs(f - forces[0]) < 1e-6 for f in forces[1:]),
            "LSTM should produce varying outputs across steps due to hidden state",
        )


if __name__ == "__main__":
    unittest.main()
