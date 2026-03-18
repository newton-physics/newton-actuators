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

"""Unit tests for actuator implementations."""

import importlib.util
import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton_actuators import (
    Actuator,
    ActuatorDCMotor,
    ActuatorDelayedPD,
    ActuatorPD,
    ActuatorPID,
    ActuatorRemotizedPD,
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


class TestActuatorPD(unittest.TestCase):
    """Tests for ActuatorPD."""

    def setUp(self):
        wp.init()

    def test_pd_actuator_creation(self):
        """Test that ActuatorPD can be created with valid parameters."""
        indices = wp.array([0, 1, 2], dtype=wp.uint32)
        actuator = ActuatorPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0, 100.0], dtype=wp.float32),
            kd=wp.array([10.0, 10.0, 10.0], dtype=wp.float32),
            max_force=wp.array([50.0, 50.0, 50.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertIsNone(actuator.state())
        self.assertFalse(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

    def test_pd_actuator_step(self):
        """Test that ActuatorPD.step() computes correct forces."""
        num_dofs = 3
        indices = wp.array([0, 1, 2], dtype=wp.uint32)

        # Create actuator
        actuator = ActuatorPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0, 100.0], dtype=wp.float32),
            kd=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
            max_force=wp.array([1000.0, 1000.0, 1000.0], dtype=wp.float32),
        )

        # Create mock state and control
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

        # Run step
        actuator.step(sim_state, sim_control, None, None)
        # Check forces: f = kp * (target - current) = 100 * [1, 2, 3] = [100, 200, 300]
        forces = sim_control.joint_f.numpy()
        np.testing.assert_allclose(forces, [100.0, 200.0, 300.0], rtol=1e-5)

    def test_pd_actuator_resolve_arguments(self):
        """Test that resolve_arguments fills defaults correctly."""
        resolved = ActuatorPD.resolve_arguments({"kp": 50.0})
        self.assertEqual(resolved["kp"], 50.0)
        self.assertEqual(resolved["kd"], 0.0)
        self.assertEqual(resolved["constant_force"], 0.0)


class TestActuatorDelayedPD(unittest.TestCase):
    """Tests for ActuatorDelayedPD."""

    def setUp(self):
        wp.init()

    def test_delayed_pd_creation(self):
        """Test that ActuatorDelayedPD can be created with valid parameters."""
        indices = wp.array([0, 1], dtype=wp.uint32)
        actuator = ActuatorDelayedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            kd=wp.array([10.0, 10.0], dtype=wp.float32),
            delay=5,
            max_force=wp.array([50.0, 50.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

    def test_delayed_pd_state(self):
        """Test that ActuatorDelayedPD.state() returns properly initialized state."""
        delay = 5
        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32)

        actuator = ActuatorDelayedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            kd=wp.array([10.0, 10.0], dtype=wp.float32),
            delay=delay,
            max_force=wp.array([50.0, 50.0], dtype=wp.float32),
        )

        state = actuator.state()
        self.assertIsInstance(state, ActuatorDelayedPD.State)
        self.assertEqual(state.write_idx, delay - 1)
        self.assertFalse(state.is_filled)
        self.assertEqual(state.buffer_pos.shape, (delay, num_dofs))
        self.assertEqual(state.buffer_vel.shape, (delay, num_dofs))
        self.assertEqual(state.buffer_act.shape, (delay, num_dofs))

    def test_delayed_pd_resolve_arguments_requires_delay(self):
        """Test that resolve_arguments raises error if delay not provided."""
        with self.assertRaises(ValueError):
            ActuatorDelayedPD.resolve_arguments({"kp": 50.0})

    def test_delayed_pd_delay_behavior(self):
        """Test that ActuatorDelayedPD correctly delays targets by N steps."""
        delay = 3
        num_dofs = 1
        indices = wp.array([0], dtype=wp.uint32)

        # Create actuator with kp=1, kd=0 so force = target_pos - current_pos
        actuator = ActuatorDelayedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            delay=delay,
            max_force=wp.array([1000.0], dtype=wp.float32),
        )

        # Create double-buffered states
        stateA = actuator.state()
        stateB = actuator.state()

        # Track which targets we send and which forces we get
        target_history = []
        force_history = []

        # Run for delay + 3 steps
        for step in range(delay + 3):
            target_value = float(step + 1) * 10.0  # T0=10, T1=20, T2=30, ...
            target_history.append(target_value)

            # Current state is always at position 0
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

            # Alternate states (double buffering)
            if step % 2 == 0:
                current, next_state = stateA, stateB
            else:
                current, next_state = stateB, stateA

            actuator.step(sim_state, sim_control, current, next_state, dt=0.01)
            force_history.append(sim_control.joint_f.numpy()[0])

        # For steps 0, 1, 2: is_filled=False, force should be 0
        for i in range(delay):
            self.assertEqual(force_history[i], 0.0, f"Step {i}: expected 0 force during fill phase")

        # For step 3: should use target from step 0 (T0=10)
        # force = kp * (delayed_target - current_pos) = 1 * (10 - 0) = 10
        self.assertAlmostEqual(
            force_history[3],
            target_history[0],
            places=5,
            msg=f"Step 3: expected force={target_history[0]}, got {force_history[3]}",
        )

        # For step 4: should use target from step 1 (T1=20)
        self.assertAlmostEqual(
            force_history[4],
            target_history[1],
            places=5,
            msg=f"Step 4: expected force={target_history[1]}, got {force_history[4]}",
        )

        # For step 5: should use target from step 2 (T2=30)
        self.assertAlmostEqual(
            force_history[5],
            target_history[2],
            places=5,
            msg=f"Step 5: expected force={target_history[2]}, got {force_history[5]}",
        )


class TestActuatorPID(unittest.TestCase):
    """Tests for ActuatorPID."""

    def setUp(self):
        wp.init()

    def test_pid_actuator_creation(self):
        """Test that ActuatorPID can be created with valid parameters."""
        indices = wp.array([0, 1], dtype=wp.uint32)
        actuator = ActuatorPID(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            ki=wp.array([10.0, 10.0], dtype=wp.float32),
            kd=wp.array([5.0, 5.0], dtype=wp.float32),
            max_force=wp.array([50.0, 50.0], dtype=wp.float32),
            integral_max=wp.array([10.0, 10.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())

    def test_pid_actuator_state(self):
        """Test that ActuatorPID.state() returns properly initialized state."""
        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32)

        actuator = ActuatorPID(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            ki=wp.array([10.0, 10.0], dtype=wp.float32),
            kd=wp.array([5.0, 5.0], dtype=wp.float32),
            max_force=wp.array([50.0, 50.0], dtype=wp.float32),
            integral_max=wp.array([10.0, 10.0], dtype=wp.float32),
        )

        state = actuator.state()
        self.assertIsInstance(state, ActuatorPID.State)
        self.assertEqual(state.integral.shape[0], num_dofs)

        np.testing.assert_array_equal(state.integral.numpy(), [0.0, 0.0])


class TestActuatorDCMotor(unittest.TestCase):
    """Tests for ActuatorDCMotor."""

    def setUp(self):
        wp.init()

    def test_dc_motor_creation(self):
        """Test that ActuatorDCMotor can be created with valid parameters."""
        indices = wp.array([0, 1], dtype=wp.uint32)
        actuator = ActuatorDCMotor(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            kd=wp.array([10.0, 10.0], dtype=wp.float32),
            max_force=wp.array([50.0, 50.0], dtype=wp.float32),
            saturation_effort=wp.array([80.0, 80.0], dtype=wp.float32),
            velocity_limit=wp.array([10.0, 10.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertFalse(actuator.is_stateful())
        self.assertIsNone(actuator.state())
        self.assertTrue(actuator.is_graphable())

    def test_dc_motor_resolve_arguments_requires_velocity_limit(self):
        """Test that resolve_arguments raises error if velocity_limit not provided."""
        with self.assertRaises(ValueError):
            ActuatorDCMotor.resolve_arguments({"kp": 50.0})

    def test_dc_motor_resolve_arguments(self):
        """Test that resolve_arguments fills defaults correctly."""
        resolved = ActuatorDCMotor.resolve_arguments({"kp": 50.0, "velocity_limit": 10.0})
        self.assertEqual(resolved["kp"], 50.0)
        self.assertEqual(resolved["kd"], 0.0)
        self.assertEqual(resolved["velocity_limit"], 10.0)

    def test_dc_motor_zero_velocity_full_torque(self):
        """At zero velocity, DC motor can produce full torque up to saturation/max_force."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = ActuatorDCMotor(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            max_force=wp.array([200.0], dtype=wp.float32),
            saturation_effort=wp.array([150.0], dtype=wp.float32),
            velocity_limit=wp.array([10.0], dtype=wp.float32),
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
        # PD force = 100*(1-0) = 100, at v=0: max_torque = clamp(150*(1-0), 0, 200) = 150
        # 100 < 150, so not clipped
        self.assertAlmostEqual(force, 100.0, places=3)

    def test_dc_motor_velocity_reduces_max_torque(self):
        """At high velocity, available torque in direction of motion is reduced."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = ActuatorDCMotor(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1000.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            max_force=wp.array([200.0], dtype=wp.float32),
            saturation_effort=wp.array([100.0], dtype=wp.float32),
            velocity_limit=wp.array([10.0], dtype=wp.float32),
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([5.0], dtype=wp.float32),  # half of v_max
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )

        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        # PD force = 1000*(1-0) = 1000 (large positive)
        # At v=5, v_max=10: max_torque = clamp(100*(1 - 5/10), 0, 200) = clamp(50, 0, 200) = 50
        # min_torque = clamp(100*(-1 - 5/10), -200, 0) = clamp(-150, -200, 0) = -150
        # Force clamped to 50
        self.assertAlmostEqual(force, 50.0, places=3)

    def test_dc_motor_at_velocity_limit(self):
        """At v_max, no torque can be produced in direction of motion."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = ActuatorDCMotor(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1000.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            max_force=wp.array([200.0], dtype=wp.float32),
            saturation_effort=wp.array([100.0], dtype=wp.float32),
            velocity_limit=wp.array([10.0], dtype=wp.float32),
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([10.0], dtype=wp.float32),  # exactly at v_max
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )

        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        # At v=v_max: max_torque = clamp(100*(1 - 10/10), 0, 200) = clamp(0, 0, 200) = 0
        self.assertAlmostEqual(force, 0.0, places=3)

    def test_dc_motor_negative_velocity_increases_positive_limit(self):
        """Negative velocity allows more torque in the positive direction."""
        indices = wp.array([0], dtype=wp.uint32)
        actuator = ActuatorDCMotor(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1000.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            max_force=wp.array([200.0], dtype=wp.float32),
            saturation_effort=wp.array([100.0], dtype=wp.float32),
            velocity_limit=wp.array([10.0], dtype=wp.float32),
        )

        sim_state = MockSimState(
            joint_q=wp.array([0.0], dtype=wp.float32),
            joint_qd=wp.array([-5.0], dtype=wp.float32),  # moving backwards
        )
        sim_control = MockSimControl(
            joint_target_pos=wp.array([1.0], dtype=wp.float32),
            joint_target_vel=wp.array([0.0], dtype=wp.float32),
            joint_act=wp.array([0.0], dtype=wp.float32),
            joint_f=wp.zeros(1, dtype=wp.float32),
        )

        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        # At v=-5: max_torque = clamp(100*(1 - (-5)/10), 0, 200) = clamp(150, 0, 200) = 150
        # PD force = 1000, clamped to 150
        self.assertAlmostEqual(force, 150.0, places=3)


class TestActuatorRemotizedPD(unittest.TestCase):
    """Tests for ActuatorRemotizedPD."""

    def setUp(self):
        wp.init()

    def _make_lookup(self):
        """Create a simple lookup table: torque limit varies from 10 to 50 over angles -1 to 1."""
        angles = wp.array([-1.0, 0.0, 1.0], dtype=wp.float32)
        torques = wp.array([10.0, 30.0, 50.0], dtype=wp.float32)
        return angles, torques

    def test_remotized_pd_creation(self):
        """Test that ActuatorRemotizedPD can be created with valid parameters."""
        indices = wp.array([0, 1], dtype=wp.uint32)
        angles, torques = self._make_lookup()

        actuator = ActuatorRemotizedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0, 100.0], dtype=wp.float32),
            kd=wp.array([10.0, 10.0], dtype=wp.float32),
            delay=3,
            lookup_angles=angles,
            lookup_torques=torques,
        )
        self.assertIsInstance(actuator, ActuatorDelayedPD)
        self.assertTrue(actuator.is_stateful())
        self.assertTrue(actuator.is_graphable())
        self.assertEqual(actuator.lookup_size, 3)

    def test_remotized_pd_resolve_arguments_requires_delay_and_lookup(self):
        """Test that resolve_arguments raises errors for missing required args."""
        with self.assertRaises(ValueError):
            ActuatorRemotizedPD.resolve_arguments({"kp": 50.0})
        with self.assertRaises(ValueError):
            ActuatorRemotizedPD.resolve_arguments({"kp": 50.0, "delay": 3})

    def test_remotized_pd_angle_dependent_clipping(self):
        """Test that torque is clamped based on the lookup table at the current joint angle."""
        delay = 2
        indices = wp.array([0], dtype=wp.uint32)
        # Lookup: at angle=0, max_torque=30
        angles, torques = self._make_lookup()

        actuator = ActuatorRemotizedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1000.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            delay=delay,
            lookup_angles=angles,
            lookup_torques=torques,
        )

        stateA = actuator.state()
        stateB = actuator.state()

        # Fill the delay buffer first (delay=2 steps to fill)
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
        # PD force = 1000*(1-0) = 1000, but at angle=0.0 the lookup gives max_torque=30
        # So force is clamped to 30
        self.assertAlmostEqual(force, 30.0, places=3)

    def test_remotized_pd_different_angles(self):
        """Test that lookup interpolation works at different joint angles."""
        delay = 2
        indices = wp.array([0], dtype=wp.uint32)
        angles, torques = self._make_lookup()

        actuator = ActuatorRemotizedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([1000.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            delay=delay,
            lookup_angles=angles,
            lookup_torques=torques,
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
            # PD force = 1000*10 = 10000, always clamped to lookup limit
            self.assertAlmostEqual(
                force, expected_limit, places=2, msg=f"At angle={test_angle}, expected limit={expected_limit}"
            )

    def test_remotized_pd_no_force_during_fill(self):
        """Test that no force is applied while the delay buffer is filling."""
        delay = 3
        indices = wp.array([0], dtype=wp.uint32)
        angles, torques = self._make_lookup()

        actuator = ActuatorRemotizedPD(
            input_indices=indices,
            output_indices=indices,
            kp=wp.array([100.0], dtype=wp.float32),
            kd=wp.array([0.0], dtype=wp.float32),
            delay=delay,
            lookup_angles=angles,
            lookup_torques=torques,
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
            self.assertEqual(force, 0.0, f"Step {step}: expected 0 force during fill phase")


class MockAttribute:
    """Mock USD attribute for testing."""

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
    """Mock USD relationship for testing."""

    def __init__(self, targets=None):
        self._targets = targets or []

    def GetTargets(self):
        return self._targets


class MockPrim:
    """Mock USD prim for testing ActuatorParser."""

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
        """Return list of attribute objects with names."""
        return [MockAttribute(attr._value, name) for name, attr in self._attributes.items()]

    def GetRelationship(self, name):
        return self._relationships.get(name)

    def GetAppliedSchemas(self):
        return self._schemas


class TestActuatorParser(unittest.TestCase):
    """Tests for ActuatorParser and USD parsing utilities."""

    def test_parse_pd_actuator_prim(self):
        """Test parsing a PD actuator prim."""
        from newton_actuators import ActuatorPD, parse_actuator_prim

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
        self.assertEqual(result.actuator_class, ActuatorPD)
        self.assertEqual(result.target_paths, ["/World/Robot/Joint1"])
        self.assertEqual(result.kwargs.get("kp"), 100.0)
        self.assertEqual(result.kwargs.get("kd"), 10.0)

    def test_parse_delayed_pd_actuator_prim(self):
        """Test parsing a Delayed PD actuator prim."""
        from newton_actuators import ActuatorDelayedPD, parse_actuator_prim

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
        self.assertEqual(result.actuator_class, ActuatorDelayedPD)
        self.assertEqual(result.kwargs.get("kp"), 50.0)
        self.assertEqual(result.kwargs.get("delay"), 5)

    def test_parse_pid_actuator_prim(self):
        """Test parsing a PID actuator prim."""
        from newton_actuators import ActuatorPID, parse_actuator_prim

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
        self.assertEqual(result.actuator_class, ActuatorPID)
        self.assertEqual(result.kwargs.get("kp"), 100.0)
        self.assertEqual(result.kwargs.get("ki"), 5.0)

    def test_parse_multi_target_actuator(self):
        """Test parsing an actuator with multiple targets."""
        from newton_actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={
                "newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp"),
                "newton:actuator:transmission": MockAttribute([0.5, 0.3, 0.2], "newton:actuator:transmission"),
            },
            relationships={
                "newton:actuator:target": MockRelationship(
                    [
                        "/World/Robot/Joint1",
                        "/World/Robot/Joint2",
                        "/World/Robot/Joint3",
                    ]
                ),
            },
            schemas=["PDControllerAPI"],
        )

        result = parse_actuator_prim(prim)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.target_paths), 3)
        self.assertEqual(result.transmission, [0.5, 0.3, 0.2])

    def test_parse_multiple_actuators(self):
        """Test parsing multiple actuator prims."""
        from newton_actuators import ActuatorPD, parse_actuator_prim

        prim1 = MockPrim(
            type_name="Actuator",
            attributes={"newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp")},
            relationships={"newton:actuator:target": MockRelationship(["/World/Robot/Joint1"])},
            schemas=["PDControllerAPI"],
        )
        prim2 = MockPrim(
            type_name="Actuator",
            attributes={"newton:actuator:kp": MockAttribute(200.0, "newton:actuator:kp")},
            relationships={"newton:actuator:target": MockRelationship(["/World/Robot/Joint2"])},
            schemas=["PDControllerAPI"],
        )

        result1 = parse_actuator_prim(prim1)
        result2 = parse_actuator_prim(prim2)

        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertEqual(result1.actuator_class, ActuatorPD)
        self.assertEqual(result1.target_paths, ["/World/Robot/Joint1"])
        self.assertEqual(result1.kwargs.get("kp"), 100.0)
        self.assertEqual(result2.target_paths, ["/World/Robot/Joint2"])
        self.assertEqual(result2.kwargs.get("kp"), 200.0)

    def test_parse_non_actuator_prim_returns_none(self):
        """Test that non-Actuator prims return None."""
        from newton_actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Mesh",  # Not an Actuator
            attributes={},
            relationships={},
            schemas=[],
        )

        result = parse_actuator_prim(prim)
        self.assertIsNone(result)

    def test_parse_actuator_without_targets_returns_none(self):
        """Test that actuator without targets returns None."""
        from newton_actuators import parse_actuator_prim

        prim = MockPrim(
            type_name="Actuator",
            attributes={"newton:actuator:kp": MockAttribute(100.0, "newton:actuator:kp")},
            relationships={},  # No targets
            schemas=["PDControllerAPI"],
        )

        result = parse_actuator_prim(prim)
        self.assertIsNone(result)

    def test_parse_dc_motor_actuator_prim(self):
        """Test parsing a DC motor actuator prim with PD + saturation params."""
        from newton_actuators import ActuatorDCMotor, parse_actuator_prim

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
        )

        result = parse_actuator_prim(prim)

        self.assertIsNotNone(result)
        self.assertEqual(result.actuator_class, ActuatorDCMotor)
        self.assertEqual(result.kwargs.get("kp"), 100.0)
        self.assertEqual(result.kwargs.get("kd"), 10.0)
        self.assertEqual(result.kwargs.get("max_force"), 200.0)
        self.assertEqual(result.kwargs.get("saturation_effort"), 150.0)
        self.assertEqual(result.kwargs.get("velocity_limit"), 10.0)

    def test_parse_dc_motor_velocity_limit_zero_raises(self):
        """Test that velocity_limit=0 raises ValueError during parsing."""
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
        )

        with self.assertRaises(ValueError, msg="velocity_limit=0 should raise ValueError"):
            parse_actuator_prim(prim)

    def test_parse_dc_motor_velocity_limit_negative_raises(self):
        """Test that negative velocity_limit raises ValueError during parsing."""
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
        )

        with self.assertRaises(ValueError, msg="negative velocity_limit should raise ValueError"):
            parse_actuator_prim(prim)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestActuatorNetMLP(unittest.TestCase):
    """Tests for ActuatorNetMLP."""

    def setUp(self):
        wp.init()
        import torch

        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.wp_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _make_mlp(self, input_dim, hidden=32):
        """Create a simple MLP: input_dim -> hidden -> 1."""
        return self.torch.nn.Sequential(
            self.torch.nn.Linear(input_dim, hidden),
            self.torch.nn.ELU(),
            self.torch.nn.Linear(hidden, 1),
        )

    def test_mlp_creation(self):
        """Test that ActuatorNetMLP can be created with valid parameters."""
        from newton_actuators import ActuatorNetMLP

        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=2)

        actuator = ActuatorNetMLP(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([50.0, 50.0], dtype=wp.float32, device=self.wp_device),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertFalse(actuator.is_stateful())
        self.assertFalse(actuator.is_graphable())
        self.assertEqual(actuator.history_length, 1)

    def test_mlp_internal_state_shape(self):
        """Test that internal history buffers have the correct shape."""
        from newton_actuators import ActuatorNetMLP

        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=6)  # 3 timesteps * 2 (pos+vel)

        actuator = ActuatorNetMLP(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([50.0, 50.0, 50.0], dtype=wp.float32, device=self.wp_device),
            input_idx=[0, 1, 2],
        )

        self.assertEqual(actuator.pos_error_history.shape, (3, 3))
        self.assertEqual(actuator.vel_history.shape, (3, 3))

    def test_mlp_step_runs(self):
        """Test that step() executes without errors and produces output."""
        from newton_actuators import ActuatorNetMLP

        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=2)

        actuator = ActuatorNetMLP(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([1000.0, 1000.0], dtype=wp.float32, device=self.wp_device),
        )

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

        actuator.step(sim_state, sim_control, None, None)
        forces = sim_control.joint_f.numpy()
        # Network has random weights so we just check it produced some output
        self.assertEqual(forces.shape, (2,))

    def test_mlp_clamping(self):
        """Test that output is clamped to max_force."""
        from newton_actuators import ActuatorNetMLP

        num_dofs = 1
        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)

        # Create a network with large constant output by setting bias high
        network = self.torch.nn.Sequential(self.torch.nn.Linear(2, 1, bias=True))
        with self.torch.no_grad():
            network[0].weight.fill_(0.0)
            network[0].bias.fill_(999.0)

        max_force_val = 10.0
        actuator = ActuatorNetMLP(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([max_force_val], dtype=wp.float32, device=self.wp_device),
            torque_scale=1.0,
        )

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

        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, max_force_val, places=3)

    def test_mlp_torque_scale(self):
        """Test that torque_scale is applied to network output."""
        from newton_actuators import ActuatorNetMLP

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)

        # Network that always outputs 5.0
        network = self.torch.nn.Sequential(self.torch.nn.Linear(2, 1, bias=True))
        with self.torch.no_grad():
            network[0].weight.fill_(0.0)
            network[0].bias.fill_(5.0)

        actuator = ActuatorNetMLP(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([1000.0], dtype=wp.float32, device=self.wp_device),
            torque_scale=3.0,
        )

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

        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, 15.0, places=3)

    def test_mlp_history_persistence(self):
        """Test that internal history buffers accumulate values across steps."""
        from newton_actuators import ActuatorNetMLP

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=4)  # 2 timesteps * 2

        actuator = ActuatorNetMLP(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([1000.0], dtype=wp.float32, device=self.wp_device),
            input_idx=[0, 1],
        )

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
            actuator.step(sim_state, sim_control, None, None)

        # After 3 steps, the internal history should have non-zero values
        self.assertFalse(self.torch.all(actuator.pos_error_history == 0.0).item())
        self.assertFalse(self.torch.all(actuator.vel_history == 0.0).item())

    def test_mlp_invalid_input_order(self):
        """Test that an invalid input_order raises ValueError during step."""
        from newton_actuators import ActuatorNetMLP

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)
        network = self._make_mlp(input_dim=2)

        actuator = ActuatorNetMLP(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([50.0], dtype=wp.float32, device=self.wp_device),
            input_order="invalid",
        )

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

        with self.assertRaises(ValueError):
            actuator.step(sim_state, sim_control, None, None)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestActuatorNetLSTM(unittest.TestCase):
    """Tests for ActuatorNetLSTM."""

    def setUp(self):
        wp.init()
        import torch  # noqa: PLC0415

        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.wp_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _make_lstm(self, hidden_size=8, num_layers=1):
        import torch  # noqa: PLC0415

        class _SimpleLSTMNet(torch.nn.Module):
            """Test LSTM network: LSTM encoder + Linear decoder."""

            def __init__(self, input_size=2, hidden_size=8, output_size=1, num_layers=1):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.decoder = torch.nn.Linear(hidden_size, output_size)

            def forward(self, x, hc):
                lstm_out, (h_new, c_new) = self.lstm(x, hc)
                output = self.decoder(lstm_out[:, -1, :])
                return output, (h_new, c_new)

        return _SimpleLSTMNet(input_size=2, hidden_size=hidden_size, num_layers=num_layers)

    def test_lstm_creation(self):
        """Test that ActuatorNetLSTM can be created with valid parameters."""
        from newton_actuators import ActuatorNetLSTM

        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm()

        actuator = ActuatorNetLSTM(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([50.0, 50.0], dtype=wp.float32, device=self.wp_device),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertFalse(actuator.is_stateful())
        self.assertFalse(actuator.is_graphable())

    def test_lstm_internal_state_shape(self):
        """Test that internal hidden and cell tensors have the correct shape."""
        from newton_actuators import ActuatorNetLSTM

        hidden_size = 16
        num_layers = 2
        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm(hidden_size=hidden_size, num_layers=num_layers)

        actuator = ActuatorNetLSTM(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([50.0, 50.0, 50.0], dtype=wp.float32, device=self.wp_device),
        )

        self.assertEqual(actuator.hidden.shape, (num_layers, 3, hidden_size))
        self.assertEqual(actuator.cell.shape, (num_layers, 3, hidden_size))

    def test_lstm_step_runs(self):
        """Test that step() executes without errors and produces output."""
        from newton_actuators import ActuatorNetLSTM

        num_dofs = 2
        indices = wp.array([0, 1], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm()

        actuator = ActuatorNetLSTM(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([1000.0, 1000.0], dtype=wp.float32, device=self.wp_device),
        )

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

        actuator.step(sim_state, sim_control, None, None)
        forces = sim_control.joint_f.numpy()
        self.assertEqual(forces.shape, (2,))

    def test_lstm_clamping(self):
        """Test that output is clamped to max_force."""
        from newton_actuators import ActuatorNetLSTM

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)

        # Create network with decoder bias set very high
        network = self._make_lstm(hidden_size=4)
        with self.torch.no_grad():
            network.decoder.weight.fill_(0.0)
            network.decoder.bias.fill_(500.0)

        max_force_val = 10.0
        actuator = ActuatorNetLSTM(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([max_force_val], dtype=wp.float32, device=self.wp_device),
        )

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

        actuator.step(sim_state, sim_control, None, None)
        force = sim_control.joint_f.numpy()[0]
        self.assertAlmostEqual(force, max_force_val, places=3)

    def test_lstm_state_evolves(self):
        """Test that internal hidden/cell state changes after a step."""
        from newton_actuators import ActuatorNetLSTM

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm()

        actuator = ActuatorNetLSTM(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([1000.0], dtype=wp.float32, device=self.wp_device),
        )

        # Verify initial state is zero
        self.assertTrue(self.torch.all(actuator.hidden == 0.0).item())
        self.assertTrue(self.torch.all(actuator.cell == 0.0).item())

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

        actuator.step(sim_state, sim_control, None, None)

        # After one step with non-zero input, hidden/cell should be non-zero
        self.assertFalse(self.torch.all(actuator.hidden == 0.0).item())
        self.assertFalse(self.torch.all(actuator.cell == 0.0).item())

    def test_lstm_multi_step_different_outputs(self):
        """Test that LSTM produces different outputs over multiple steps (temporal memory)."""
        from newton_actuators import ActuatorNetLSTM

        indices = wp.array([0], dtype=wp.uint32, device=self.wp_device)
        network = self._make_lstm()

        actuator = ActuatorNetLSTM(
            input_indices=indices,
            output_indices=indices,
            network=network,
            max_force=wp.array([1000.0], dtype=wp.float32, device=self.wp_device),
        )

        forces = []

        for _step in range(5):
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

            actuator.step(sim_state, sim_control, None, None)
            forces.append(sim_control.joint_f.numpy()[0])

        # With the same constant input, LSTM hidden state evolves so outputs differ
        # (at least not all identical due to recurrent dynamics)
        self.assertFalse(
            all(abs(f - forces[0]) < 1e-6 for f in forces[1:]),
            "LSTM should produce varying outputs across steps due to hidden state",
        )


if __name__ == "__main__":
    unittest.main()
