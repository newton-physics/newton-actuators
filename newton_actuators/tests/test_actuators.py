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

import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton_actuators import (
    Actuator,
    ActuatorDelayedPD,
    ActuatorPD,
    ActuatorPID,
)


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
            constant_force=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertIsNone(actuator.state())
        self.assertFalse(actuator.is_stateful())

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
            constant_force=wp.array([0.0, 0.0, 0.0], dtype=wp.float32),
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
            constant_force=wp.array([0.0, 0.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())

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
            constant_force=wp.array([0.0, 0.0], dtype=wp.float32),
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
            constant_force=wp.array([0.0], dtype=wp.float32),
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
        self.assertAlmostEqual(force_history[3], target_history[0], places=5,
                               msg=f"Step 3: expected force={target_history[0]}, got {force_history[3]}")

        # For step 4: should use target from step 1 (T1=20)
        self.assertAlmostEqual(force_history[4], target_history[1], places=5,
                               msg=f"Step 4: expected force={target_history[1]}, got {force_history[4]}")

        # For step 5: should use target from step 2 (T2=30)
        self.assertAlmostEqual(force_history[5], target_history[2], places=5,
                               msg=f"Step 5: expected force={target_history[2]}, got {force_history[5]}")


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
            constant_force=wp.array([0.0, 0.0], dtype=wp.float32),
        )
        self.assertIsInstance(actuator, Actuator)
        self.assertTrue(actuator.is_stateful())

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
            constant_force=wp.array([0.0, 0.0], dtype=wp.float32),
        )

        state = actuator.state()
        self.assertIsInstance(state, ActuatorPID.State)
        self.assertEqual(state.integral.shape[0], num_dofs)

        np.testing.assert_array_equal(state.integral.numpy(), [0.0, 0.0])


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
        from newton_actuators import parse_actuator_prim, ActuatorPD

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
        from newton_actuators import parse_actuator_prim, ActuatorDelayedPD

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
        from newton_actuators import parse_actuator_prim, ActuatorPID

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
                "newton:actuator:target": MockRelationship([
                    "/World/Robot/Joint1",
                    "/World/Robot/Joint2",
                    "/World/Robot/Joint3",
                ]),
            },
            schemas=["PDControllerAPI"],
        )

        result = parse_actuator_prim(prim)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.target_paths), 3)
        self.assertEqual(result.transmission, [0.5, 0.3, 0.2])

    def test_parse_multiple_actuators(self):
        """Test parsing multiple actuator prims."""
        from newton_actuators import parse_actuator_prim, ActuatorPD

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


if __name__ == "__main__":
    unittest.main()
