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

"""Warp kernels for actuator computations."""

import warp as wp


@wp.kernel
def pd_controller_kernel(
    current_pos: wp.array(dtype=float),
    current_vel: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    target_vel: wp.array(dtype=float),
    control_input: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    target_indices: wp.array(dtype=wp.uint32),
    output_indices: wp.array(dtype=wp.uint32),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    max_force: wp.array(dtype=float),
    gear: wp.array(dtype=float),
    constant_force: wp.array(dtype=float),
    output: wp.array(dtype=float),
):
    """PD control: f = clamp(G*(constant + act + kp*(target_pos - G*q) + kd*(target_vel - G*v)), ±max_force). Adds to output."""
    i = wp.tid()
    state_idx = state_indices[i]
    target_idx = target_indices[i]
    out_idx = output_indices[i]

    g = gear[i]
    position_error = target_pos[target_idx] - g * current_pos[state_idx]
    velocity_error = target_vel[target_idx] - g * current_vel[state_idx]

    const_f = float(0.0)
    if constant_force:
        const_f = constant_force[i]

    act = float(0.0)
    if control_input:
        act = control_input[target_idx]

    force = g * (const_f + act + kp[i] * position_error + kd[i] * velocity_error)
    force = wp.clamp(force, -max_force[i], max_force[i])

    output[out_idx] = output[out_idx] + force


@wp.kernel
def pid_controller_kernel(
    current_pos: wp.array(dtype=float),
    current_vel: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    target_vel: wp.array(dtype=float),
    control_input: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    target_indices: wp.array(dtype=wp.uint32),
    output_indices: wp.array(dtype=wp.uint32),
    kp: wp.array(dtype=float),
    ki: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    max_force: wp.array(dtype=float),
    integral_max: wp.array(dtype=float),
    gear: wp.array(dtype=float),
    constant_force: wp.array(dtype=float),
    dt: float,
    current_integral: wp.array(dtype=float),
    output: wp.array(dtype=float),
):
    """PID control with anti-windup: f = clamp(G*(constant + act + kp*(target_pos - G*q) + ki*integral + kd*(target_vel - G*v)), ±max_force). Adds to output."""
    i = wp.tid()
    state_idx = state_indices[i]
    target_idx = target_indices[i]
    out_idx = output_indices[i]

    g = gear[i]
    position_error = target_pos[target_idx] - g * current_pos[state_idx]
    velocity_error = target_vel[target_idx] - g * current_vel[state_idx]

    integral = current_integral[i] + position_error * dt
    integral = wp.clamp(integral, -integral_max[i], integral_max[i])

    const_f = float(0.0)
    if constant_force:
        const_f = constant_force[i]

    act = float(0.0)
    if control_input:
        act = control_input[target_idx]

    force = g * (const_f + act + kp[i] * position_error + ki[i] * integral + kd[i] * velocity_error)
    force = wp.clamp(force, -max_force[i], max_force[i])

    output[out_idx] = output[out_idx] + force


@wp.kernel
def pid_integral_state_kernel(
    current_pos: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    target_indices: wp.array(dtype=wp.uint32),
    integral_max: wp.array(dtype=float),
    gear: wp.array(dtype=float),
    dt: float,
    current_integral: wp.array(dtype=float),
    next_integral: wp.array(dtype=float),
):
    """Update PID integral state with anti-windup."""
    i = wp.tid()
    state_idx = state_indices[i]
    target_idx = target_indices[i]

    g = gear[i]
    position_error = target_pos[target_idx] - g * current_pos[state_idx]

    integral = current_integral[i] + position_error * dt
    integral = wp.clamp(integral, -integral_max[i], integral_max[i])

    next_integral[i] = integral


@wp.kernel
def delay_buffer_state_kernel(
    target_pos_global: wp.array(dtype=float),
    target_vel_global: wp.array(dtype=float),
    control_input_global: wp.array(dtype=float),
    indices: wp.array(dtype=wp.uint32),
    copy_idx: int,
    write_idx: int,
    current_buffer_pos: wp.array2d(dtype=float),
    current_buffer_vel: wp.array2d(dtype=float),
    current_buffer_act: wp.array2d(dtype=float),
    next_buffer_pos: wp.array2d(dtype=float),
    next_buffer_vel: wp.array2d(dtype=float),
    next_buffer_act: wp.array2d(dtype=float),
):
    """Update delay circular buffer: copy missing entry, write new entry."""
    i = wp.tid()
    global_idx = indices[i]

    next_buffer_pos[copy_idx, i] = current_buffer_pos[copy_idx, i]
    next_buffer_vel[copy_idx, i] = current_buffer_vel[copy_idx, i]
    next_buffer_act[copy_idx, i] = current_buffer_act[copy_idx, i]

    next_buffer_pos[write_idx, i] = target_pos_global[global_idx]
    next_buffer_vel[write_idx, i] = target_vel_global[global_idx]
    next_buffer_act[write_idx, i] = control_input_global[global_idx]
