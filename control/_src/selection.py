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

"""Helper functions and kernels for actuator-view index mapping.

These utilities enable mapping between actuator indices and view layouts,
useful for getting/setting actuator attributes from a view's perspective.
"""

import warp as wp


@wp.kernel
def scatter_actuator_to_view_kernel(
    actuator_values: wp.array(dtype=float),
    index_map: wp.array(dtype=wp.int32),
    output: wp.array(dtype=float),
):
    """Scatter actuator values to view array at mapped indices.

    Copies values from actuator array to view array based on index mapping.
    Thread i copies actuator_values[i] to output[index_map[i]] if index_map[i] >= 0.

    Args:
        actuator_values: Source values from actuator, shape (N,).
        index_map: Mapping from actuator indices to view indices, shape (N,).
            -1 indicates no mapping.
        output: Destination view array to write to.
    """
    tid = wp.tid()
    view_idx = index_map[tid]
    if view_idx >= 0:
        output[view_idx] = actuator_values[tid]


@wp.kernel
def gather_view_to_actuator_kernel(
    view_values: wp.array(dtype=float),
    index_map: wp.array(dtype=wp.int32),
    output: wp.array(dtype=float),
):
    """Gather from view array to actuator array at mapped indices.

    Copies values from view array to actuator array based on index mapping.
    Thread i copies view_values[index_map[i]] to output[i] if index_map[i] >= 0.

    Args:
        view_values: Source values from view, shape (M,).
        index_map: Mapping from actuator indices to view indices, shape (N,).
            -1 indicates no mapping.
        output: Destination actuator array to write to, shape (N,).
    """
    tid = wp.tid()
    view_idx = index_map[tid]
    if view_idx >= 0:
        output[tid] = view_values[view_idx]

