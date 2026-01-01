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

"""State types for stateful actuators."""

from dataclasses import dataclass

import warp as wp


@dataclass
class DelayedActuatorState:
    """Circular buffer state for delayed actuators."""

    buffer_pos: wp.array = None  # Shape (delay, N)
    buffer_vel: wp.array = None  # Shape (delay, N)
    buffer_act: wp.array = None  # Shape (delay, N)
    write_idx: int = 0           # Last write position
    is_filled: bool = False      # Buffer filled at least once


@dataclass
class PIDActuatorState:
    """Integral state for PID actuators."""

    integral: wp.array = None  # Shape (N,)
