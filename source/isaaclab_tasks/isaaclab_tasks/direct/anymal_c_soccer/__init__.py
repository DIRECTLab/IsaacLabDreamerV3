# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .anymal_c_soccer_dreamer import AnymalDreamerSoccerEnv, AnymalDreamerSoccerEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Anymal-C-Dreamer-Soccer-v0",
    entry_point=AnymalDreamerSoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalDreamerSoccerEnvCfg,
        "dreamer_cfg_entry_point": f"{agents.__name__}:dreamer_cfg.yaml",
    },
)

