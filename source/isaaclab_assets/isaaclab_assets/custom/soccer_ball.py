# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a Minitank robot with an arm joint."""

from pathlib import Path

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
import isaaclab_assets

asset_path = Path(isaaclab_assets.__file__).parent / "custom"

SOCCERBALL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(asset_path, "soccer_ball.usda")),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={},
)
