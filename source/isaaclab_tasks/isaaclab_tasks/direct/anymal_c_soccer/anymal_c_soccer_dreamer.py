# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, subtract_frame_transforms

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_assets.custom.soccer_ball import SOCCERBALL_CFG  # isort: skip


def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())


# @configclass
# class EventCfg:
#     """Configuration for randomization."""

#     physics_material_0 = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot_0", body_names=".*"),
#             "static_friction_range": (0.8, 0.8),
#             "dynamic_friction_range": (0.6, 0.6),
#             "restitution_range": (0.0, 0.2),
#             "num_buckets": 64,
#         },
#     )

#     add_base_mass_0 = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot_0", body_names="base"),
#             "mass_distribution_params": (-5.0, 5.0),
#             "operation": "add",
#         },
#     )


@configclass
class AnymalDreamerSoccerEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 100.0
    action_scale = 0.5
    action_space = 12
    # observation space: robot_state (45) + camera RGB (128*128*3)
    observation_space = 45 + 128*128*3  # robot state + camera RGB
    state_space = 0
    possible_agents = ["robot_0"]

    # events: EventCfg = EventCfg()
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    # robot_0.init_state.rot = get_quaternion_tuple_from_xyz(0,torch.pi,0)
    robot_0.init_state.pos = (0.0, 0.0, 0.3)

    wall_0 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object0",
        spawn=sim_utils.CuboidCfg(
            size=(20, 0.5, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 5.0, 1), rot=(1.0, 0.0, 0.0, 0.0)  # Position originally was (0.0, 0, 0.61)
        ),
    )

    wall_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object1",
        spawn=sim_utils.CuboidCfg(
            size=(20, 0.5, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -5.0, 1), rot=(1.0, 0.0, 0.0, 0.0)  # Position originally was (0.0, 0, 0.61)
        ),
    )

    wall_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0)  # Position originally was (0.0, 0, 0.61)
        ),
    )

    wall_3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object3",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0)  # Position originally was (0.0, 0, 0.61)
        ),
    )

    ball = SOCCERBALL_CFG.replace(prim_path="/World/envs/env_.*/Object4")
    ball.init_state.pos = (0.0, 0.0, 0.1)

    env_spacing = 20.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=env_spacing, replicate_physics=True)

    # Contact sensor for go_to_point rewards
    contact_sensor_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # Camera for visual observations
    camera_0 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot_0/base/front_cam",
        update_period=0.1,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.5, 0.0, 0.0), rot=get_quaternion_tuple_from_xyz(0, 0, 0), convention="world"),
    )

    # Reward scales from stage_1, stage_2, and go_to_point_soccer
    # Stage 1 rewards
    dist_to_ball_reward_scale = 1.0

    # Stage 2 rewards
    goal_reward_scale = 20
    ball_to_goal_reward_scale = 1.0

    # Go to point soccer rewards
    reached_goal_reward = 10.0
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0


class AnymalDreamerSoccerEnv(DirectRLEnv):
    cfg: AnymalDreamerSoccerEnvCfg

    def __init__(
        self, cfg: AnymalDreamerSoccerEnvCfg, render_mode: str | None = None, headless: bool | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.headless = headless

        self.env_spacing = self.cfg.env_spacing

        # Joint position command
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # Stage 1 rewards
                "dist_to_ball_reward",
                # Stage 2 rewards
                "ball_to_goal_reward",
                "goal_reward",
                # Go to point soccer rewards
                "goal_reached",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "goal1": sim_utils.CuboidCfg(
                    size=(1, 3, 0.1),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
                "goal2": sim_utils.CuboidCfg(
                    size=(1, 3, 0.1),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "goal_to_score": sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
        self.goal_area = VisualizationMarkers(marker_cfg)
        self.goal1_pos, self.goal2_pos, self.goal1_area, self.goal2_area = self._get_goal_areas()
        self.target_goal = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_0)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor_0)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._camera = TiledCamera(self.cfg.camera_0)
        self.scene.sensors["camera"] = self._camera
        
        self.wall_0 = RigidObject(self.cfg.wall_0)
        self.wall_1 = RigidObject(self.cfg.wall_1)
        self.wall_2 = RigidObject(self.cfg.wall_2)
        self.wall_3 = RigidObject(self.cfg.wall_3)
        self.ball = RigidObject(self.cfg.ball)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _draw_goal_areas(self):
        marker_ids0 = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        marker_ids1 = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        marker_ids2 = 2 * torch.ones(self.num_envs, dtype=torch.int32, device=self.device)

        marker_ids = torch.concat([marker_ids0, marker_ids1, marker_ids2], dim=0)

        goal_to_score_pos = torch.where(self.target_goal.unsqueeze(1) == 0, self.goal1_pos, self.goal2_pos)

        marker_locations = torch.concat([self.goal1_pos, self.goal2_pos, goal_to_score_pos], dim=0)

        self.goal_area.visualize(marker_locations, marker_indices=marker_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        self.ball.update(self.step_dt)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # Robot state observation (45 dims)
        robot_state = torch.cat(
            [
                self._robot.data.root_lin_vel_b,  # 3
                self._robot.data.root_ang_vel_b,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12
                self._robot.data.joint_vel,  # 12
                self._actions,  # 12
            ],
            dim=-1,
        )  # Total: 45 dimensions

        # Camera observation - get RGB image and flatten
        camera_data = self._camera.data.output["rgb"].to(self.device)
        # Normalize RGB image (flatten and normalize)
        camera_data = camera_data.reshape(self.num_envs, -1)  # [num_envs, 128*128*3]
        camera_data = torch.nan_to_num(camera_data, nan=0.0, posinf=255.0, neginf=0.0)
        # Normalize to [-1, 1]
        camera_data = camera_data / 255.0 * 2.0 - 1.0
        camera_data = torch.clamp(camera_data, -1.0, 1.0)

        # Concatenate robot state and camera observation
        obs = torch.cat([robot_state, camera_data], dim=-1)

        self._previous_actions = self._actions.clone()

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        ball_in_goal1, ball_in_goal2 = self._ball_in_goal_area()

        goal_pos = torch.zeros_like(self.goal1_pos)
        goal_pos[self.target_goal == 0] = self.goal1_pos[self.target_goal == 0]
        goal_pos[self.target_goal == 1] = self.goal2_pos[self.target_goal == 1]

        rewards = {}

        # ===== Stage 1 Rewards: Distance to ball =====
        robot_distance_to_ball = torch.linalg.norm(
            self._robot.data.root_pos_w[:, :3] - self.ball.data.root_pos_w, dim=1
        )
        robot_distance_to_ball_mapped = 1 - torch.tanh(robot_distance_to_ball / 0.8)
        rewards["dist_to_ball_reward"] = (
            robot_distance_to_ball_mapped * self.cfg.dist_to_ball_reward_scale * self.step_dt
        )

        # ===== Stage 2 Rewards: Ball to goal and goal scoring =====
        ball_distance_to_goal = torch.linalg.norm(self.ball.data.root_pos_w - goal_pos, dim=1)
        ball_distance_to_goal_mapped = 1 - torch.tanh(ball_distance_to_goal / 0.8)
        rewards["ball_to_goal_reward"] = (
            ball_distance_to_goal_mapped * self.cfg.ball_to_goal_reward_scale * self.step_dt
        )

        goal_reward = torch.zeros(self.num_envs, device=self.device)
        goal_reward[ball_in_goal1 & (self.target_goal == 0)] = 1.0
        goal_reward[ball_in_goal2 & (self.target_goal == 1)] = 1.0
        goal_reward[ball_in_goal1 & (self.target_goal == 1)] = -1.0
        goal_reward[ball_in_goal2 & (self.target_goal == 0)] = -1.0
        rewards["goal_reward"] = goal_reward * self.cfg.goal_reward_scale

        # ===== Go to Point Soccer Rewards: Robot state quality and goal reaching =====
        # Goal reached reward (based on distance to goal)
        # NOTE: Using ball distance as proxy for "goal" in soccer context
        dists_to_ball = torch.norm(
            self._robot.data.root_pos_w[:, :2] - self.ball.data.root_pos_w[:, :2], dim=-1
        )
        goal_reached = self._ball_in_goal_area()[0]  # Check if ball is in the correct goal area
        rewards["goal_reached"] = goal_reached.float() * self.cfg.reached_goal_reward

        # Z velocity penalty (keep flat locomotion)
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        rewards["lin_vel_z_l2"] = z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt

        # Angular velocity x/y penalty (keep stable)
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        rewards["ang_vel_xy_l2"] = ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt

        # Joint torque penalty
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        rewards["dof_torques_l2"] = joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt

        # Joint acceleration penalty
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        rewards["dof_acc_l2"] = joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt

        # Action rate penalty
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        rewards["action_rate_l2"] = action_rate * self.cfg.action_rate_reward_scale * self.step_dt

        # Feet air time reward (encourage dynamic gait)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)
        rewards["feet_air_time"] = air_time * self.cfg.feet_air_time_reward_scale * self.step_dt

        # Undesired contact penalty (avoid body collisions)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(
                torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1),
                dim=1,
            )[0]
            > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        rewards["undesired_contacts"] = contacts * self.cfg.undesired_contact_reward_scale * self.step_dt

        # Flat orientation reward (keep robot upright)
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )
        rewards["flat_orientation_l2"] = flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt

        # Compute total reward
        rewards = {k: torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6) for k, v in rewards.items()}
        reward = torch.sum(torch.stack([rewards[key] for key in rewards.keys()]), dim=0)

        # Log rewards
        for key in rewards.keys():
            self._episode_sums[key] += rewards[key]

        return reward

    def _get_goal_areas(self):
        goal1_size = self.goal_area.cfg.markers["goal1"].size
        goal1_pos = self.scene.env_origins.clone() + torch.tensor([-9, 0.0, 0.05], device=self.device)

        goal2_size = self.goal_area.cfg.markers["goal2"].size
        goal2_pos = self.scene.env_origins.clone() + torch.tensor([9, 0.0, 0.05], device=self.device)

        # Extract goal area from goal post positions
        goal1_min = goal1_pos + torch.tensor([-goal1_size[0] / 2, -goal1_size[1] / 2, 0], device=self.device)
        goal1_max = goal1_pos + torch.tensor([goal1_size[0] / 2, goal1_size[1] / 2, 0], device=self.device)
        goal2_min = goal2_pos + torch.tensor([-goal2_size[0] / 2, -goal2_size[1] / 2, 0], device=self.device)
        goal2_max = goal2_pos + torch.tensor([goal2_size[0] / 2, goal2_size[1] / 2, 0], device=self.device)

        return goal1_pos, goal2_pos, (goal1_min, goal1_max), (goal2_min, goal2_max)

    def _ball_in_goal_area(self):
        ball_pos = self.ball.data.root_pos_w[:, :2]
        in_goal1 = torch.all((ball_pos >= self.goal1_area[0][:, :2]) & (ball_pos <= self.goal1_area[1][:, :2]), dim=1)
        in_goal2 = torch.all((ball_pos >= self.goal2_area[0][:, :2]) & (ball_pos <= self.goal2_area[1][:, :2]), dim=1)
        return in_goal1, in_goal2

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ball_in_goal1, ball_in_goal2 = self._ball_in_goal_area()
        ball_in_any_goal = ball_in_goal1 | ball_in_goal2

        dones = ball_in_any_goal

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self._draw_goal_areas()

        sampled_grid_pos = self._sample_positions_grid(env_ids, 2, 1, 1)

        # Cache for convenience
        origins = self.scene.env_origins[env_ids]  # (N, 3)

        ball_default_state = self.ball.data.default_root_state.clone()[env_ids]
        ball_default_state[:, :2] = (
            ball_default_state[:, :2] + self.scene.env_origins[env_ids][:, :2] + sampled_grid_pos[:, 1]
        )
        self.ball.write_root_state_to_sim(ball_default_state, env_ids)
        self.ball.reset(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # Place robot
        default_root_state[:, :2] = origins[:, :2]
        default_root_state[:, 2] += self._robot.data.default_root_state[env_ids][:, 2]
        default_root_state[:, :2] += sampled_grid_pos[:, 0]

        # Write to sim
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

    def _sample_positions_grid(self, env_ids, num_samples, min_dist=1.0, grid_spacing=1.0):
        device = self.scene.env_origins.device
        N = len(env_ids)

        offsets = torch.zeros((N, num_samples, 2), device=device)
        env_origins = self.scene.env_origins[env_ids][:, :2].clone()

        _, _, goal1_area, goal2_area = self._get_goal_areas()
        goal1_min, goal1_max = goal1_area
        goal2_min, goal2_max = goal2_area

        all_valid_points = []  # collect per-env lists

        for i in range(N):
            xs = torch.arange(env_origins[i, 0] - 9, env_origins[i, 0] + 10, grid_spacing, device=device)
            ys = torch.arange(env_origins[i, 1] - 4, env_origins[i, 1] + 5, grid_spacing, device=device)
            xv, yv = torch.meshgrid(xs, ys, indexing="ij")
            grid_points = torch.stack([xv.flatten(), yv.flatten()], dim=-1)

            # mask out goal areas
            in_goal1 = (
                (grid_points[:, 0] >= goal1_min[i, 0])
                & (grid_points[:, 0] <= goal1_max[i, 0])
                & (grid_points[:, 1] >= goal1_min[i, 1])
                & (grid_points[:, 1] <= goal1_max[i, 1])
            )
            in_goal2 = (
                (grid_points[:, 0] >= goal2_min[i, 0])
                & (grid_points[:, 0] <= goal2_max[i, 0])
                & (grid_points[:, 1] >= goal2_min[i, 1])
                & (grid_points[:, 1] <= goal2_max[i, 1])
            )
            mask = ~(in_goal1 | in_goal2)

            valid_points = grid_points[mask]
            all_valid_points.append(valid_points)

            if valid_points.shape[0] < num_samples:
                raise ValueError(f"Not enough valid grid points outside goals for env {i}")

            idx = torch.randperm(valid_points.shape[0], device=device)[:num_samples]
            offsets[i] = valid_points[idx] - env_origins[i]

        # save for visualization
        self.valid_points = all_valid_points
        # self._draw_grid_markers()

        return offsets

    @torch.no_grad()
    def _draw_grid_markers(self):
        """
        Draws green dots at every valid grid point for every environment
        stored in self.valid_points (populated by _sample_positions_grid).
        """
        device = self.device
        if not hasattr(self, "valid_points"):
            raise RuntimeError("Run _sample_positions_grid first to populate valid_points")

        pos_chunks = []
        idx_chunks = []
        marker_counter = 0

        for i, pts in enumerate(self.valid_points):
            z_col = torch.full((pts.shape[0], 1), 0.05, device=device)
            pos_i = torch.cat([pts, z_col], dim=1)

            pos_chunks.append(pos_i)
            idx_chunks.append(marker_counter + torch.arange(pts.shape[0], device=device, dtype=torch.long))
            marker_counter += pts.shape[0]

        marker_positions = torch.cat(pos_chunks, dim=0)  # (M, 3)
        marker_indices = torch.cat(idx_chunks, dim=0)  # (M,)
        marker_scales = 10 * torch.ones((marker_positions.shape[0], 3), device=device)

        marker_orientations = torch.zeros((marker_positions.shape[0], 4), device=device)
        marker_orientations[:, 0] = 1.0  # identity quaternion

        if not hasattr(self, "grid_markers"):
            markers = {
                f"grid_{i}": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
                for i in range(marker_counter)
            }

            grid_marker_cfg = VisualizationMarkersCfg(prim_path="/World/GridMarkers", markers=markers)
            self.grid_markers = VisualizationMarkers(grid_marker_cfg)

        self.grid_markers.visualize(
            marker_positions,
            marker_orientations,
            scales=marker_scales,
            marker_indices=marker_indices,
        )
