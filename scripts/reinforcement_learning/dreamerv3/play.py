# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



"""Play an algorithm (supports both coordination + adversarial HARL runners)."""

import argparse
import os
import pprint
import sys
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
import ruamel.yaml as yaml
import portal
from functools import partial as bind

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play an RL agent with HARL.", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    "--algorithm",
    type=str,
    default="happo",
    choices=[
        "happo",
        "hatrpo",
        "haa2c",
        "haddpg",
        "hatd3",
        "hasac",
        "had3qn",
        "maddpg",
        "matd3",
        "mappo",
        "happo_adv",
    ],
    help="Algorithm name.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_env_steps", type=int, default=None, help="Total environment steps to play.")
parser.add_argument("--dir", type=str, default=None, help="Folder with trained models (local path).")
parser.add_argument("--debug", action="store_true", help="Run in debug mode for visualization.")
parser.add_argument(
    "--load_starting_policy",
    action="store_true",
    help="If set, load the starting policy for this env from HuggingFace (if one exists).",
)
parser.add_argument(
    "--load_trained_policy",
    action="store_true",
    help="If set, load the trained policy for this env from HuggingFace (if one exists).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# --------------------------------------------------------------------------------------
# Launch Omniverse
# --------------------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------------------
# Imports that require the app
# --------------------------------------------------------------------------------------

import sys
from pathlib import Path

import dreamerv3
from dreamerv3.main import make_agent, make_logger, make_replay, make_env, wrap_env, make_stream
folder = Path(dreamerv3.__file__).parent
sys.path.insert(0, str(folder.parent))

import embodied
import elements

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"dreamer_cfg_entry_point"

def _max_action_dim(action_space) -> int:
    """Recursively find the maximum action dimension across nested dict action spaces."""
    if isinstance(action_space, dict):
        if len(action_space) == 0:
            return 0
        return max(_max_action_dim(v) for v in action_space.values())
    # assume a gymnasium space-like object with .shape
    shape = getattr(action_space, "shape", None)
    if not shape:
        return 0
    return int(shape[0])

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, configs: dict):
    args = args_cli.__dict__
    env_args: dict = {}
    if args.get("num_envs") is not None:
        env_cfg.scene.num_envs = args["num_envs"]
    else:
        # ensure downstream uses an int
        args["num_envs"] = int(env_cfg.scene.num_envs)

    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    env_args["video_settings"] = {"video": False}
    env_args["headless"] = args["headless"]
    env_args["debug"] = args["debug"]

    configs["defaults"]["task"] = f"isaaclab_{configs['defaults']['task']}"
    # HARL runner args
    args["env"] = "isaaclab"
    args["algo"] = args["algorithm"]
    args["exp_name"] = "play"


    parsed, other = elements.Flags(configs=['defaults']).parse_known()
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)
    config = config.update(logdir=(
        config.logdir.format(timestamp=elements.timestamp())))

    if 'JOB_COMPLETION_INDEX' in os.environ:
        config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
    print('Replica:', config.replica, '/', config.replicas)

    logdir = elements.Path(config.logdir)
    print('Logdir:', logdir)
    print('Run script:', config.script)
    if not config.script.endswith(('_env', '_replay')):
        logdir.mkdir()
        config.save(logdir / 'config.yaml')

    def init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        errfile=config.errfile and logdir / 'error',
        clientkw=dict(logging_color='cyan'),
        serverkw=dict(logging_color='cyan'),
        initfns=[init],
        ipv6=config.ipv6,
    )

    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    if config.script == 'train':
        embodied.run.train(
            bind(make_agent, config, env_args=env_args),
            bind(make_replay, config, 'replay'),
            bind(make_env, config, env_args=env_args),
            bind(make_stream, config),
            bind(make_logger, config),
            args)

    elif config.script == 'train_eval':
        embodied.run.train_eval(
            bind(make_agent, config),
            bind(make_replay, config, 'replay'),
            bind(make_replay, config, 'eval_replay', 'eval'),
            bind(make_env, config),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args)

    elif config.script == 'eval_only':
        embodied.run.eval_only(
            bind(make_agent, config),
            bind(make_env, config),
            bind(make_logger, config),
            args)

    elif config.script == 'parallel':
        embodied.run.parallel.combined(
            bind(make_agent, config),
            bind(make_replay, config, 'replay'),
            bind(make_replay, config, 'replay_eval', 'eval'),
            bind(make_env, config),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args)

    elif config.script == 'parallel_env':
        is_eval = config.replica >= args.envs
        embodied.run.parallel.parallel_env(
            bind(make_env, config), config.replica, args, is_eval)

    elif config.script == 'parallel_envs':
        is_eval = config.replica >= args.envs
        embodied.run.parallel.parallel_envs(
            bind(make_env, config), bind(make_env, config), args)

    elif config.script == 'parallel_replay':
        embodied.run.parallel.parallel_replay(
            bind(make_replay, config, 'replay'),
            bind(make_replay, config, 'replay_eval', 'eval'),
            bind(make_stream, config),
            args)

    else:
        raise NotImplementedError(config.script)



    

    obs, _, _ = runner.env.reset()

    # determine max action dim (supports nested dict action spaces for adv)
    max_action_space = 0
    for _, sp in runner.env.action_space.items():
        max_action_space = max(max_action_space, _max_action_dim(sp))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    actions = torch.zeros(
        (args["num_envs"], runner.num_agents, max_action_space),
        dtype=torch.float32,
        device=device,
    )
    rnn_states = torch.zeros(
        (args["num_envs"], runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
        dtype=torch.float32,
        device=device,
    )
    masks = torch.ones((args["num_envs"], runner.num_agents, 1), dtype=torch.float32, device=device)

    # logging + progress bar (from adv script)
    log_infos: dict[str, float] = {}
    envs_completed = 0
    pbar = tqdm(total=args["num_env_steps"], desc="Playing")

    while simulation_app.is_running():
        with torch.inference_mode():
            if is_adv:
                # obs: {team: {agent_id: obs_tensor}}
                for team, agents_obs in obs.items():
                    for agent_id, agent_obs in agents_obs.items():
                        agent_num = runner.env.env._agent_map[agent_id]
                        action, _, rnn_state = runner.actors[team][agent_id].get_actions(
                            agent_obs,
                            rnn_states[:, agent_num, :],
                            masks[:, agent_num, :],
                            None,
                            None,
                        )
                        a_dim = action.shape[1]
                        actions[:, agent_num, :a_dim] = action
                        rnn_states[:, agent_num, :] = rnn_state
            else:
                # obs: {agent_name: obs_tensor}  (or similar mapping)
                for agent_num, agent_obs in enumerate(obs.values()):
                    action, _, rnn_state = runner.actor[agent_num].get_actions(
                        agent_obs,
                        rnn_states[:, agent_num, :],
                        masks[:, agent_num, :],
                        None,
                        None,
                    )
                    a_dim = action.shape[1]
                    actions[:, agent_num, :a_dim] = action
                    rnn_states[:, agent_num, :] = rnn_state

        # step env
        obs, _, _, dones, _, _ = runner.env.step(actions)

        # episode completion aggregation
        dones_env = torch.all(dones, dim=1)
        curr_envs_completed = int(dones_env.sum().item())
        envs_completed += curr_envs_completed

        if curr_envs_completed > 0 and hasattr(runner.env, "log_info"):
            for k, v in runner.env.log_info.items():
                if k not in log_infos:
                    log_infos[k] = 0.0
                log_infos[k] += float(v) * curr_envs_completed

        # reset masks/rnn where done
        masks = torch.ones((args["num_envs"], runner.num_agents, 1), dtype=torch.float32, device=device)
        masks[dones_env] = 0.0

        if curr_envs_completed > 0:
            rnn_states[dones_env] = torch.zeros(
                (curr_envs_completed, runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
                dtype=torch.float32,
                device=device,
            )

        # update pbar using total env-steps (matches adv script semantics)
        sim_steps = int(runner.env.unwrapped.sim._number_of_steps)
        num_steps = int(args["num_envs"]) * sim_steps
        pbar.update(max(0, num_steps - pbar.n))

        if num_steps >= int(args["num_env_steps"]):
            break

    runner.env.close()
    pbar.close()

    # print averaged log infos (from adv script)
    if envs_completed <= 0:
        envs_completed = 1
    for k in list(log_infos.keys()):
        log_infos[k] /= envs_completed
    if log_infos:
        pprint.pprint(log_infos)


if __name__ == "__main__":
    main()
    simulation_app.close()
