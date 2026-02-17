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

parser = argparse.ArgumentParser(description="Train an RL agent with DreamerV3.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_env_steps", type=int, default=None, help="Total environment steps to play.")
parser.add_argument("--dir", type=str, default=None, help="Folder with trained models (local path).")
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

    configs["defaults"]["task"] = f"isaaclab_{args['task']}"
    configs["defaults"]["run"]["envs"] = args["num_envs"]
    # HARL runner args
    args["env"] = "isaaclab"
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

    embodied.run.train(
        bind(make_agent, config, env_args=env_args),
        bind(make_replay, config, 'replay'),
        bind(make_env, config, env_args=env_args),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

if __name__ == "__main__":
    main()
    simulation_app.close()
