import os
import pathlib
from functools import partial

import fire
import gym
import yaml
from common.train_helpers import run

from . import AGENT_DEFAULT_PARAMS, AGENT_REG

cur_dir = pathlib.Path(os.path.split(os.path.realpath(__file__))[0])
with open(cur_dir/os.environ["CONFIG_FNAME"]) as f:
    CONFIG = yaml.safe_load(f)

CONFIG['agent_register'] = AGENT_REG
CONFIG['agent_default_params'] = AGENT_DEFAULT_PARAMS
CONFIG['experiment']['dest_dir'] = cur_dir
CONFIG['experiment']['discounted_return'] = True

env = gym.make('LunarLander-v2')
CONFIG['env']['num_states'] = env.observation_space.shape[0]
CONFIG['env']['num_actions'] = env.action_space.n

run_single_job = partial(run, CONFIG, env)

if __name__ == '__main__':
    fire.Fire(run_single_job)
