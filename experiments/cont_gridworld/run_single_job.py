import os
import pathlib
import random
import time
from dataclasses import asdict
from functools import partial

import config
import fire
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import yaml
from common.train_utils import run
from environments.cont_gridworld import ContinuousGridWorld
from fastprogress.fastprogress import master_bar, progress_bar

from . import AGENT_DEFAULT_PARAMS, AGENT_REG

cur_dir = pathlib.Path(os.path.split(os.path.realpath(__file__))[0])
with open(cur_dir/os.environ["CONFIG_FNAME"]) as f:
    CONFIG = yaml.safe_load(f)

CONFIG['agent_register'] = AGENT_REG
CONFIG['agent_default_params'] = AGENT_DEFAULT_PARAMS
CONFIG['experiment']['dest_dir'] = cur_dir
CONFIG['experiment']['discounted_return'] = True

env_setting = {
    "s_noise": CONFIG['env']['s_noise'],
    'edge_scale': CONFIG['env']['edge_scale'],
    'step_len': CONFIG['env']['step_len'],
}

run_single_job = partial(run, CONFIG, ContinuousGridWorld(env_setting))

if __name__ == '__main__':
    fire.Fire(run_single_job)
