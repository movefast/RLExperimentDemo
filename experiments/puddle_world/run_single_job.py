import os
import pathlib
from functools import partial

import fire
import yaml
from common.train_helpers import run
from environments.puddle_world import PuddleWorld

from . import AGENT_DEFAULT_PARAMS, AGENT_REG

cur_dir = pathlib.Path(os.path.split(os.path.realpath(__file__))[0])
with open(cur_dir/os.environ["CONFIG_FNAME"]) as f:
    CONFIG = yaml.safe_load(f)

CONFIG['agent_register'] = AGENT_REG
CONFIG['agent_default_params'] = AGENT_DEFAULT_PARAMS
CONFIG['experiment']['dest_dir'] = cur_dir

run_single_job = partial(run, CONFIG, PuddleWorld())

if __name__ == '__main__':
    fire.Fire(run_single_job)
