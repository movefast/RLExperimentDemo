import glob
import inspect
import os
import pathlib
import random
from itertools import product

import fire
import numpy as np
import yaml
from fastprogress.fastprogress import master_bar, progress_bar

from . import AGENT_REG, params

MAX_EVALS = 200
count = 0

cur_dir = pathlib.Path(os.path.split(os.path.realpath(__file__))[0])
EXP_DIR = str(cur_dir).split('/')[-1]
JOBS_DIR = "jobs"


def create_job(agent_type, run_num, hyper_params):
    global count
    cmd = f"CONFIG_FNAME=\"{CONFIG_FNAME}\" python -m experiments.{EXP_DIR}.run_single_job --agent_type=\"{agent_type}\" --run_num={run_num} --hyper_params=\"{hyper_params}\""
    with open(cur_dir/f"jobs/tasks_{count}.sh", 'w') as f:
        f.write(cmd)
    print(count, cmd)
    count += 1


def random_search(agent_type, param_grid, max_evals=MAX_EVALS):
    """Random search for hyperparameter optimization"""
    for i in progress_bar(range(max_evals)):
        param_keys, values = zip(*param_grid.items())
        param_combos = [dict(zip(flatten(param_keys), flatten(combo))) for combo in product(*values)]
        hyper_params = random.sample(param_combos,1)[0]
        print(hyper_params)
        # Evaluate randomly selected hyperparameters
        for run_num in range(START_RUN, num_runs):
            create_job(agent_type, run_num, hyper_params)


def grid_search(agent_type, param_grid):
    """grid search for hyperparameter optimization"""
    param_keys, values = zip(*param_grid.items())

    param_combos = [dict(zip(flatten(param_keys), flatten(combo))) for combo in product(*values)]

    mb = master_bar(np.random.permutation(param_combos))
    files = set(glob.glob(str(cur_dir/'metrics/*')))
    for i, hyper_params in enumerate(mb):
        print(hyper_params)
        for run_num in range(START_RUN, num_runs):
            algorithm = agent_type + '_' + '_'.join([f'{k}_{v}' for k, v in hyper_params.items()])
            if str(cur_dir/f'metrics/{algorithm}_{run_num}.torch') not in files:
                create_job(agent_type, run_num, hyper_params)

def flatten(t):
    result = []
    for ele in t:
        if type(ele) is list or type(ele) is tuple:
            result.extend(ele)
        else:
            result.append(ele)
    return result

def get_lr(b=1e-2, a=2, n=5):
    return list(b/a**np.array(list(range(0, n))))


def write_jobs(append=True, agents=None, config_fname="base_config.yaml", param_dname="params_to_search", start_run=0, run_hr=3):
    global CONFIG_FNAME, num_runs, START_RUN
    CONFIG_FNAME = config_fname
    START_RUN = start_run
    with open(cur_dir/config_fname) as f:
        CONFIG = yaml.safe_load(f)
    num_runs = CONFIG['experiment']['num_runs']

    params_to_search = getattr(params, param_dname)

    (cur_dir/f"{JOBS_DIR}").mkdir(parents=True, exist_ok=True)
    cur_tsk_fs = [f for f in os.listdir(f"experiments/{EXP_DIR}/{JOBS_DIR}") if f.startswith("tasks")]

    if append:
        global count
        count = len(cur_tsk_fs)
    else:
        for fname in cur_tsk_fs:
            os.remove(cur_dir/f"{JOBS_DIR}"/fname)

    bgn_count = count

    if agents is None:
        agents = list(params_to_search.keys())
    elif not isinstance(agents, list):
        agents = [agents]

    for agent_type in master_bar(agents):
        if agent_type not in list(AGENT_REG.keys()):
            print(f"{agent_type} is not found in experiments/__init__.py; skipping")
            continue
        print(agent_type)
        grid_search(agent_type, params_to_search[agent_type])


if __name__ == "__main__":
    fire.Fire(write_jobs)
