import os
import sys
import time
from concurrent.futures import (FIRST_EXCEPTION, ProcessPoolExecutor,
                                ThreadPoolExecutor, as_completed)
from itertools import product

import fire
import numpy as np

from .params import *
from .run_single_job import run_single_job as run


def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def test_process():
    print("Executing our Task on Process {}".format(os.getpid()))


def simple_param_search(hp_dict_name="params_to_search", num_runs=1):
    start = time.time()
    futures = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        for agent_type, param_grid in getattr(sys.modules[__name__], hp_dict_name).items():
            param_keys, values = zip(*param_grid.items())
            param_combos = [dict(zip(param_keys, combo)) for combo in product(*values)]
            for param_combo in param_combos:
                for run_num in range(num_runs):
                    futures.append(executor.submit(run, agent_type, param_combo, run_num=run_num))
    results = [f.result() for f in futures]
    print('Total time taken: {}'.format(time.time() - start))
    print(results)

def run_mult_jobs(agent_type, hyper_params, num_runs):
    start = time.time()
    futures = []
    with ProcessPoolExecutor(max_workers=num_cpus()//2) as executor:
        for run_num in range(num_runs):
            futures.append(executor.submit(run, agent_type, hyper_params, run_num=run_num))
    results = [f.result() for f in futures]
    print('Total time taken: {}'.format(time.time() - start))
    print(results)
    temp = [x[1] for x in results]
    print('Average cum returns: {}; std deviation: {}; avg num of episodes:{}'.format(sum(temp) / len(temp), np.std(temp), np.mean([x[3] for x in results])))


if __name__ == "__main__":
    fire.Fire(simple_param_search)
