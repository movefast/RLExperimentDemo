import collections
import concurrent.futures as cf
import glob
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import fire
import joblib
import torch

import config

ROOT_DIR = pathlib.Path(os.path.split(os.path.realpath(__file__))[0])

SKIP_METRICS = []


def dict_merge(dct, merge_dct):
    """ Recursive dict merge.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            if k in SKIP_METRICS:
                continue
            dict_merge(dct[k], merge_dct[k])
        elif k in dct and isinstance(dct[k], list) and isinstance(v, list):
            dct[k].extend(merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def combine_results(exp, file_name):
    today = date.today().strftime("%m_%d")
    metrics = config.get_empty_metrics(episodic_exp=False)
    files = glob.glob(str(ROOT_DIR/f'experiments/{exp}/metrics/*'))
    num_worker = os.cpu_count()//2

    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        for m in filter(None, executor.map(load, files)):
            dict_merge(metrics, m)
        (ROOT_DIR/"metrics").mkdir(parents=True, exist_ok=True)
        import pickle
        pickle.dump(metrics, open(ROOT_DIR/f'metrics/metrics_{today}_{file_name}.torch', 'wb'), protocol=4)

def load(file):
    if not os.path.isfile(file):
        return None
    try:
        return torch.load(file)
    except Exception:
        return joblib.load(file)


if __name__ == '__main__':
    fire.Fire(combine_results)
