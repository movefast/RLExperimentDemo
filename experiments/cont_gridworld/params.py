import numpy as np


def get_lr(start=1e-2, a=2, n=5):
    return list(start/a**np.array(list(range(n))))


params_cont_gw_nn_1031 = {
    "Forward": {
        "step_size": get_lr(n=6, a=2, start=0.01),
        'seq_len': [5],
        'mom': [0, 0.9],
        'tarnetfreq':[10],
        'is_episodic_bound': [False],
    },
    "Backward": {
        "step_size": get_lr(n=6, a=2, start=0.01),
        'seq_len': [5],
        'mom': [0, 0.9],
        'tarnetfreq':[10],
        'is_episodic_bound': [False],
    },
}

params_cont_gw_tc_1031 = {
    "Forward": {
        "step_size": get_lr(n=6, a=2, start=0.5),
        'seq_len': [5],
        'opt': ['sgd'],
        'init': [0.0625],
        'tarnetfreq':[10],
        'is_episodic_bound': [False],
    },
    "Backward": {
        "step_size": get_lr(n=6, a=2, start=0.5),
        'seq_len': [5],
        'opt': ['sgd'],
        'init': [0.0625],
        'tarnetfreq':[10],
        'is_episodic_bound': [False],
    },
}
