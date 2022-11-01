import numpy as np


def get_lr(start=1e-2, a=2, n=5):
    return list(start/a**np.array(list(range(n))))

params_pw_tc_search_1031 = {
    "QLearning": {
        "step_size": get_lr(n=4, a=2, start=1),
        'seq_len': [5],
        'opt': ['sgd'],
        'init': [0],
        'tarnetfreq':[1],
    },
    "QR": {
        "step_size": get_lr(n=4, a=2, start=1),
        'seq_len': [5],
        'opt': ['sgd'],
        'init': [0],
        'tarnetfreq':[1],
        'num_quant': [2],
    },
    "PER": {
        "step_size": get_lr(n=4, a=2, start=1),
        'seq_len': [5],
        'ptype': [1],
        'per_alpha': [0.6, 0.4, 0.2],
        'importance_sampling': [True],
        'opt': ['sgd'],
        'init': [0],
        'tarnetfreq':[1],
    },
}

params_pw_tc_1031 = {
    "QLearning": {
        "step_size": [0.5],
        'seq_len': [5],
        'opt': ['sgd'],
        'init': [0],
        'tarnetfreq':[1],
    },
    # "QR": {
    #     "step_size": [1],
    #     'seq_len': [5],
    #     'opt': ['sgd'],
    #     'init': [0],
    #     'tarnetfreq':[1],
    #     'num_quant': [2],
    # },
    # "PER": {
    #     "step_size": [0.125],
    #     'seq_len': [5],
    #     'ptype': [1],
    #     'per_alpha': [0.2],
    #     'importance_sampling': [True],
    #     'opt': ['sgd'],
    #     'init': [0],
    #     'tarnetfreq':[1],
    # },
}
