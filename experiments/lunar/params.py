import numpy as np


def get_lr(start=1e-2, a=2, n=5):
    return list(start/a**np.array(list(range(n))))

params_lunar_nn_search_1031 = {
    "QLearning": {
        "step_size": get_lr(n=4, a=10, start=0.01),
        'seq_len': [5],
        'mom': [0, 0.9],
        'tarnetfreq':[10],
    },
    "QR": {
        "step_size": get_lr(n=4, a=10, start=0.01),
        'seq_len': [5],
        'mom': [0, 0.9],
        'tarnetfreq':[10],
        'num_quant': [2,4],
    },
    "PER": {
        "step_size": get_lr(n=4, a=10, start=0.01),
        'seq_len': [5],
        'ptype': [1, 2],
        'per_alpha': [1, 0.8, 0.6, 0.4, 0.2],
        'importance_sampling': [True],
        'mom': [0, 0.9],
        'tarnetfreq':[10],
    },
}
