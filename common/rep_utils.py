from enum import IntEnum

import numpy as np
import torch.nn as nn

from common.nets import FCNN, LinearNN, SimpleNN
from common.replay.prioritized_replay_buffer import Memory as PER
from common.replay.replay_buffer import Memory as Replay

#####################
# Factory Interface #
#####################


class Factory():
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


##################
# Replay Factory #
##################


class PriorityTYPE(IntEnum):
    PER = 1
    GNORM = 2


def create_new_basic_replay(**kwargs):
    return Replay(capacity=kwargs["buffer_size"], is_episodic_bound=kwargs['is_episodic_bound'])

def create_new_per(**kwargs):
    return PER(kwargs['buffer_size'], True, alpha=kwargs['per_alpha'], beta=kwargs['buffer_beta'], beta_increment=kwargs['beta_increment'], min_weight=kwargs['min_weight'])


replay_factory = Factory()
replay_factory.register_builder('basic', create_new_basic_replay)
replay_factory.register_builder('per', create_new_per)


##########################
# Representation Factory #
##########################

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch as T
from config import device

import common.tile3 as tile3


class RepresentationFactory:

    @classmethod
    def __call__(cls, rep_type, rep_params):
        if rep_type is None:
            return Raw(rep_params)
        elif rep_type == "TC":
            return TileCoder(rep_params)
        elif rep_type == "OneHot":
            return OneHot2DRep(rep_params)


class Raw:
    def __init__(self, rep_params):
        return

    def __call__(self, input_):
        return input_

    def get_rep_size(self, input_dim):
        return input_dim


class TileCoder:
    def __init__(self, rep_params):
        self.mem = rep_params[0]
        self.num_tiling = rep_params[1]
        self.num_tile = rep_params[2]
        self.low_thrds = np.array(rep_params[3])
        self.high_thrds = np.array(rep_params[4])
        self.use_hashing = rep_params[5] if len(rep_params) > 5 else True
        self.tile_all_dim = rep_params[6] if len(rep_params) > 6 else False # True: tile coding all dimensions together; False: tile coding each dim separately

        self.normalize = (self.high_thrds - self.low_thrds) / self.num_tile
        if self.tile_all_dim:
            self.iht = tile3.IHT(self.mem)
        else:
            self.ihts = [tile3.IHT(self.mem) for _ in range(len(self.low_thrds))]

    def __call__(self, x):
        if self.tile_all_dim:
            tile_coded_features = T.zeros(x.size(0), self.mem)
            for i in range(x.size(0)):
                if self.use_hashing:
                    idxs = np.array(tile3.tiles(self.mem, self.num_tiling, [float(x[i][j].item() - self.low_thrds[j]) / self.normalize[j] for j in range(x.size(1))]))
                else:
                    idxs = np.array(tile3.tiles(self.iht, self.num_tiling, [float(x[i][j].item() - self.low_thrds[j]) / self.normalize[j] for j in range(x.size(1))]))
                tile_coded_features[i][idxs] = 1
        else:
            tile_coded_features = T.zeros(x.size(0), self.mem * x.size(1))
            for i in range(x.size(0)):
                for j in range(x.size(1)):
                    if self.use_hashing:
                        # 2) use hashing
                        idxs = np.array(tile3.tiles(self.mem, self.num_tiling, [float(x[i][j].item() - self.low_thrds[j]) / self.normalize[j]])) + j * self.mem
                    else:
                        # 1) use iht
                        idxs = np.array(tile3.tiles(self.ihts[j], self.num_tiling, [float(x[i][j].item() - self.low_thrds[j]) / self.normalize[j]])) + j * self.mem
                    tile_coded_features[i][idxs] = 1
        return tile_coded_features

    def get_rep_size(self, input_dim):
        if not self.tile_all_dim:
            return self.mem * input_dim
        else:
            return self.mem


class OneHot2DRep:
    def __init__(self, rep_params):
        self.maze_dim = rep_params
        self.num_states = self.maze_dim[0] * self.maze_dim[1]

    def get_observation(self, state):
        return state[0] * self.maze_dim[1] + state[1]

    def __call__(self, state):
        state_features = [
            T.eye(int(self.num_states))[self.get_observation(state[i])]
            for i in range(state.size(0))
        ]
        return T.stack(state_features).to(device)

    def get_rep_size(self, input_dim=None):
        return self.num_states

###################
# Network Factory #
###################
hidden_sizes = {
    "tiny": [16],
    "small": [32, 16],
    "med": [128, 64],
    "deep": [128, 64, 32, 16],
}

class NetworkFactory:
    @classmethod
    def create_value_network(cls, input_size, output_size, bias, net_type, **kwds):
        if net_type is None:
            return SimpleNN(input_size, output_size, bias)
        elif net_type in hidden_sizes:
            return FCNN(input_size, output_size, hidden_sizes[net_type], nn.ReLU, bias)
        elif net_type == "linear":
            return LinearNN(input_size, output_size, **kwds)
        else:
            raise NotImplementedError

net_factory = NetworkFactory()

class LossFuncFactory:
    @classmethod
    def get_loss_func(cls, name, is_not_reduce=False):
        return {
            "Huber": T.nn.SmoothL1Loss(reduction='none' if is_not_reduce else 'mean'),
            "L2": T.nn.MSELoss(reduction='none' if is_not_reduce else 'mean'),
        }[name]
