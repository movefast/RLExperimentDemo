import sys

import numpy as np
import torch as T
from common.replay.replay_buffer import Transition
from config import device

epsilon = sys.float_info.epsilon

class Memory:

    def __init__(self, capacity, prioritized, **kwargs):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.num_ele = 0
        self.prioritized = prioritized
        self.beta = kwargs.get('beta', 1)
        self.beta_increment_per_sampling = kwargs.get('beta_increment')
        if self.prioritized:
            self.min_weight = kwargs['min_weight']
            self.a = kwargs['alpha']
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.max_priority = 1

        self.last_sampled_idxs = []

    def has_terminal_state(self):
        return len(self.is_terminal.nonzero()[0]) > 0

    def maxp(self):
        return np.max(self.tree.nodes[-1])

    def _get_priority(self, error):
        return (np.abs(error) + epsilon) ** self.a

    def __len__(self):
        return self.num_ele

    def add(self, error, *args):
        self.data[self.ptr] = Transition(*args)

        priority = self._get_priority(error)
        priority = np.maximum(priority, self.min_weight)
        self.tree.set(self.ptr, priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.num_ele = min(self.num_ele + 1, self.capacity)

    def update(self, ind, priority):
        priority = self._get_priority(priority)
        priority = np.maximum(priority, self.min_weight)
        self.max_priority = max(priority, self.max_priority)
        self.tree.set(ind, priority)

    def batch_update(self, ind, priority):
        priority = self._get_priority(priority)
        self.max_priority = max(priority.max().item(), self.max_priority)
        priority = np.maximum(priority, self.min_weight)
        self.tree.batch_set(ind, priority)

    def batch_update_trace(self, ind, traces):
        self.traces[ind] = traces

    def sample(self, n):
        if self.prioritized:
            sampled_idxs = list(self.tree.sample(n))
        else:
            sampled_idxs = np.random.randint(0, self.num_ele, size=n)
        sampled_idxs = [idx for idx in sampled_idxs if idx != len(self.tree.nodes[-1])-1]

        if self.prioritized:
            weights = (np.array(self.tree.nodes[-1][sampled_idxs]) + epsilon) ** -self.beta
            weights /= weights.max() + epsilon
        else:
            weights = np.ones(len(sampled_idxs))
        if self.beta_increment_per_sampling:
            self.beta = min(self.beta + self.beta_increment_per_sampling, 1) # Hardcoded: 0.4 + 2e-7 * 3e6 = 1.0. Only used by PE
        return (self.data[sampled_idxs], sampled_idxs, T.FloatTensor(weights).to(device).reshape(-1, 1))


class SumTree(object):
    def __init__(self, max_size):
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2

    def set(self, node_index, new_priority):
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    def batch_set(self, node_index, new_priority):
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index
    def sample(self, batch_size):
        if batch_size > 1:
            # systematic
            rngs = np.linspace(0,self.nodes[0][0],batch_size+1)
            query_value = np.array([np.random.uniform(*rngs[slice(i,i+2)]) for i in range(batch_size)])
        else:
            # regular
            query_value = np.random.uniform(0, self.nodes[0][0], size=batch_size)
        node_index = np.zeros(batch_size, dtype=int)

        for nodes in self.nodes[1:]:
            node_index *= 2
            left_sum = nodes[node_index]

            is_greater = np.greater(query_value, left_sum)
            # If query_value > left_sum -> go right (+1), else go left (+0)
            node_index += is_greater
            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            query_value -= left_sum * is_greater

        return node_index
