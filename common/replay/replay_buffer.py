import itertools
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Any

import numpy as np

Trans = namedtuple('Transition', ('state', 'action', 'new_state', 'new_action', 'reward', 'discount', 'action_probs', 'next_action_probs'))

class Transition(Trans):
    __slots__ = ()
    def __new__(cls, state, action, new_state, new_action, reward, discount=None, action_probs=None, next_action_probs=None):
        return super(Transition, cls).__new__(cls, state, action, new_state, new_action, reward, discount, action_probs, next_action_probs)

class sliceable_deque(deque):
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(itertools.islice(self, index.start, index.stop, index.step))
        return deque.__getitem__(self, index)

class EpisodicMemory(object):

    def __init__(self, capacity=math.inf):
        self.capacity = capacity
        self.memory = sliceable_deque([])

    def add(self, *args):
        """Saves a transition."""
        if len(self.memory) == self.capacity:
            self.memory.popleft()
        self.memory.append(Transition(*args))

    def retrieve(self):
        return self.memory

    def clear(self):
        self.memory = sliceable_deque([])
        self.position = 0

    def __len__(self):
        return len(self.memory)

class Memory(object):

    def __init__(self, capacity, is_episodic_bound):
        self.capacity = capacity
        self.memory = sliceable_deque([])
        self.position = 0
        self.last_sampled_idxs = []
        self.is_episodic_bound = is_episodic_bound

    @property
    def num_ele(self):
        return len(self.memory)

    def add(self, *args):
        if len(self.memory) == self.capacity:
            self.memory.popleft()
        self.memory.append(Transition(*args))
        self.last_sampled_idxs = [i - 1 for i in self.last_sampled_idxs]

    def append_state(self, state):
        if len(self.memory) == self.capacity:
            self.memory.popleft()
        self.memory.append(state)

    def sample(self, batch_size):
        # same seed under random and np.random would result in different sequence
        # return random.sample(self.memory, batch_size)
        idxs = np.random.randint(0, len(self.memory), size=batch_size)
        return [self.memory[i] for i in idxs]

    # TODO: (does not support sequence sampling in episodic task right now; needs to handle timestep v.s. episode)
    def sample_sequence(self, batch_size, seq_len):
        assert len(self.memory) > seq_len, "we don't have long enough trajectory to sample from"
        start_idxes = np.random.choice(len(self.memory)-seq_len, batch_size)
        end_idxes = start_idxes + seq_len
        return [self.memory[slice(start, end)] for (start, end) in zip(start_idxes, end_idxes)]

    def sample_successive(self, seq_len):
        assert len(self.memory) > seq_len, "we don't have long enough trajectory to sample from"
        start_idx = np.random.choice(len(self.memory)-seq_len)
        end_idx = start_idx + seq_len
        return self.memory[slice(start_idx, end_idx)]

    def last_n(self, seq_len):
        assert len(self.memory) > seq_len, "we don't have long enough trajectory to sample from"
        end_idx = len(self.memory)
        start_idx = end_idx - seq_len
        return self.memory[slice(start_idx, end_idx)]

    def forward_sample(self, n):
        idxs = []
        for i in self.last_sampled_idxs:
            new_idx = i + 1
            if new_idx == len(self.memory) or new_idx < 0:
                continue
            idxs.append(new_idx)

        num_to_be_sampled = n - len(idxs)
        if num_to_be_sampled > 0:
            idxs.extend(np.random.randint(0, len(self.memory), size=num_to_be_sampled))
        if self.is_episodic_bound:
            self.last_sampled_idxs = [i for i in idxs if self.memory[i].discount != 0]
        else:
            self.last_sampled_idxs = idxs
        return [self.memory[i] for i in idxs]

    def backward_sample(self, n):
        idxs = []
        for i in self.last_sampled_idxs:
            new_idx = i - 1
            if new_idx < 0:
                continue
            idxs.append(new_idx)

        num_to_be_sampled = n - len(idxs)
        if num_to_be_sampled > 0:
            idxs.extend(np.random.randint(0, len(self.memory), size=num_to_be_sampled))
        if self.is_episodic_bound:
            self.last_sampled_idxs = [i for i in idxs if i > 0 and self.memory[i-1].discount != 0]
        else:
            self.last_sampled_idxs = [i for i in idxs if i > 0]
        return [self.memory[i] for i in idxs]

    def clear(self):
        self.memory = sliceable_deque([])
        self.position = 0

    def __len__(self):
        return len(self.memory)
