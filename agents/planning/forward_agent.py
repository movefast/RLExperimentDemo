import numpy as np
import torch as T
from agents.planning.base_agent import QLearningAgent
from common.replay.replay_buffer import Transition
from config import device


class ForwardAgent(QLearningAgent):

    def agent_init(self, agent_init_info):
        # Store the parameters provided in agent_init_info.
        allowed_attrs = {
            # forward specific
            'replay_across_iter',
        }
        self.__dict__.update((k, v) for k, v in agent_init_info.items() if k in allowed_attrs)

        super().agent_init(agent_init_info)

    def plan(self):
        total = 0
        while total < self.total_planning:
            for _ in range(self.seq_len):
                self.updates += 1
                total += 1
                self.nn.train()
                transitions = self.buffer.forward_sample(self.batch_size)
                batch = Transition(*zip(*transitions))
                self.train(batch, planning=True)
            if not self.replay_across_iter:
                self.buffer.last_sampled_idxs = []


class BackwardAgent(QLearningAgent):

    def agent_init(self, agent_init_info):
        allowed_attrs = {
            'replay_across_iter',
        }
        self.__dict__.update((k, v) for k, v in agent_init_info.items() if k in allowed_attrs)

        super().agent_init(agent_init_info)

    def plan(self):
        total = 0
        while total < self.total_planning:
            for _ in range(self.seq_len):
                self.updates += self.batch_size
                total += 1
                self.nn.train()
                transitions = self.buffer.backward_sample(self.batch_size)
                batch = Transition(*zip(*transitions))
                self.train(batch, planning=True)
            if not self.replay_across_iter:
                self.buffer.last_sampled_idxs = []
