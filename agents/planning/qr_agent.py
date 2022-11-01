import numpy as np
import torch as T
import torch.nn as nn
from agents.planning.base_agent import BaseAgent
from common.rep_utils import (LossFuncFactory, RepresentationFactory,
                              net_factory, replay_factory)
from common.replay.replay_buffer import Transition
from config import device


class QRAgent(BaseAgent):

    def agent_init(self, agent_init_info):
        allowed_attrs = {
            'num_quant',
        }
        self.__dict__.update((k, v) for k, v in agent_init_info.items() if k in allowed_attrs)
        # TODO: changed agent_init to __init__ with kwarg
        super().agent_init(agent_init_info)
        self.criterion = LossFuncFactory.get_loss_func(self.loss_func, True)
        self.tau = T.Tensor((2 * np.arange(self.num_quant) + 1) / (2.0 * self.num_quant)).view(1, -1)

    def _get_action(self, state):
        with T.no_grad():
            current_q = self.nn(self.rep_func(state).to(device)).view(-1, self.num_actions, self.num_quant).mean(2)
        current_q.squeeze_()
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        return action, current_q

    def set_buffer(self):
        self.buffer = replay_factory.create("basic", **self.__dict__)

    def train(self, trans, planning=False):
        state_batch = T.cat(trans.state)
        action_batch = T.LongTensor(trans.action).view(-1, 1).to(device)
        new_state_batch = T.cat(trans.new_state)
        reward_batch = T.FloatTensor(trans.reward).to(device)
        discount_batch = T.FloatTensor(trans.discount).to(device)

        # Preprocess observation
        state_batch = self.rep_func(state_batch)
        new_state_batch = self.rep_func(new_state_batch)
        current_q = self.nn(state_batch).view(-1, self.num_actions, self.num_quant)
        q_learning_action_values = current_q[np.arange(state_batch.size(0)), action_batch.view(-1)]
        with T.no_grad():
            if self.use_target:
                new_q = self.nn_target(new_state_batch).view(-1, self.num_actions, self.num_quant)
            else:
                new_q = self.nn(new_state_batch).view(-1, self.num_actions, self.num_quant)
        max_q = new_q[np.arange(state_batch.size(0)), new_q.mean(2).max(1)[1]]

        target = reward_batch.unsqueeze(-1) + discount_batch.unsqueeze(-1) * max_q
        loss = self.criterion(q_learning_action_values, target) * (self.tau - ((q_learning_action_values - target).detach() < 0).float()).abs()

        loss = loss.mean()
        if planning:
            loss = loss * (self.total_planning**(-1*self.beta))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
