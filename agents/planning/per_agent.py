import numpy as np
import torch as T
import torch.nn.functional as F
from agents.planning.base_agent import BaseAgent
from common.rep_utils import (LossFuncFactory, PriorityTYPE,
                                       replay_factory)
from common.replay.replay_buffer import Transition
from config import device
from opacus.grad_sample import GradSampleModule


class PERAgent(BaseAgent):

    def agent_init(self, agent_init_info):
        allowed_attrs = {
            # PER specific
            'ptype',
            'beta_increment',
            'importance_sampling',
            ## buffer
            'per_alpha',
            'buffer_beta',
            'min_weight',
        }
        self.__dict__.update((k, v) for k, v in agent_init_info.items() if k in allowed_attrs)
        # TODO: changed agent_init to __init__ with kwarg
        super().agent_init(agent_init_info)
        self.criterion = LossFuncFactory.get_loss_func(self.loss_func, self.importance_sampling)
        if self.ptype == PriorityTYPE.GNORM and self.batch_size > 1:
            self.nn = GradSampleModule(self.nn)
            update_freq = self.__dict__.get("tarnetfreq")
            if update_freq and update_freq != 0:
                self.nn_target = GradSampleModule(self.nn_target)

    def set_buffer(self):
        self.buffer = replay_factory.create("per", **self.__dict__)

    def _cache(self, state, action, q):
        self.prev_state = state
        self.prev_action = action
        self.prev_action_value = q[action]

    def agent_start(self, state):
        # Choose action using epsilon greedy.
        action, current_q = self._get_action(state)
        self._cache(state, action, current_q)
        return action

    def agent_step(self, reward, state):
        # Choose action using epsilon greedy.
        action, current_q = self._get_action(state)
        grad_norm = self.train_online(self.prev_state, self.prev_action, state, reward, self.discount)
        if self.ptype == PriorityTYPE.PER:
            error = T.abs(self.prev_action_value - reward - self.discount * current_q.max()).item()
        elif self.ptype == PriorityTYPE.GNORM:
            error = grad_norm
        self.buffer.add(error, self.prev_state, self.prev_action, state, action, reward, self.discount)

        self._cache(state, action, current_q)
        self._plan()
        return action

    def agent_end(self, reward, state, append_buffer=True):
        temp = self.train_online(self.prev_state, self.prev_action, state, reward, 0 if append_buffer else self.discount)

        if self.ptype == PriorityTYPE.PER:
            error = T.abs(self.prev_action_value - reward).item()
        elif self.ptype == PriorityTYPE.GNORM:
            error = temp
        else:
            raise NotImplementedError

        if append_buffer:
            self.buffer.add(error, self.prev_state, self.prev_action, state, 0, reward, 0)
        else:
            with T.no_grad():
                current_q = self.nn(self.rep_func(state).to(device))
            current_q.squeeze_()
            error = T.abs(self.prev_action_value - reward - self.discount * current_q.max()).item()
            # ---> Here we use inf action to denote early term
            self.buffer.add(error, self.prev_state, self.prev_action, state, np.inf, reward, self.discount)
        self._plan()

    def train_online(self, state, action, new_state, reawrd, discount):
        cur_tran = Transition([state], [action], [new_state], None, [reawrd], [discount])
        return self.train(cur_tran)

    def plan(self):
        total = 0
        while total < self.total_planning:
            self.updates += 1
            total += 1
            self.nn.train()
            transitions, idxs, is_weight = self.buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            self.train(batch, idxs, is_weight, planning=True)

    def train(self, trans, idxs=None, is_weight=None, planning=False):
        state_batch = T.cat(trans.state)
        action_batch = T.LongTensor(trans.action).view(-1, 1).to(device)
        new_state_batch = T.cat(trans.new_state)
        reward_batch = T.FloatTensor(trans.reward).to(device)
        discount_batch = T.FloatTensor(trans.discount).to(device)

        state_batch = self.rep_func(state_batch)
        new_state_batch = self.rep_func(new_state_batch)

        current_q = self.nn(state_batch)
        q_learning_action_values = current_q.gather(1, action_batch)
        with T.no_grad():
            if self.use_target:
                new_q = self.nn_target(new_state_batch)
            else:
                new_q = self.nn(new_state_batch)
        max_q = new_q.max(1)[0]
        target = reward_batch
        target += discount_batch * max_q

        online = idxs is None
        if online or not self.importance_sampling:
            loss = self.criterion(q_learning_action_values.view(-1), target)
        else:
            temp = self.criterion(q_learning_action_values.view(-1), target)
            loss = temp @ is_weight

        # TODO: to be removed if beta is no longer used
        if planning:
            loss = loss * (self.total_planning**(-1*self.beta))

        self.optimizer.zero_grad()
        if self.ptype == PriorityTYPE.GNORM and self.batch_size > 1:
            self.nn.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.ptype == PriorityTYPE.GNORM:
            if online:
                grad_norm_sqrd = sum(param.grad.data.norm(2).item() ** 2 for param in self.nn.parameters())
                return grad_norm_sqrd ** 0.5
            else:
                if self.batch_size > 1:
                    grad_norms = T.zeros(self.batch_size)
                    for param in self.nn.parameters():
                        grad_norms += param.grad_sample.norm(2, dim=(-2,-1)) ** 2
                    # TODO: to be removed if beta is no longer used
                    grad_norms *= self.total_planning**(2 * self.beta)
                    self.buffer.batch_update(idxs, T.sqrt(grad_norms))
                else:
                    error = sum(
                        param.grad.data.norm(2).item() ** 2
                        for param in self.nn.parameters()
                    )
                    # TODO: to be removed if beta is no longer used
                    error *= self.total_planning**(2 * self.beta)
                    self.buffer.update(idxs[0], error ** 0.5)
        elif self.ptype == PriorityTYPE.PER:
            if planning:
                with T.no_grad():
                    errors = T.abs(q_learning_action_values.view(-1) - target).cpu().numpy()
                    self.buffer.batch_update(idxs, errors)
        else:
            raise NotImplementedError
