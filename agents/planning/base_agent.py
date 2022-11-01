import numpy as np
import torch as T
from agents.agent import AbstractBaseAgent
from common.agent_helpers import (LossFuncFactory, RepresentationFactory,
                                  net_factory, replay_factory)
from common.replay.replay_buffer import Transition
from config import device


class BaseAgent(AbstractBaseAgent):
    def agent_init(self, agent_init_info):
        # Store the parameters provided in agent_init_info.
        allowed_attrs = {
            # generic
            'num_actions',
            'num_states',
            'epsilon',
            'step_size',
            'discount',
            'batch_size',
            'seq_len',
            'update_interval',
            'bias',
            'net_type', 'opt', 'loss_func',
            ## buffer
            'buffer_size',
            # representation
            'rep_type',
            'rep_param',

            'total_planning',
            'beta',

            'mom',
            'secmo',

            'is_episodic_bound',

            'init',
            'tarnetfreq',
        }
        self.__dict__.update((k, v) for k, v in agent_init_info.items() if k in allowed_attrs)
        self.rep_func = RepresentationFactory()(self.rep_type, self.rep_param)
        if self.rep_type == "TC" and self.net_type == "linear":
            self.step_size /= self.rep_param[1] # if using tile coding, then divide learning rate by number of tilings
            if not self.rep_func.tile_all_dim:
                self.step_size /= len(self.rep_func.ihts)
        elif self.rep_type == "LunarLander":
            self.step_size /= self.rep_param[1] * len(self.rep_func.dims)

        self.rep_dim = self.rep_func.get_rep_size(self.num_states)

        init = self.__dict__.get('init') if 'init' in self.__dict__ else None
        num_output = self.num_actions * self.num_quant if 'num_quant' in self.__dict__ else self.num_actions

        self.nn = net_factory.create_value_network(self.rep_dim, num_output, self.bias, net_type=self.__dict__.get("net_type"), init=init).to(device)
        self.nn_target = net_factory.create_value_network(self.rep_dim, num_output, self.bias, net_type=self.__dict__.get("net_type")).to(device)
        self.nn_target.load_state_dict(self.nn.state_dict())
        update_freq = self.__dict__.get("tarnetfreq")
        if not update_freq or update_freq == 0:
            self.use_target = False
        else:
            self.nn_tar_update_freq = update_freq
            self.use_target = True

        if self.__dict__.get("opt") is None or self.__dict__.get("opt") == "adam":
            mom = self.__dict__.get("mom") if self.__dict__.get("mom") is not None else 0.9
            secmo = self.__dict__.get("secmo") if self.__dict__.get("secmo") is not None else 0.999
            self.optimizer = T.optim.Adam(self.nn.parameters(), lr=self.step_size, betas=(mom, secmo))
        elif self.__dict__.get("opt") == "rmsprop":
            self.optimizer = T.optim.RMSprop(self.nn.parameters(), lr=self.step_size)
        elif self.__dict__.get("opt") == "sgd":
            self.optimizer = T.optim.SGD(self.nn.parameters(), lr=self.step_size)
        else:
            raise NotImplementedError

        self.criterion = LossFuncFactory.get_loss_func(self.loss_func)
        self.set_buffer()
        self.updates = 0
        self.steps = 0

    def set_buffer(self):
        raise NotImplementedError

    def _get_action(self, state):
        with T.no_grad():
            current_q = self.nn(self.rep_func(state).to(device))
        current_q.squeeze_()
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        return action, current_q

    def _cache(self, state, action):
        self.prev_state = state
        self.prev_action = action

    def _plan(self):
        if self.is_ready_to_plan():
            self.plan()
        if self.use_target and self.steps % self.nn_tar_update_freq == 0:
            self.nn_target.load_state_dict(self.nn.state_dict())
        self.steps += 1

    def agent_start(self, state):
        action, _ = self._get_action(state)
        self._cache(state, action)
        self._plan()
        return action

    def agent_step(self, reward, state):
        action, _ = self._get_action(state)
        self.train_online(self.prev_state, self.prev_action, state, reward, self.discount)

        self.buffer.add(self.prev_state, self.prev_action, state, action, reward, self.discount)

        self._cache(state, action)
        self._plan()
        return action

    def agent_end(self, reward, state, append_buffer=True):
        self.train_online(self.prev_state, self.prev_action, state, reward, 0 if append_buffer else self.discount)
        if append_buffer:
            self.buffer.add(self.prev_state, self.prev_action, state, 0, reward, 0)
        self._plan()

    def is_ready_to_plan(self):
        return (
            len(self.buffer) > self.batch_size
            and self.steps % self.update_interval == 0
        )

    def train_online(self, state, action, new_state, reawrd, discount):
        cur_tran = Transition([state], [action], [new_state], None, [reawrd], [discount])
        self.train(cur_tran)

    def plan(self):
        total = 0
        while total < self.total_planning:
            for _ in range(self.seq_len):
                self.updates += 1
                total += 1
                self.nn.train()
                transitions = self.buffer.sample(self.batch_size)
                batch = Transition(*zip(*transitions))
                self.train(batch, planning=True)

    def train(self, trans, planning=False):
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
        loss = self.criterion(q_learning_action_values.view(-1), target)
        if planning:
            loss = loss * (self.total_planning**(-1*self.beta))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def argmax(self, q_values):
        """argmax with random tie-breaking."""
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)

class QLearningAgent(BaseAgent):

    def set_buffer(self):
        self.buffer = replay_factory.create("basic", **self.__dict__)
