import torch as T
from agents.planning.base_agent import BaseAgent
from common.agent_helpers import (LossFuncFactory, RepresentationFactory,
                              net_factory, replay_factory)
from config import device


class EsarsaLambdaAgent():
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
        if self.rep_type != "TC" or self.net_type != "linear":
            raise NotImplementedError

        self.step_size /= self.rep_param[1]  # if using tile coding, then divide learning rate by number of tilings
        if not self.rep_func.tile_all_dim:
            self.step_size /= len(self.rep_func.ihts)
        self.rep_dim = self.rep_func.get_rep_size(self.num_states)

        kwds = {}
        if 'init' in self.__dict__:
            kwds['init'] = self.__dict__.get('init')

        self.nn = net_factory.create_value_network(self.rep_dim, self.num_actions, self.bias, net_type=self.__dict__.get("net_type"), **kwds).to(device)
        self.nn_target = net_factory.create_value_network(self.rep_dim, self.num_actions, self.bias, net_type=self.__dict__.get("net_type"), **kwds).to(device)
        self.nn_target.load_state_dict(self.nn.state_dict())
        self.etrace = T.zeros(self.num_actions, self.rep_dim).to(device)

        update_freq = self.__dict__.get("tarnetfreq")
        if not update_freq or update_freq == 0:
            self.use_target = False
        else:
            self.nn_tar_update_freq = update_freq
            self.use_target = True

        # 1) tabular
        # self.sampled_state = np.zeros(self.num_states)
        # 2) cont 1 x 1 grid
        # self.sampled_state = T.zeros(20,20)
        self.updates = 0
        self.steps = 0

    def agent_start(self, state):
        # Choose action using epsilon greedy.
        action, _ = self._get_action(state)
        self._cache(state, action)
        self.etrace = T.zeros_like(self.etrace).to(device)
        return action

    def agent_step(self, reward, state):
        # Choose action using epsilon greedy.
        action, _ = self._get_action(state)
        self.train_online(self.prev_state, self.prev_action, state, reward, self.discount)

        self._cache(state, action)
        return action

    def agent_end(self, reward, state, append_buffer=True):
        self.train_online(self.prev_state, self.prev_action, state, reward, 0 if append_buffer else self.discount)

    def _update_target_net(self):
        self.steps += 1
        if self.use_target and self.steps % self.nn_tar_update_freq == 0:
            self.nn_target.load_state_dict(self.nn.state_dict())

    def train(self, trans, planning=False):
        if planning:
            raise NotImplementedError
        state_batch = T.cat(trans.state)
        action_batch = T.LongTensor(trans.action).view(-1, 1).to(device)
        new_state_batch = T.cat(trans.new_state)
        reward_batch = T.FloatTensor(trans.reward).to(device)
        discount_batch = T.FloatTensor(trans.discount).to(device)

        state_batch = self.rep_func(state_batch)
        new_state_batch = self.rep_func(new_state_batch)
        active_tiles = state_batch.squeeze().bool()
        self.etrace *= self.discount * 0.9
        self.etrace[trans.action, active_tiles] = 1

        with T.no_grad():
            current_q = self.nn(state_batch)
            q_learning_action_values = current_q.gather(1, action_batch).squeeze()
            if self.use_target:
                new_q = self.nn_target(new_state_batch)
            else:
                new_q = self.nn(new_state_batch)
            expected_q = self._get_pi(new_q.squeeze()).dot(new_q.squeeze())
            # max_q = new_q.max(1)[0]
            target = reward_batch
            target += discount_batch * expected_q
            td_error = target - q_learning_action_values
            self.nn.i2o.weight.data += self.step_size * td_error * self.etrace
        self._update_target_net()

    def _get_pi(self, q):
        pi_b = T.zeros(self.num_actions)
        pi_b[self.argmax(q)] = 1 - self.epsilon
        pi_b += self.epsilon / self.num_actions
        return pi_b
