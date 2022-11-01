import os
import sys

import numpy as np
import torch
from config import device


class ContinuousGridWorld:
    def __init__(self, env_info={}):
        self.s_noise = env_info.get("s_noise", 0)
        self.edge_scale = env_info.get("edge_scale", 1)
        self.step_len = env_info.get("step_len", 0.025)
        self.eps = 1e-5

        self.x = 0.0 * self.edge_scale + self.eps
        self.y = 0.5 * self.edge_scale

        self.goal_x = np.array([0.9, 1.0]) * self.edge_scale
        self.goal_y = np.array([0.9, 1.0]) * self.edge_scale

        self.wall_pos = np.array([[0.2, 0.3, 0.3, 0.9],
                                  [0.5, 0.6, 0.0, 0.4],
                                  [0.8, 0.9, 0.5, 1.0]]) * self.edge_scale # x_l, x_r, y_low, y_high
        self.wall_edges = self.generate_wall_edges()

        self.r_sparse = True
        return

    def seed(self, seed):
        pass

    def generate_wall_edges(self):
        vertical = []
        horizontal = []
        for pos in self.wall_pos:
            # up
            horizontal.append([pos[0], pos[1], pos[3], pos[3]]) # x_l, x_r, y_low, y_high
            # down
            horizontal.append([pos[0], pos[1], pos[2], pos[2]])
            # left
            vertical.append([pos[0], pos[0], pos[2], pos[3]])
            # right
            vertical.append([pos[1], pos[1], pos[2], pos[3]])
        return [horizontal, vertical]

    def check_pos(self, old_x, old_y, new_x, new_y):
        for hori_edge in self.wall_edges[0]:
            x_l, x_r, y, _ = hori_edge
            if (old_y < y and new_y > y) and x_l <= new_x <= x_r:
                new_y = y - self.eps
            elif (old_y > y and new_y < y) and x_l <= new_x <= x_r:
                new_y = y + self.eps
        for vert_edge in self.wall_edges[1]:
            x, _, y_low, y_high = vert_edge
            if (old_x < x and new_x > x) and y_low <= new_y <= y_high:
                new_x = x - self.eps
            elif (old_x > x and new_x < x) and y_low <= new_y <= y_high:
                new_x = x + self.eps
        return new_x, new_y

    def generate_obs(self):
        return np.array([self.x, self.y])

    def reset(self):
        self.x = 0.0 * self.edge_scale + self.eps
        self.y = 0.5 * self.edge_scale
        return self.env_start()

    def env_start(self, keep_history=False):
        return self.generate_obs()

    def step(self, action):
        x, y = self.move(action, self.x, self.y)
        self.x, self.y = x, y
        self.x = np.clip(self.x, 0.0 * self.edge_scale + self.eps, 1.0 * self.edge_scale - self.eps)
        self.y = np.clip(self.y, 0.0 * self.edge_scale + self.eps, 1.0 * self.edge_scale - self.eps)

        reward, terminate = self.check_goal()

        return self.generate_obs(), reward, terminate, {}

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        self.reset()

    def move(self, action, x, y):
        old_x, old_y = x, y
        # up
        if action == 0:
            y += self.step_len + np.random.normal(0, self.s_noise)
        # down
        elif action == 1:
            y -= self.step_len + np.random.normal(0, self.s_noise)
        # right
        elif action == 2:
            x += self.step_len + np.random.normal(0, self.s_noise)
        # left
        elif action == 3:
            x -= self.step_len + np.random.normal(0, self.s_noise)
        else:
            print("Environment: action out of range. Action is:", action)
        x, y = self.check_pos(old_x, old_y, x, y)
        return x, y

    def check_goal(self):
        if self.r_sparse:
            if self.x >= self.goal_x[0] and self.x <= self.goal_x[1]\
                    and self.y >= self.goal_y[0] and self.y <= self.goal_y[1]:
                return 1, 1
            else:
                return 0, 0
        else:
            if self.x >= self.goal_x[0] and self.x <= self.goal_x[1]\
                    and self.y >= self.goal_y[0] and self.y <= self.goal_y[1]:
                return 0, 1
            else:
                return -1, 0

    def get_state_dim(self):
        return 2

    def get_action_num(self):
        return 4


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import matplotlib.pyplot as plt

    env = ContinuousGridWorld(env_info={"s_noise":0.001, "edge_scale": 1, "step_len": 0.05})
    pos = []
    obs = env.reset().cpu().detach().numpy()
    pos.append(obs)
    for i in range(10000):
        a = np.random.randint(4)
        obs, r, t = env.step(a)
        pos.append(obs.cpu().detach().numpy())
        if t:
            env.reset()
            print("Reset")

    pos = np.array(pos)
    plt.figure()
    plt.scatter(pos[:, 0, 0], pos[:, 0, 1])
    plt.show()
