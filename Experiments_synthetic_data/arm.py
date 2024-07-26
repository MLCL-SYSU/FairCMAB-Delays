# -*- coding: utf-8 -*-
import numpy as np
import math

class Arm():
    def __init__(self, reward_type, delay_type, mean, p=0, sigma=1, m=1):
        self.mean = mean
        self.sigma = sigma
        self.p = p
        self.reward_type = reward_type
        self.delay_type = delay_type
        self.m=1

    def pull(self,arm_index=0):
        if self.reward_type == 'Gaussian_reward':
            reward = np.random.normal(self.mean, self.sigma)
        elif self.reward_type == 'Bernoulli_reward':
            reward = np.random.binomial(n=1, p=self.mean, size=1)[0]

        if self.delay_type == 'Packetloss_delay':
            if np.random.uniform(0,1) < self.p:
                delay = 0
            else:
                delay = math.inf
        elif self.delay_type == 'Geometric_delay':
            delay = np.random.geometric(self.p, 1)[0]
        elif self.delay_type == 'Fixed_delay':
            delay = self.p
        elif self.delay_type == 'Pareto_delay':
            delay = (np.random.pareto(self.p, None) + 1) * self.m
        elif self.delay_type == 'Undelayed':
            delay = 0
        elif self.delay_type == 'Fixed_delay_RD':
            if arm_index == 2 or  arm_index == 3 or  arm_index == 4:
                if reward == 1:
                    delay = self.p
                else:
                    delay = 0
            else:
                if reward == 0:
                    delay = self.p
                else:
                    delay = 0
        return reward, delay

    def get_mean(self):
        return self.mean

    def __str__(self):
        return f"{self.type} Arm with mean = {self.mean}"