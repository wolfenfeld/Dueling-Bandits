import math
import numpy as np
from scipy.stats import hmean


def ind_max(x):
    m = max(x)

    return x.index(m)


class UCB3():
    def __init__(self, counts, values, log_horizon, values_in_epoch, plays_in_epoch, alpha=0.5, hot_start_plays=5):
        self.counts = counts
        self.values = values
        self.alpha = alpha
        self.log_horizon = log_horizon
        self.values_in_epoch = values_in_epoch
        self.plays_in_epoch = plays_in_epoch
        self.hot_start_plays = hot_start_plays
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]

        self.values = [0.0 for col in range(n_arms)]
        self.values_in_epoch = np.zeros([n_arms, self.log_horizon+1])
        self.plays_in_epoch = np.zeros([n_arms, self.log_horizon+1])
        return

    def calculate_bonus(self, p, t, arm):
        inverse_sum = 0
        for n_i in self.plays_in_epoch[arm, :]:
            if n_i > 0:
                inverse_sum += 1/(n_i+.0)

        return math.sqrt(self.alpha*math.log(t)*inverse_sum/(p**2+.0))

    def select_arm(self, p, t):

        n_arms = len(self.counts)

        for arm in range(n_arms):
            if self.plays_in_epoch[arm, p] < self.hot_start_plays:

            # if sum(arms_used[arm, :]) < 1:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        bonus = [0.0 for arm in range(n_arms)]

        for arm in range(n_arms):

            bonus[arm] = self.calculate_bonus(p, t, arm)

            ucb_values[arm] = self.values[arm] + bonus[arm]

        return ind_max(ucb_values)

    def update(self, chosen_arm, reward, p):

        self.plays_in_epoch[chosen_arm, p] += 1
        self.values_in_epoch[chosen_arm, p] += reward

        new_value = self.calculate_value(chosen_arm, p)

        self.values[chosen_arm] = new_value

        return

    def calculate_value(self, arm, p):

        average_of_averages = 0

        for i in range(p+1):
            if self.plays_in_epoch[arm, i] != 0:
                average_of_averages += (self.values_in_epoch[arm, i] / (self.plays_in_epoch[arm, i]+.0))/(p+.0)

        return average_of_averages