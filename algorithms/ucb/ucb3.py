import math
import numpy as np


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

        self.counts = np.zeros(n_arms)

        self.values = np.zeros(n_arms)
        self.values_in_epoch = np.zeros([n_arms, self.log_horizon+1])
        self.plays_in_epoch = np.zeros([n_arms, self.log_horizon+1])
        return

    def calculate_bonus(self, p, t, arm, hot_start_shift_p):
        inverse_sum = 0
        for n_i in self.plays_in_epoch[arm, :]:
            if n_i > 0:
                inverse_sum += 1/(n_i+.0)

        return math.sqrt(self.alpha*math.log(t)*inverse_sum/((p-hot_start_shift_p+1)**2+.0))

    def select_arm(self, p, t, hot_start_shift_p):

        n_arms = len(self.counts)

        for arm in range(n_arms):
            if self.plays_in_epoch[arm, p] < self.hot_start_plays:
                return arm

        ucb_values = np.zeros(n_arms)
        bonus = np.zeros(n_arms)

        for arm in range(n_arms):

            bonus[arm] = self.calculate_bonus(p, t, arm, hot_start_shift_p)

            ucb_values[arm] = self.values[arm] + bonus[arm]

        return np.argmax(ucb_values)

    def update(self, chosen_arm, new_value, p):

        self.plays_in_epoch[chosen_arm, p] += 1

        self.values[chosen_arm] = new_value

        return