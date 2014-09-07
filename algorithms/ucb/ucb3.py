import math
import numpy as np
from scipy.stats import hmean


def ind_max(x):
    m = max(x)

    return x.index(m)


class UCB3():
    def __init__(self, counts, values, alpha=0.5):
        self.counts = counts
        self.values = values
        self.alpha = alpha
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def calculate_bonus(self, p, t, arms_used, arm):
        inverse_sum = 0
        for n_i in arms_used[arm, :]:
            if n_i > 0:
                inverse_sum += 1/(n_i+.0)

        return math.sqrt(self.alpha*math.log(t)*inverse_sum/(p**2+.0))

    def select_arm(self, p, t, arms_used):

        n_arms = len(self.counts)

        for arm in range(n_arms):
            if self.counts[arm] < 1:

            # if sum(arms_used[arm, :]) < 1:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        bonus = [0.0 for arm in range(n_arms)]

        for arm in range(n_arms):

            bonus[arm] = self.calculate_bonus(p, t, arms_used, arm)

            ucb_values[arm] = self.values[arm] + bonus[arm]

        return ind_max(ucb_values)

    def update(self, chosen_arm, reward):

        self.counts[chosen_arm] += 1

        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]

        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward

        self.values[chosen_arm] = new_value

        return
