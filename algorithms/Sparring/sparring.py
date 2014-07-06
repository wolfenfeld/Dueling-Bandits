import numpy as np
from arms.bernoulli import *
from algorithms.ucb.ucb1 import *


def observe_b_t(left_reward, right_reward):
    """ observe_b_t() - This function returns b_t."""

    random_variable = random.random()

    if random_variable >= (left_reward - right_reward + 1.0)/2:
        return 1
    else:
        return 0


def run_sparring_algorithm(means, horizon):
    """ run_sparring_algorithm() - This function runs the Sparring algorithm. """

    # The number of arms
    n_arms = len(means)

    # Shuffling the means vector.
    random.shuffle(means)

    # Assigning Bernoulli arms
    arms = map(lambda (mu): BernoulliArm(mu), means)

    # Assigning the black-boxes with the UCB 1 algorithm
    left_black_box = UCB1([], [])
    right_black_box = UCB1([], [])

    # Initializing the black-boxes.
    left_black_box.initialize(n_arms)
    right_black_box.initialize(n_arms)

    # Initializing rewards and regrets
    average_reward = [0]*horizon

    regret = [0]*horizon

    cumulative_average_reward = [0]*horizon

    cumulative_regret = [0]*horizon

    for t in range(horizon):

        # Using the black-boxes to select the arms
        left_arm = left_black_box.select_arm()
        right_arm = right_black_box.select_arm()

        # Acquiring the rewards
        left_reward = arms[left_arm].draw()

        right_reward = arms[right_arm].draw()

        b = observe_b_t(left_reward, right_reward)
        b_not = 1 - b

        # Updating the black-boxes
        left_black_box.update(left_arm, b_not)
        right_black_box.update(right_arm, b)

        # Assigning the average reward.
        average_reward[t] = float(right_reward + left_reward) / 2

        # Assigning the regret
        regret[t] = max(means) - average_reward[t]

        # Assigning the cumulative regret and rewards
        if t == 1:
            cumulative_average_reward[t] = average_reward[t]

            cumulative_regret[t] = regret[t]
        else:
            cumulative_average_reward[t] = average_reward[t] + cumulative_average_reward[t-1]

            cumulative_regret[t] = regret[t] + cumulative_regret[t-1]

    # Returning the average regret.
    return cumulative_regret


def run_several_iterations(iterations, means, horizon):
    """ test_several_iterations() - This function runs several iterations of the Sparring algorithm. """

    # Initializing the results vector.
    results = [0]*horizon

    for iteration in range(iterations):

        # The current cumulative regret.
        results = np.add(results, run_sparring_algorithm(means, horizon))

    # Returning the average cumulative regret.
    return results/(iterations +.0)