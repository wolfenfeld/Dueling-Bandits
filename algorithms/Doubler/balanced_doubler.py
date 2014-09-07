import numpy as np
from arms.bernoulli import *
from algorithms.ucb.ucb3 import *
import random


def time_interval(p):
    """ time_interval() - This function returns the time interval T_p. """
    return [value for value in range(int(2**(p-1)), int(2**p))]


def observe_b_t(left_reward, right_reward):
    """ observe_b_t() - This function returns b_t."""

    random_variable = random.random()

    if random_variable >= (left_reward - right_reward + 1.0)/2:
        return 1
    else:
        return 0


def choose_from_probability_vector(probability_vector):
    """ choose_from_probability_vector() - This function receives a probability vector and returns the chosen index."""
    r = random.random()
    index = 0

    while r >= 0 and index < len(probability_vector):

        r -= probability_vector[index]

        index += 1

    return index - 1


def construct_probability_vector(histogram):
    """ construct_probability_vector() - This function returns the normalized histogram."""

    return np.divide(histogram, sum(histogram)+.0)


def run_doubler_algorithm(means, log_horizon):
    """ run_doubler_algorithm() - This function runs the doubler / improved doubler algorithm and returns the
        cumulative regret. """

    # The L set (initialized to [1,0,0,...,0])
    my_left_set = [False]*len(means)
    my_left_set[0] = 1

    # The number of arms
    n_arms = len(means)

    # Shuffling the means vector.
    random.shuffle(means)

    # Assigning Bernoulli arms
    arms = map(lambda (mu): BernoulliArm(mu), means)

    # Assigning the black-boxes with the UCB 3 algorithm
    #left_black_box = UCB3(counts=[], p_counts=log_horizon, values=[], alpha=0.5)
    right_black_box = UCB3(counts=[], values=[], alpha=0.75)

    # Initializing the black-boxes.
    right_black_box.initialize(n_arms)

    # The b observation
    observed_b = [0]*(2**log_horizon)

    # Regret and reward tracking
    average_reward = [0]*(2**log_horizon)
    regret = [0]*(2**log_horizon)
    cumulative_average_reward = [0]*(2**log_horizon)
    cumulative_regret = [0]*(2**log_horizon)

    # The record of all the arms used in each epoch
    arms_used = np.zeros([n_arms, log_horizon+1])

    # The Doubler algorithm :
    for current_p in range(1, log_horizon+1):

        # The arms used in this current round
        arms_histogram = [0]*n_arms

        # This round time interval.
        current_time_interval = time_interval(current_p)

        # Initializing the right black-box (S).

        right_black_box.initialize(n_arms)

        for t in current_time_interval:

            # Probability vector of last epoch left arm's
            probability_vector = construct_probability_vector(my_left_set)

            # Choosing the left arm from the multi-set L.
            left_arm = choose_from_probability_vector(probability_vector)

            # Choosing an arm using the right black box.
            # (I also take in consideration the arms used in the current epoch)
            right_arm = right_black_box.select_arm(current_p, t, arms_used)

            # Updating the record of used arms
            arms_used[right_arm, current_p] += 1

            # Updating the histogram of the current epoch
            arms_histogram[right_arm] += 1

            # Choosing the arms
            current_left_reward = arms[left_arm].draw()

            current_right_reward = arms[right_arm].draw()

            # Observing b_t
            observed_b[t] = observe_b_t(current_left_reward, current_right_reward)

            # Updating the right black-box with b_t and the bonus
            right_black_box.update(right_arm, observed_b[t]+right_black_box.calculate_bonus(current_p, t, arms_used, right_arm))

            # Assigning the average reward.
            average_reward[t] = float(current_left_reward + current_right_reward) / 2

            print "arms: {0}, {1}".format(left_arm, right_arm)
            # Assigning the regret
            regret[t] = max(means) - average_reward[t]

            # Assigning the cumulative regret and rewards
            if t == 1:
                cumulative_average_reward[t] = average_reward[t]

                cumulative_regret[t] = regret[t]
            else:
                cumulative_average_reward[t] = average_reward[t] + cumulative_average_reward[t-1]

                cumulative_regret[t] = regret[t] + cumulative_regret[t-1]

        # Updating the left set of arms that can be used in the next round.
        my_left_set = arms_histogram

    return cumulative_regret


def run_several_iterations(iterations, means, horizon):
    """ run_several_iterations() - This function runs several iterations of the Doubler/Improved Doubler algorithm. """

    # Initializing the  results vector
    results = [0]*horizon

    # log(horizon)
    log_horizon = int(np.log2(horizon))

    for iteration in range(iterations):

        # The current cumulative regret.
        results = np.add(results, run_doubler_algorithm(means, log_horizon=log_horizon))

    # Returning the average cumulative regret.
    return results/(iterations + .0)