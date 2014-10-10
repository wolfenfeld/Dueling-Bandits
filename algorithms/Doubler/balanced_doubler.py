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

    return histogram/(sum(histogram)+.0)


def run_doubler_algorithm(means, log_horizon):
    """ run_doubler_algorithm() - This function runs the doubler / improved doubler algorithm and returns the
        cumulative regret. """

    # The number of arms
    n_arms = len(means)

    # The L set (initialized to [1,0,0,...,0])
    my_left_set = np.ones(n_arms)

    hot_start_plays = n_arms**2

    hot_start_shift_p = int(math.ceil(math.log(hot_start_plays*n_arms, 2)+2.0))
    print hot_start_shift_p

    # Assigning Bernoulli arms
    arms = map(lambda (mu): BernoulliArm(mu), means)

    # Assigning the black-boxes with the UCB 3 algorithm
    right_black_box = UCB3(counts=[], values=[], log_horizon=log_horizon, values_in_epoch=[], plays_in_epoch=[],
                           alpha=0.25, hot_start_plays=hot_start_plays)

    # Initializing the black-boxes.
    right_black_box.initialize(n_arms)

    # The b observation
    observed_b = np.zeros(2**log_horizon)

    # Regret and reward tracking
    average_reward = np.zeros(2**log_horizon)
    regret = np.zeros(2**log_horizon)
    cumulative_average_reward = np.zeros(2**log_horizon)
    cumulative_regret = np.zeros(2**log_horizon)

    # averages use to calculate the value that will be updated in the black box
    average = np.zeros([n_arms, log_horizon+1 - hot_start_shift_p + 1])
    average_of_averages = np.zeros(n_arms)

    best_score = 0

    # The algorithm :
    for current_p in range(int(hot_start_shift_p), log_horizon+1):

        # The arms used in this current round
        arms_histogram = np.zeros(n_arms)

        # This round time interval.
        current_time_interval = time_interval(current_p)

        current_epoch_total_values = np.zeros(n_arms)
        current_epoch_total_counts = np.zeros(n_arms)

        for t in current_time_interval:

            # Probability vector of last epoch left arm's
            probability_vector = construct_probability_vector(my_left_set)

            # Choosing the left arm from the multi-set L.
            left_arm = choose_from_probability_vector(probability_vector)

            # Choosing an arm using the right black box.
            # (I also take in consideration the arms used in the current epoch)
            right_arm = right_black_box.select_arm(current_p, t, hot_start_shift_p)

            # Updating the histogram only if all the hot start phase is over
            if sum(current_epoch_total_counts) >= hot_start_plays*n_arms:
                # Updating the histogram of the current epoch
                arms_histogram[right_arm] += 1

            # Acquiring that reward
            current_left_reward = arms[left_arm].draw()

            current_right_reward = arms[right_arm].draw()

            # Observing b_t
            b_t = observe_b_t(current_left_reward, current_right_reward)

            observed_b[t] = b_t

            current_epoch_total_values[right_arm] += observed_b[t]
            current_epoch_total_counts[right_arm] += 1

            average[right_arm, current_p - hot_start_shift_p] = \
                current_epoch_total_values[right_arm] / (current_epoch_total_counts[right_arm]+.0)

            average_of_averages[right_arm] = np.average(average[right_arm, 0:(current_p-hot_start_shift_p+1)])

            # Updating the right black-box with b_t and the bonus
            right_black_box.update(chosen_arm=right_arm, new_value=average_of_averages[right_arm], p=current_p)

            # Assigning the average reward.
            average_reward[t] = float(current_left_reward + current_right_reward) / 2

            # Assigning the regret
            regret[t] = max(means) - average_reward[t]

            # Assigning the cumulative regret and rewards
            if t == 0:
                cumulative_average_reward[0] = average_reward[t]

                cumulative_regret[0] = regret[t]
            else:
                cumulative_average_reward[t] = average_reward[t] + cumulative_average_reward[t-1]

                cumulative_regret[t] = regret[t] + cumulative_regret[t-1]

        current_score = sum(current_epoch_total_counts*current_epoch_total_values/(sum(current_epoch_total_counts)**2+.0))

        # Updating the left set of arms that can be used in the next round.
        if current_score > best_score:
            best_score = current_score
            my_left_set = np.array(arms_histogram)
        else:
            my_left_set = my_left_set

    return cumulative_regret


def run_several_iterations(iterations, means, horizon):
    """ run_several_iterations() - This function runs several iterations of the Doubler/Improved Doubler algorithm. """

    # Initializing the  results vector
    results = np.zeros(horizon)

    # log(horizon)
    log_horizon = int(math.log(horizon, 2))

    for iteration in range(iterations):
        print means
        # The current cumulative regret.
        results = np.add(results, run_doubler_algorithm(means[:, iteration], log_horizon=log_horizon))

    # Returning the average cumulative regret.
    return results/(iterations + .0)