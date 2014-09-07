import numpy as np
import random
from arms.fixed import *
from arms.bernoulli import *
import matplotlib.pyplot as plt


class RCS():

    def __init__(self, n_arms, alpha):

        self.alpha = alpha
        self.utility = np.zeros([n_arms, n_arms])
        self.wins = np.zeros([n_arms, n_arms])
        self.theta = np.zeros([n_arms, n_arms])
        self.champion = None
        self.b_vector = [False]*n_arms
        self.c_vector = [False]*n_arms
        return

    def initialize(self):
        n_arms = len(self.wins)

        for arm in range(n_arms):
            self.utility[arm, arm] = 0.5

        return

    def select_arms(self, t):

        # Updating the utility matrix
        self.update_utility_matrix(t)

        self.update_theta_matrix()

        # updating the c_champion
        self.set_champion()

        chosen_left_arm = self.champion

        chosen_right_arm = self.chose_right_arm(chosen_left_arm)

        return [chosen_left_arm, chosen_right_arm]

    def update_utility_matrix(self, t):

         # Number of arms
        n_arms = len(self.wins)

        # Updating the utility for this round according to the wins of all the arms.
        # For each left arms
        for left_arm in range(n_arms):

            for right_arm in range(n_arms):

                denominator = float(self.wins[left_arm, right_arm] + self.wins.T[left_arm, right_arm])

                if denominator == 0:

                    if self.wins[left_arm, right_arm] == 0:
                        exploitation = 0
                    else:
                        exploitation = 1

                    if t == 1:
                        exploration = 0
                    else:
                        exploration = 1

                else:

                    exploitation = float(self.wins[left_arm, right_arm])/denominator

                    exploration = np.sqrt(float(self.alpha * np.log(t))/denominator)

                self.utility[left_arm, right_arm] = exploration + exploitation

        for arm in range(n_arms):
            self.utility[arm, arm] = 0.5

    def update_theta_matrix(self):
        # Number of arms
        n_arms = len(self.wins)

        # Updating the theta matrix for this round according to the wins of all the arms.
        # For each left arms

        for arm in range(n_arms):
            self.theta[arm, arm] = 0.5

        for left_arm in range(n_arms):

            for right_arm in range(n_arms):

                if left_arm < right_arm:

                    self.theta[left_arm, right_arm] = \
                        random.betavariate(
                            alpha=(self.wins[left_arm, right_arm]+1),
                            beta=(self.wins[right_arm, left_arm]+1)
                        )

                    self.theta[right_arm, left_arm] = 1 - self.theta[left_arm, right_arm]

    def update_wins_matrix(self, chosen_left_arm, chosen_right_arm, better_arm):
        """ update_wins_matrix() updating the wins matrix according to the arm value. """

        if better_arm == "left":

            self.wins[chosen_left_arm, chosen_right_arm] += 1

        elif better_arm == "right":

            self.wins[chosen_right_arm, chosen_left_arm] += 1

        return

    def set_champion(self):

        n_arms = len(self.wins)

        # Looking for an arm that satisfies theta[left_arm][:] >= 1/2
        half_matrix = np.ones([n_arms, n_arms])*0.5

        greater_then_half_matrix = self.theta >= half_matrix

        for left_arm in range(n_arms):

            if np.all(greater_then_half_matrix[left_arm, :]):
                self.c_vector[left_arm] = True
            else:
                self.c_vector[left_arm] = False

        available_champions = np.where(self.c_vector)[0]

        if len(available_champions) > 0:
            self.champion = random.choice(available_champions)
        else:
            self.champion = self.get_least_played_arm()

    def get_least_played_arm(self):

        n_arms = len(self.wins)

        number_of_plays = np.zeros([n_arms])

        for left_arm in range(n_arms):
            for right_arm in range(n_arms):
                number_of_plays[left_arm] += self.wins[left_arm, right_arm]

        arms_least_played = np.where(number_of_plays == min(number_of_plays))[0]

        return random.choice(arms_least_played)

    def create_champion_set(self):

        n_arms = len(self.wins)

        # Looking for an arm that satisfies utility[left_arm][:] >= 1/2
        half_matrix = np.ones([n_arms, n_arms])*0.5

        greater_then_half_matrix = self.utility >= half_matrix

        for left_arm in range(n_arms):

            if np.all(greater_then_half_matrix[left_arm, :]):
                self.c_vector[left_arm] = True
            else:
                self.c_vector[left_arm] = False

    def chose_left_arm(self):

        n_arms = len(self.wins)

        number_of_optional_arms = np.sum(self.c_vector)

        #print number_of_optional_arms
        if number_of_optional_arms == 0:

            chosen_left_arm = random.choice(range(n_arms))
            self.b_vector = np.bitwise_and(self.b_vector, self.c_vector)

        elif number_of_optional_arms == 1:

            chosen_left_arm = int((np.where(self.c_vector))[0])
            self.b_vector[chosen_left_arm] = True

        else:
            chosen_left_arm = self.draw_arm_from_c()
            #print "Chosen left arm - {0} from champions.".format(chosen_left_arm)
        return chosen_left_arm

    def chose_right_arm(self, left_arm):

        u_jc = self.utility[:, left_arm]

        max_utility = max(u_jc)

        max_vector = (u_jc == max_utility)

        arms_with_max_value = (np.where(max_vector))[0]

        # We need to chose the right arm to be the argmax_j(u_jc) and keep the arms different from each-other.

        if (left_arm in arms_with_max_value) and (np.sum(max_vector) > 1):
            chosen_right_arm = left_arm

            while chosen_right_arm == left_arm:

                chosen_right_arm = random.choice(arms_with_max_value)

        # If the left arm does not hold the highest value:
        else:
            chosen_right_arm = random.choice(arms_with_max_value)

        return chosen_right_arm

    def chose_best_arm(self):

        n_arms = len(self.wins)

        wins = np.zeros([n_arms, n_arms])
        wins_ratio = np.zeros([n_arms, n_arms])

        for left_arm in range(n_arms):

            for right_arm in range(n_arms):

                if self.wins[left_arm, right_arm] + self.wins.T[left_arm, right_arm] == 0:

                    wins[left_arm, right_arm] = 0

                else:

                    wins_ratio[left_arm, right_arm] = (float(self.wins[left_arm, right_arm])/float(self.wins[left_arm, right_arm] + self.wins.T[left_arm, right_arm]))

                    wins[left_arm, right_arm] = wins_ratio[left_arm, right_arm] >= 0.5

        total_wins = [0]*n_arms

        for arm in range(n_arms):
            total_wins[arm] = np.sum(wins[arm, :])

        return np.argmax(total_wins)

    def draw_arm_from_c(self):

        inverted_b_vector = np.logical_not(self.b_vector)

        arms_only_in_c = np.bitwise_and(self.c_vector, inverted_b_vector)

        probability_vector = \
            np.multiply(0.5, self.b_vector) +\
            np.multiply(float((1/(float(2**(np.sum(self.b_vector)))*float(np.sum(arms_only_in_c))))), arms_only_in_c)

        chosen_arm = choose_from_probability_vector(probability_vector=probability_vector)

        return chosen_arm


def choose_from_probability_vector(probability_vector):

    r = random.random()
    index = 0

    while r >= 0 and index < len(probability_vector):

        r -= probability_vector[index]

        index += 1

    return index - 1


def run_rcs_algorithm(means, horizon):
    """ run_rcs_algorithm() - This function runs the RUCB algorithm. """

    # The number of arms.
    n_arms = len(means)

    # Shuffling the arms
    random.shuffle(means)

    # Initializing the the average reward vector.
    average_reward = [0]*horizon

    # Initializing the regret vector.
    regret = [0]*horizon

    # Initializing the cumulative average reward vector.
    cumulative_average_reward = [0]*horizon

    # Initializing the cumulative regret vector.
    cumulative_regret = [0]*horizon

    # Assigning the arms.
    arms = map(lambda (mu): BernoulliArm(mu), means)

    # Constructing the RUCB algorithm object.
    algorithm = RCS(n_arms=n_arms, alpha=0.5)

    # Initializing the algorithm.
    algorithm.initialize()

    for t in range(horizon):

        # Selecting the arms.
        [chosen_left_arm, chosen_right_arm] = algorithm.select_arms(t+1)

        # Obtaining the rewards.
        left_reward = arms[chosen_left_arm].draw()

        right_reward = arms[chosen_right_arm].draw()

        # Choosing the better arm.

        # Tie breaking rule.
        if left_reward == right_reward:

            better_arm = random.choice(["left", "right"])

        elif left_reward > right_reward:

            better_arm = "left"

        else:

            better_arm = "right"

        # Updating the wins matrix.
        algorithm.update_wins_matrix(chosen_left_arm, chosen_right_arm, better_arm)

        # Assigning the rewards and regrets.
        average_reward[t] = (right_reward + left_reward) / 2

        regret[t] = max(means) - average_reward[t]

        if t == 1:
            cumulative_average_reward[t] = average_reward[t]

            cumulative_regret[t] = regret[t]
        else:
            cumulative_average_reward[t] = average_reward[t] + cumulative_average_reward[t-1]

            cumulative_regret[t] = regret[t] + cumulative_regret[t-1]

    # Returning the cumulative regret.
    return cumulative_regret



def print_results(x_vector, y_vector):

    plt.plot(x_vector, y_vector, 'bs')
    plt.show()


def run_several_iterations(iterations, means, horizon):
    """ run_several_iterations() - This function runs several iterations of the RUCB algorithm."""

    # Initializing the results vector.
    results = [0]*horizon

    for iteration in range(iterations):

        # Running the algorithm for this iteration.
        current_results = run_rcs_algorithm(means, horizon)

        # Adding the regret.
        results = np.add(results, current_results)

    return results/(iterations +.0)

if __name__ == '__main__':
    my_means = [0.1, 0.9, 0.7, 0.3, 0.4]
    my_horizon = 1000
    my_iterations = 10
    average_results = np.divide(run_several_iterations(my_iterations, my_means, my_horizon),
                                float(my_iterations))
    print_results(range(my_horizon),average_results)