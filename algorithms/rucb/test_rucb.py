from arms.fixed import *

# Definitions of bandit algorithms
from algorithms.rucb.rucb import *


# Convenience functions
def ind_max(x):
    m = max(x)
    return x.index(m)


def test_rucb_algorithm(algo, arms, num_sims, horizon):
    chosen_left_arms = [0.0 for ii in range(num_sims * horizon)]
    chosen_right_arms = [0.0 for ii in range(num_sims * horizon)]
    left_rewards = [0.0 for ii in range(num_sims * horizon)]
    right_rewards = [0.0 for ii in range(num_sims * horizon)]
    cumulative_left_rewards = [0.0 for ii in range(num_sims * horizon)]
    cumulative_right_rewards = [0.0 for ii in range(num_sims * horizon)]
    sim_nums = [0.0 for ii in range(num_sims * horizon)]
    times = [0.0 for ii in range(num_sims * horizon)]

    for sim in range(num_sims):
        sim += 1
        algo.initialize()

        for t in range(horizon):
            t += 1
            index = (sim - 1) * horizon + t - 1
            sim_nums[index] = sim
            times[index] = t

            [chosen_left_arm, chosen_right_arm] = algo.select_arm(t)
            chosen_left_arms[index] = chosen_left_arm
            chosen_right_arms[index] = chosen_right_arm

            left_reward = arms[chosen_left_arms[index]].draw()
            left_rewards[index] = left_reward

            right_reward = arms[chosen_right_arms[index]].draw()
            right_rewards[index] = right_reward

            if t == 1:
                cumulative_left_rewards[index] = left_reward
                cumulative_right_rewards[index] = right_reward
            else:
                cumulative_left_rewards[index] = cumulative_left_rewards[index - 1] + left_reward
                cumulative_right_rewards[index] = cumulative_right_rewards[index - 1] + right_reward

            if left_reward > right_reward:
                better_arm = "left"
            else:
                better_arm = "right"

            algo.update(chosen_left_arm, chosen_right_arm, better_arm)

    return [sim_nums, times, chosen_left_arms, left_rewards, cumulative_left_rewards,
            chosen_right_arms, right_rewards, cumulative_right_rewards]


random.seed(1)
means = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(means)
random.shuffle(means)
my_arms = map(lambda (mu): FixedArm(mu), means)
print("Best arm is " + str(ind_max(means)))

algo = RUCB(n_arms=n_arms, alpha=0.5)
algo.initialize()
results = test_rucb_algorithm(algo, my_arms, 5000, 250)

f = open("rucb_results.tsv", "w")

for i in range(len(results[0])):
    f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")

f.close()

