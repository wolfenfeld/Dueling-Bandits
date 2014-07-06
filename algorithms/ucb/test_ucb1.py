# Convenience functions
def ind_max(x):
    m = max(x)
    return x.index(m)

from arms.bernoulli import *

# Definitions of bandit algorithms
from algorithms.ucb.ucb1 import *
from algorithms.ucb.ucb2 import *
from algorithms.hedge.hedge import *

# # Testing framework
from testing_framework.tests import *


random.seed(1)
means = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(means)
random.shuffle(means)
arms = map(lambda (mu): BernoulliArm(mu), means)
print("Best arm is " + str(ind_max(means)))

algo = UCB1([], [])
algo.initialize(n_arms)
results = test_algorithm(algo, arms, 5000, 250)

f = open("ucb1_results.tsv", "w")

for i in range(len(results[0])):
    f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")

f.close()
