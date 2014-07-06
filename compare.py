import algorithms.rucb.rucb as rucb
import algorithms.Sparring.sparring as sparring
import algorithms.Doubler.doubler as doubler
import matplotlib.pyplot as plt
import random

random.seed(1)

# The means vector for the arms.
means = [0.1, 0.96, 0.8, 0.30, 0.7, 0.4]

# The horizon
horizon = 4096

# The number of iterations for this test.
iterations = 20

# The RUCB algorithm results.
rucb_results = rucb.run_several_iterations(iterations, means, horizon)

# The Sparring algorithm results.
sparring_results = sparring.run_several_iterations(iterations, means, horizon)

# The Doubler algorithm results.
doubler_results = doubler.run_several_iterations(iterations, means, horizon, improved=False)

# The improved algorithm results.
improved_doubler_results = doubler.run_several_iterations(iterations, means, horizon, improved=True)

#Ploting the outcome of all the algorithms (Cumulative regret)
T = range(horizon)
plt.plot(T, rucb_results, 'r--', label="RUCB")
plt.plot(T, sparring_results, 'b--', label="Sparring")
plt.plot(T, doubler_results, 'g--', label="Doubler")
plt.plot(T, improved_doubler_results, 'm--', label="Improved Doubler")
plt.legend(loc='upper left', shadow=True)
plt.show()