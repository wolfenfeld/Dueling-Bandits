import algorithms.rucb.rucb as rucb
import algorithms.Sparring.sparring as sparring
import algorithms.Doubler.doubler as doubler
import algorithms.Rcs.rcs as rcs
import algorithms.Doubler.balanced_doubler as b_doubler
import matplotlib.pyplot as plt
import random

random.seed(7)

# The means vector for the arms.
#means = [0.1, 0.96, 0.8, 0.30, 0.7, 0.4]
means = [random.random() for i in xrange(4)]

print "Means are : {0}".format(means)
# The horizon
horizon = 4096*8

# The number of iterations for this test.
iterations = 10

# The RUCB algorithm results.
#rucb_results = rucb.run_several_iterations(iterations, means, horizon)
#
# # The RCS algorithm results.
# rcs_results = rcs.run_several_iterations(iterations, means, horizon)
#
# The Sparring algorithm results.
#sparring_results = sparring.run_several_iterations(iterations, means, horizon)
#
# # The Doubler algorithm results.
# doubler_results = doubler.run_several_iterations(iterations, means, horizon, improved=False)
#
# The improved algorithm results.
# improved_doubler_results = doubler.run_several_iterations(iterations, means, horizon, improved=True)

# The balanced doubler algorithm results
b_doubler_results = b_doubler.run_several_iterations(iterations, means, horizon)

#Ploting the outcome of all the algorithms (Cumulative regret)
T = range(horizon)
# plt.plot(T, rucb_results, 'r--', label="RUCB")
# plt.plot(T, sparring_results, 'k--', label="Sparring")
# plt.plot(T, doubler_results, 'g--', label="Doubler")
# plt.plot(T, rcs_results, 'm--', label="RCS")
# plt.plot(T, improved_doubler_results, 'k--', label="Improved Doubler")
plt.plot(T, b_doubler_results, 'b--', label="Balanced Doubler")

plt.legend(loc='upper left', shadow=True)
plt.show()