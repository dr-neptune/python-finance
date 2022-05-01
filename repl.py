# Random Numbers
import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import matplotlib
matplotlib.use('tkAgg')

npr.seed(100)
np.set_printoptions(precision=4)

npr.rand(10)
npr.rand(5, 5)
npr.rand(10) * (10 - 5) + 10

sample_size = 500
rn1 = npr.rand(sample_size, 3)            # uniformly distributed random numbers
rn2 = npr.randint(0, 10, sample_size)     # random integers for a given interval
rn3 = npr.sample(size=sample_size)
a = [0, 25, 50, 75, 100]
rn4 = npr.choice(a, size=sample_size)     # randomly sampled values from a finite list object

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1.hist(rn1, bins=25, stacked=True)
ax1.set_title('rand')
ax1.set_ylabel('frequency')
ax2.hist(rn2, bins=25)
ax2.set_title('randint')
ax3.hist(rn3, bins=25)
ax3.set_title('sample')
ax3.set_ylabel('frequency')
ax4.hist(rn4, bins=25)
ax4.set_title('choice')
plt.show()

# Simulation
S0 = 100
r = 0.05
sigma = 0.25
T = 2.0
I = 10000
ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * npr.standard_normal(I))

plt.figure()
plt.hist(ST1, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.show()

# we could also try lognormal to take into account the right leaning distribution
ST2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T,
                         sigma * math.sqrt(T), size = I)

plt.figure()
plt.hist(ST2, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.show()

import scipy.stats as scs

scs.describe(ST1)
scs.describe(ST2)

# Stochastic Processes
