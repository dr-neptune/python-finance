# Kelly Criterion
import math
import time
import numpy as np
import pandas as pd
import datetime as dt
import cufflinks as cf
import matplotlib.pyplot as plt
np.random.seed(1000)
import matplotlib
matplotlib.use('tkAgg')
plt.style.use('seaborn')


p = 0.55             # fix the probability of heads
f = p - (1 - p)      # calculate optimal fraction according to Kelly criterion
f

I = 50               # number of series to be simulated
n = 100              # number of trials per series

def run_simulation(f):
    # instantiate an ndarray to store simulation results
    c = np.zeros((n, I))
    c[0] = 100  # initialize starting capital with 100
    for i in range(I):             # series simulations
        for t in range(1, n):      # the series itself
            o = np.random.binomial(1, p)   # simulate the tossing of a coin
            if o > 0:
                c[t, i] = (1 + f) * c[t - 1, i]  # if win, add the win to the capital
            else:
                c[t, i] = (1 - f) * c[t - 1, i]  # otherwise, remove the amount from the capital
    return c

c_1 = run_simulation(f)
c_1.round(2)

# 50 simulated series with 100 trials each (red line = average)
plt.figure()
plt.plot(c_1, 'b', lw=0.5)
plt.plot(c_1.mean(axis=1), 'r', lw=2.5)
plt.show()

c_2 = run_simulation(0.05)
c_3 = run_simulation(0.25)
c_4 = run_simulation(0.5)

plt.figure()
plt.plot(c_1.mean(axis=1), 'r', label='$f^*=0.1$')
plt.plot(c_2.mean(axis=1), 'b', label='$f^*=0.05$')
plt.plot(c_3.mean(axis=1), 'y', label='$f^*=0.25$')
plt.plot(c_4.mean(axis=1), 'm', label='$f^*=0.5$')
plt.legend(loc=0)
plt.show()
