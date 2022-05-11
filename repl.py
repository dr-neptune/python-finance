# Benchmark Case
import math
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')
plt.style.use('seaborn')

def gen_paths(S0, r, sigma, T, M, I):
    """
    Generate Monte Carlo paths for geometric Brownian motion

    Args:
        S0: initial stock/index value
        r : constant short rate
        sigma: constant volatility
        T : final time horizon
        M : number of time steps / intervals
        I : number of paths to be simulated

    Returns:
        paths: simulated paths given the parameters
    """
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        # standardize / match 1st and 2nd moment
        rand = (rand - rand.mean()) / rand.std()
        # vectorized Euler discretization of geometric Brownian motion
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * math.sqrt(dt) * rand)
    return paths

S0 = 100.
r = 0.05
sigma = 0.2
T = 1.0
M = 50
I = 250000
np.random.seed(1000)

paths = gen_paths(S0, r, sigma, T, M, I)

S0 * math.exp(r * T)  # expected value and average simulated value

paths[-1].mean()

# ten simulated paths of geometric Brownian motion
plt.figure()
plt.plot(paths[:, :10])
plt.xlabel('time steps')
plt.ylabel('index level')
plt.show()

log_returns = np.log(paths[1:] / paths[:-1]).round(4)

def print_statistics(array):
    """
    prints selected statistics
    """
    sta = scs.describe(array)
    print('%14s %15s' % ('statistic', 'value'))
    print(30 * '-')
    print('%14s %15.5f' % ('size', sta[0]))
    print('%14s %15.5f' % ('min', sta[1][0]))
    print('%14s %15.5f' % ('max', sta[1][1]))
    print('%14s %15.5f' % ('mean', sta[2]))
    print('%14s %15.5f' % ('std', np.sqrt(sta[3])))
    print('%14s %15.5f' % ('skew', sta[4]))
    print('%14s %15.5f' % ('kurtosis', sta[5]))

print_statistics(log_returns.flatten())

log_returns.mean() * M + 0.5 * sigma ** 2  # annualized mean log return after correction for the Ito term

log_returns.std() * math.sqrt(M)  # annualized volatility, i.e., annualized std dev of log returns

# histogram of log returns of geometric Brownian motion and normal density function
plt.figure()
plt.hist(log_returns.flatten(), bins=70, density=True, label='frequency', color='b')
plt.xlabel('log return')
plt.ylabel('frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, loc=r/M, scale=sigma/np.sqrt(M)))
plt.show()


# QQ plots
sm.qqplot(log_returns.flatten()[::500], line='s')
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.show()

def normality_tests(arr):
    """
    Tests for normality distribution of a given data set
    Args:
        arr: object to generate statistics on
    Side Effect:
        prints a bunch of stats to the console
    """
    print(f'Skew:\t {scs.skew(arr):30.3f}')
    print(f'Skew test p_value:\t {scs.skewtest(arr)[1]:14.3f}')
    print(f'Kurt of data set:\t {scs.kurtosis(arr):14.3f}')
    print(f'Kurt test p-value:\t {scs.kurtosistest(arr)[1]:14.3f}')
    print(f'Norm test p-value:\t {scs.normaltest(arr)[1]:14.3f}')

normality_tests(log_returns.flatten())

# histogram of simualted end-of-period index levels for geometric Brownian motion
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(paths[-1], bins=30)
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax1.set_title('regular data')
ax2.hist(np.log(paths[-1]), bins=30)
ax2.set_xlabel('log index level')
ax2.set_title('log data')
plt.show()


# Real-World Data
import pandas as pd
raw = pd.read_csv('data/tr_eikon_eod_data.csv',
                  index_col=0, parse_dates=True).dropna()
symbols = ['SPY', 'GLD', 'AAPL.O', 'MSFT.O']

data = raw[symbols]

data.info()
data.head()

# normalized prices of financial instruments over time
(data / data.iloc[0] * 100).plot()
plt.show()

# histograms of log returns for financial instruments
log_returns = np.log(data / data.shift(1))
log_returns.head()

log_returns.hist(bins=50)
plt.show()

for sym in symbols:
    print(f'\nResults for symbol {sym}')
    print(30 * '-')
    log_data = np.array(log_returns[sym].dropna())
    print_statistics(log_data)
    print('\nNormality Tests\n')
    normality_tests(log_data)

# Portfolio Optimization
noa = len(symbols)  # number of assets

rets = np.log(data / data.shift(1))
rets.hist(bins=40)
plt.show()

rets.mean() * 252  # annualized mean returns
rets.cov() * 252   # annualized covariance matrix

# The Basic Theory
weights = np.random.random(noa)
weights /= np.sum(weights)

def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252

def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

# monte carlo simulation of portfolio weights
prets, pvols = [], []
for p in range(2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(port_ret(weights))
    pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)

# expected return and volatility for random portfolio weights
plt.figure()
plt.scatter(pvols, prets, c=prets / pvols,
            marker='o', cmap='coolwarm')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

# optimal portfolios
import scipy.optimize as sco

def min_func_sharpe(weights):
    # function to be minimized
    return -port_ret(weights) / port_vol(weights)

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # equality constraint
bnds = tuple((0, 1) for x in range(noa))  # bounds for the parameters
eweights = np.array(noa * [1. / noa,])  # equal weights vector

opts = sco.minimize(min_func_sharpe,
                    eweights,
                    method='SLSQP',
                    bounds=bnds,
                    constraints=cons)

# maximize Sharpe ratio
optimals = opts['x'].round(3)  # optimal portfolio weights
port_ret(optimals)  # resulting portfolio return
port_vol(optimals)  # resulting portfolio volatility
port_ret(optimals) / port_vol(optimals)  # maximum Sharpe ratio

# minimize variance
optv = sco.minimize(port_vol,
                    eweights,
                    method='SLSQP',
                    bounds=bnds,
                    constraints=cons)

optimals = optv['x'].round(3)  # optimal portfolio weights
port_ret(optimals)  # resulting portfolio return
port_vol(optimals)  # resulting portfolio volatility
port_ret(optimals) / port_vol(optimals)  # maximum Sharpe ratio

# Efficient Frontier

trets = np.linspace(0.05, 0.2, 50)

# the two binding constraints for the efficient frontier
cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

bnds = tuple((0, 1) for x in weights)

tvols = []
for tret in trets:
    # minimization of portfolio volatility for different target returns
    res = sco.minimize(port_vol,
                       eweights,
                       method='SLSQP',
                       bounds=bnds,
                       constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

# minimum risk portfolios for given return levels (efficient frontier)
plt.figure()
plt.scatter(pvols, prets, c=prets/pvols,
            marker='.', alpha=0.8, cmap='coolwarm')
plt.plot(tvols, trets, 'b', lw=2.0)
plt.plot(port_vol(opts['x']),
         port_ret(opts['x']),
         'y*',
         markersize=15.0)
plt.plot(port_vol(optv['x']),
         port_ret(optv['x']),
         'r*',
         markersize=15.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe Ratio')
plt.show()
