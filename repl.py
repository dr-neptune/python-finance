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

# Capital Market Line
import scipy.interpolate as sci

ind = np.argmin(tvols)  # index position of minimum volatility portfolio
evols = tvols[ind:]     # relevant portfolio volatility
erets = trets[ind:]     # relevant portfolio returns

tck = sci.splrep(evols, erets)

def f(x):
    """Efficient frontier function (spline approximation)"""
    return sci.splev(x, tck, der=0)

def df(x):
    """First derivative of efficient frontier function"""
    return sci.splev(x, tck, der=1)

def equations(p, rf=0.01):
    # equations describing the capital market line
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

opt = sco.fsolve(equations, [0.01, 0.5, 0.15])

np.round(equations(opt), 6)

# Capital market line and tangent portfolio (star) for risk-free rate of 1%
plt.figure()
plt.scatter(pvols, prets, c=(prets - 0.01)/pvols,
            marker='.', cmap='coolwarm')
plt.plot(evols, erets, 'b', lw=2.0)
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5)
plt.plot(opt[2], f(opt[2]), 'y*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe Ratio')
plt.show()

# get portfolio weights of the optimal (tangent) portfolio
cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - f(opt[2])},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)

res['x'].round(3)

port_ret(res['x'])
port_vol(res['x'])
port_ret(res['x']) / port_vol(res['x'])

# Bayesian Regression
x = np.linspace(0, 10, 500)
y = 4 + 2 * x + np.random.standard_normal(len(x)) * 2
reg = np.polyfit(x, y, 1)

# sample data points and a regression line
plt.figure()
plt.scatter(x, y, c=y, marker='v', cmap='coolwarm')
plt.plot(x, reg[1] + reg[0] * x, lw=2.0)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

import pymc3 as pm

with pm.Model() as model:
    # model
    # define priors
    alpha = pm.Normal('alpha', mu=0, sd=20)
    beta = pm.Normal('beta', mu=0, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    # define model specification (linear regression)
    y_est = alpha + beta * x
    # define likelihood
    likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)
    # inference
    # find starting value by optimization
    start = pm.find_MAP()
    # instantiate the MCMC algorithm
    step = pm.NUTS()
    # draw posterior samples using NUTS
    trace = pm.sample(100, tune=1000, start=start, progressbar=True)

# show summary statistics from samplings
pm.summary(trace)
# estimates from the first sample
trace[0]

# trace plot
pm.traceplot(trace, lines={'alpha': 4, 'beta': 2, 'sigma': 2})
plt.show()

# regression lines based on different estimates
plt.figure()
plt.scatter(x, y, c=y, marker='v', cmap='coolwarm')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
for i in range(len(trace)):
    plt.plot(x, trace['alpha'][i] + trace['beta'][i] * x)
plt.show()

# Two Financial Instruments
data = raw[['GDX', 'GLD']].dropna()
data = data / data.iloc[0]
# normalized prices for GLD and GDX over time
data.plot()
plt.show()

# scatter plot of GLD prices against GDX prices
mpl_dates = matplotlib.dates.date2num(data.index.to_pydatetime())
plt.figure()
plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o', cmap='coolwarm')
plt.xlabel('GDX')
plt.ylabel('GLD')
plt.colorbar(ticks=matplotlib.dates.DayLocator(interval=250),
             format=matplotlib.dates.DateFormatter('%d %b %y'))
plt.show()

# implement a Bayesian regression on the basis of these two time series
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=20)
    beta = pm.Normal('beta', mu=0, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    y_est = alpha + beta * data['GDX'].values
    likelihood = pm.Normal('GLD', mu=y_est, sd=sigma, observed=data['GLD'].values)
    start = pm.find_MAP()
    step = pm.NUTS()
    trace = pm.sample(250, tune=2000, start=start, progressbar=True)


pm.summary(trace)
fig = pm.traceplot(trace)
plt.show()

# multiple Bayesian regression lines through GDX and GLD data
plt.figure()
plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o', cmap='coolwarm')
plt.xlabel('GDX')
plt.ylabel('GLD')
for i in range(len(trace)):
    plt.plot(data['GDX'], trace['alpha'][i] + trace['beta'][i] * data['GDX'])
plt.colorbar(ticks=matplotlib.dates.DayLocator(interval=250),
             format=matplotlib.dates.DateFormatter('%d %b %y'))
plt.show()

# Updating Estimates over Time
from pymc3.distributions.timeseries import GaussianRandomWalk

subsample_alpha, subsample_beta = 50, 50

model_randomwalk = pm.Model()
with model_randomwalk:
    # define priors for the random walk parameters
    sigma_alpha = pm.Exponential('sig_alpha', 1. / .02, testval=.1)
    sigma_beta = pm.Exponential('sig_beta', 1. / .02, testval=.1)
    # models for the random walks
    alpha = GaussianRandomWalk('alpha', sigma_alpha ** -2,
                               shape=int(len(data) / subsample_alpha))
    beta = GaussianRandomWalk('beta', sigma_beta ** -2,
                               shape=int(len(data) / subsample_beta))
    # brings the parameter vectors to interval length
    alpha_r = np.repeat(alpha, subsample_alpha)
    beta_r = np.repeat(beta, subsample_beta)
    # defines the regression model
    regression = alpha_r + beta_r * data['GDX'][:1950].values[:2100]
    # the prior for the standard deviation
    sd = pm.Uniform('sd', 0, 20)
    # defines the likelihood with mu from regression results
    likelihood = pm.Normal('GLD', mu=regression, sd=sd, observed=data['GLD'][:1950].values[:2100])


with model_randomwalk:
    start = pm.find_MAP(vars=[alpha, beta])
    step = pm.NUTS(scaling=start)
    trace_rw = pm.sample(250, tune=1000, start=start, progressbar=True)

pm.summary(trace_rw).head()  # the summary statistics per interval

sh = np.shape(trace_rw['alpha'])
sh  # shape of the object with parameter estimates

# creates a list of dates to match the number of intervals
part_dates = np.linspace(min(mpl_dates), max(mpl_dates), sh[1])

from datetime import datetime

index = [datetime.fromordinal(int(date)) for date in part_dates]

# collects the relevant parameter time series in two DataFrame objects
alpha = {'alpha_%i' % i: v for i, v in enumerate(trace_rw['alpha']) if i < 20}

beta = {'beta_%i' % i: v for i, v in enumerate(trace_rw['beta']) if i < 20}

df_alpha = pd.DataFrame(alpha, index=index)
df_beta = pd.DataFrame(beta, index=index)

ax = df_alpha.plot(color='b', style='-.', legend=False, lw=0.7)
df_beta.plot(color='r', style='-.', legend=False, lw=0.5, ax=ax)
plt.ylabel('alpha/beta')
plt.show()

# scatter plot with time-dependent regression lines (updated estimates)
plt.figure()
plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o', cmap='coolwarm')
plt.colorbar(ticks=matplotlib.dates.DayLocator(interval=250),
             format=matplotlib.dates.DateFormatter('%d %b %y'))
plt.xlabel('GDX')
plt.ylabel('GLD')
x = np.linspace(min(data['GDX']), max(data['GDX']))
for i in range(sh[1]):
    alpha_rw = np.mean(trace_rw['alpha'].T[i])
    beta_rw = np.mean(trace_rw['beta'].T[i])
    plt.plot(x, alpha_rw + beta_rw * x, '--', lw=0.7, color=plt.cm.coolwarm(i / sh[1]))
plt.show()


# Machine Learning
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=250, centers=4, random_state=500, cluster_std=1.25)

# sample data for the application of clustering algorithms
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

# k-means clustering
from sklearn.cluster import KMeans

model = KMeans(n_clusters=4, random_state=0)
model.fit(X)

y_kmeans = model.predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='coolwarm')
plt.show()

# Gaussian mixture
from sklearn.mixture import GaussianMixture

model = GaussianMixture(n_components=4, random_state=0)
model.fit(X)

y_gm = model.predict(X)
(y_gm == y_kmeans).all()

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_gm, cmap='coolwarm')
plt.show()

# Supervised Learning
from sklearn.datasets import make_classification

n_samples = 100

X, y = make_classification(n_samples=n_samples,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_repeated=0,
                           random_state=250)

plt.figure()
plt.hist(X)
plt.show()
plt.figure()
plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap='coolwarm')
plt.show()

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

model = GaussianNB()
model.fit(X, y)

model.predict_proba(X).round(4)[:5]

pred = model.predict(X)

accuracy_score(y, pred)

Xc = X[y == pred]
Xf = X[y != pred]
plt.figure()
# correct predictions
plt.scatter(x=Xc[:, 0], y=Xc[:, 1], c=y[y == pred], marker='o', cmap='coolwarm')
# false predictions
plt.scatter(x=Xf[:, 0], y=Xf[:, 1], c=y[y != pred], marker='x', cmap='coolwarm')
plt.show()

# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1, solver='lbfgs')
model.fit(X, y)

model.predict_proba(X).round(4)[:5]

pred = model.predict(X)

accuracy_score(y, pred)

Xc = X[y == pred]
Xf = X[y != pred]
plt.figure()
# correct predictions
plt.scatter(x=Xc[:, 0], y=Xc[:, 1], c=y[y == pred], marker='o', cmap='coolwarm')
# false predictions
plt.scatter(x=Xf[:, 0], y=Xf[:, 1], c=y[y != pred], marker='x', cmap='coolwarm')
plt.show()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=1)
model.fit(X, y)

pred = model.predict(X)
accuracy_score(y, pred)


print('{:>8s} | {:8s}'.format('depth', 'accuracy'))
print(20 * '-')
for depth in range(1, 7):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    print('{:8d} | {:8.2f}'.format(depth, acc))

# deep neural networks

# DNN with scikit-learn
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=2 * [75], random_state=10)
model.fit(X, y)
pred = model.predict(X)
accuracy_score(y, pred)

# DNNs with TensorFlow
import tensorflow as tf

fc = [tf.contrib.layers.real_valued_column('features')]
model = tf.contrib.learn.DNNClassifier(hidden_units=5 * [250],
                                       n_classes=2,
                                       feature_columns=fc)

def input_fn():
    fc = {'features': tf.constant(X)}
    la = tf.constant(y)
    return fc, la

model.fit(input_fn=input_fn, steps=100)
model.evaluate(input_fn, steps=1)


model.fit(input_fn=input_fn, steps=750)
model.evaluate(input_fn, steps=1)
