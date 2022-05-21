import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')

def get_year_deltas(date_list, day_count=365.):
    '''
    Return vector of floats with day deltas in year fractions. Initial value normalized to 0
    '''
    start = date_list[0]
    delta_list = [(date - start).days / day_count for date in date_list]
    return np.array(delta_list)

dates = [dt.datetime(2020, 1, 1), dt.datetime(2020, 7, 1), dt.datetime(2021, 1, 1)]
get_year_deltas(dates)

# constant short rate
class ConstantShortRate:
    '''
    class for constant short rate discounting
    '''
    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            raise ValueError('short rate negative')

    def get_discount_factors(self, date_list, dtobjects=True):
        if dtobjects is True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        dflist = np.exp(self.short_rate * np.sort(-dlist))
        return np.array((date_list, dflist)).T

csr = ConstantShortRate('csr', 0.05)

csr.get_discount_factors(dates)
deltas = get_year_deltas(dates)
csr.get_discount_factors(deltas, dtobjects=False)

# market environments
class MarketEnvironment:
    '''class to model a market environment relevant for valuation'''
    def __init__(self, name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.constants = {}
        self.lists = {}
        self.curves = {}

    def add_constant(self, key, constant):
        self.constants[key] = constant

    def get_constant(self, key):
        return self.constants[key]

    def add_list(self, key, list_object):
        self.lists[key] = list_object

    def get_list(self, key):
        return self.lists[key]

    def add_curve(self, key, curve):
        self.curves[key] = curve

    def get_curve(self, key):
        return self.curves[key]

    def add_environment(self, env):
        self.constants.update(env.constants)
        self.lists.update(env.lists)
        self.curves.update(env.curves)

me = MarketEnvironment('me_gbm', dt.datetime(2020, 1, 1))
me.add_constant('initial_value', 36.)
me.add_constant('volatility', 0.2)
me.add_constant('final_date', dt.datetime(2020, 12, 31))
me.add_constant('currency', 'EUR')
me.add_constant('frequency', 'M')
me.add_constant('paths', 10000)
me.add_curve('discount_curve', csr)
me.get_constant('volatility')
me.get_curve('discount_curve').short_rate

# Chapter 18 : Simulation of Financial Models
def sn_random_numbers(shape, antithetic=True, moment_matching=True, fixed_seed=False):
    '''Returns an ndarray of shape shape with (pseudo)random numbers that are std normally distributed'''
    if fixed_seed:
        np.random.seed(8888)
    if antithetic:
        ran = np.random.standard_normal((shape[0], shape[1], shape[2] // 2))
        ran = np.concatenate((ran, -ran), axis=2)
    else:
        ran = np.random.standard_normal(shape)
    if moment_matching:
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran)
    if shape[0] == 1:
        return ran[0]
    else:
        return ran

snrn = sn_random_numbers((2, 2, 2), False, False, True)

# generic simulation class
class SimulationClass:
    '''provides base methods for simulation classes'''
    def __init__(self, name, mar_env, corr):
        self.name = name
        self.pricing_date = mar_env.pricing_date
        self.initial_value = mar_env.get_constant('initial_value')
        self.volatility = mar_env.get_constant('volatility')
        self.final_date = mar_env.get_constant('final_date')
        self.currency = mar_env.get_constant('currency')
        self.frequency = mar_env.get_constant('frequency')
        self.paths = mar_env.get_constant('paths')
        self.discount_curve = mar_env.get_curve('discount_curve')

        try:
            self.time_grid = mar_env.get_list('time_grid')
        except:
            self.time_grid = None

        try:
            self.special_dates = mar_env.get_list('special_dates')
        except:
            self.special_dates = []

        self.instrument_values = None
        self.correlated = corr

        if corr:
            self.cholesky_matrix = mar_env.get_list('cholesky_matrix')
            self.rn_set = mar_env.get_list('rn_set')[self.name]
            self.random_numbers = mar_env.get_list('random_numbers')

    def generate_time_grid(self):
        start = self.pricing_date
        end = self.final_date
        time_grid = list(pd.date_range(start=start, end=end, freq=self.frequency).to_pydatetime())

        if start not in time_grid:
            time_grid.insert(0, start)
        if end not in time_grid:
            time_grid.append(end)
        if len(self.special_dates) > 0:
            time_grid.extend(self.special_dates)
            time_grid = list(set(time_grid))
            time_grid.sort()
        self.time_grid = np.array(time_grid)

    def get_instrument_values(self, fixed_seed=True):
        if self.instrument_values is None or not fixed_seed:
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.)
        return self.instrument_values

# geometric brownian motion
class GeometricBrownianMotion(SimulationClass):
    '''class to generate simulated paths based on the Black-Scholes-Metric geometric Brownian motion model'''
    def __init__(self, name, mar_env, corr=False):
        super(GeometricBrownianMotion, self).__init__(name, mar_env, corr)

    def update(self, initial_value=None, volatility=None, final_date=None):
        if initial_value:
            self.initial_value = initial_value
        if volatility:
            self.volatility = volatility
        if final_date:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()

        # num of dates for time grid
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        paths[0] = self.initial_value

        if not self.correlated:
            rand = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            rand = self.random_numbers

        short_rate = self.discount_curve.short_rate

        for t in range(1, len(self.time_grid)):
            if not self.correlated:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            # difference between 2 dates as year fraction
            paths[t] = paths[t - 1] * np.exp((short_rate - 0.5 * self.volatility ** 2) * dt +
                                             self.volatility * np.sqrt(dt) * ran)
        self.instrument_values = paths

# a use case
me_gbm = MarketEnvironment('me_gbm', dt.datetime(2020, 1, 1))
me_gbm.add_constant('initial_value', 36.)
me_gbm.add_constant('volatility', 0.2)
me_gbm.add_constant('final_date', dt.datetime(2020, 12, 31))
me_gbm.add_constant('currency', 'EUR')
me_gbm.add_constant('frequency', 'M')
me_gbm.add_constant('paths', 10000)
me_gbm.add_curve('discount_curve', ConstantShortRate('csr', 0.06))

gbm = GeometricBrownianMotion('gbm', me_gbm)
gbm.generate_time_grid()
gbm.time_grid
paths_1 = gbm.get_instrument_values()
paths_1

gbm.update(volatility=0.5)
paths_2 = gbm.get_instrument_values()
paths_2

plt.figure()
p1 = plt.plot(gbm.time_grid, paths_1[:, :10], 'b')
p2 = plt.plot(gbm.time_grid, paths_2[:, :10], 'r-.')
l1 = plt.legend([p1[0], p2[0]], ['low volatility', 'high volatility'], loc=2)
plt.gca().add_artist(l1)
plt.xticks(rotation=30)
plt.show()

# jump diffusion
class JumpDiffusion(SimulationClass):
    '''class to generate simulated paths based on the Merton jump diffusion model'''
    def __init__(self, name, mar_env, corr=False):
        super(JumpDiffusion, self).__init__(name, mar_env, corr)
        self.lamb = mar_env.get_constant('lambda')
        self.mu = mar_env.get_constant('mu')
        self.delt = mar_env.get_constant('delta')

    def update(self, initial_value = None, volatility = None, lamb = None, mu = None, delta = None, final_date = None):
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if lamb is not None:
            self.lamb = lamb
        if mu is not None:
            self.mu = mu
        if delta is not None:
            self.delta = delta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        paths[0] = self.initial_value

        if self.correlated is False:
            sn1 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            sn1 = self.random_numbers

        sn2 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)

        rj = self.lamb * (np.exp(self.mu + 0.5 * self.delt ** 2) - 1)

        short_rate = self.discount_curve.short_rate

        for t in range(1, len(self.time_grid)):
            if self.correlated is False:
                ran = sn1[t]
            else:
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            poi = np.random.poisson(self.lamb * dt, I)
            paths[t] = paths[t - 1] * (np.exp((short_rate - rj - 0.5 * self.volatility ** 2) * dt +
                                              self.volatility * np.sqrt(dt) * ran) +
                                              (np.exp(self.mu + self.delt * sn2[t]) - 1) * poi)
        self.instrument_values = paths

# a use case
me_jd = MarketEnvironment('me_jd', dt.datetime(2020, 1, 1))
me_jd.add_constant('lambda', 0.3)
me_jd.add_constant('mu', -0.75)
me_jd.add_constant('delta', 0.1)

# add a complete environment to the existing one
me_jd.add_environment(me_gbm)

jd = JumpDiffusion('jd', me_jd)

paths_3 = jd.get_instrument_values()

jd.update(lamb=0.9)
paths_4 = jd.get_instrument_values()

plt.figure()
p1 = plt.plot(gbm.time_grid, paths_3[:, :10], 'b')
p2 = plt.plot(gbm.time_grid, paths_4[:, :10], 'r-.')
l1 = plt.legend([p1[0], p2[0]], ['low volatility', 'high volatility'], loc=2)
plt.gca().add_artist(l1)
plt.xticks(rotation=30)
plt.show()

# square root diffusion
class SquareRootDiffusion(SimulationClass):
    '''class to generate simulated paths based on the Cox-Ingersoll-Ross Square Root Diffusion model'''
    def __init__(self, name, mar_env, corr=False):
        super(SquareRootDiffusion, self).__init__(name, mar_env, corr)
        self.kappa = mar_env.get_constant('kappa')
        self.theta = mar_env.get_constant('theta')

    def update(self, initial_value = None, volatility = None, kappa = None, theta = None, final_date = None):
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if kappa is not None:
            self.kappa = kappa
        if theta is not None:
            self.theta = theta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths

        paths = np.zeros((M, I))
        paths[0] = self.initial_value

        paths_ = np.zeros_like(paths)
        paths_[0] = self.initial_value

        if self.correlated is False:
            rand = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            rand = self.random_numbers

        for t in range(1, len(self.time_grid)):
            if self.correlated is False:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]

            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count

            paths_[t] = (paths_[t - 1] + \
                         self.kappa * \
                         (self.theta - \
                         np.maximum(0, paths_[t - 1, :])) * \
                         dt + \
                         np.sqrt(np.maximum(0, paths_[t - 1, :])) * \
                         self.volatility * \
                         np.sqrt(dt) * \
                         ran)
            paths[t] = np.maximum(0, paths_[t])
        self.instrument_values = paths

# a use case
me_srd = MarketEnvironment('me_srd', dt.datetime(2020, 1, 1))
me_srd.add_constant('initial_value', 0.25)
me_srd.add_constant('volatility', 0.05)
me_srd.add_constant('final_date', dt.datetime(2020, 12, 31))
me_srd.add_constant('currency', 'EUR')
me_srd.add_constant('frequency', 'W')
me_srd.add_constant('paths', 10000)
me_srd.add_constant('kappa', 4.0)
me_srd.add_constant('theta', 0.2)

me_srd.add_curve('discount_curve', ConstantShortRate('r', 0.0))
srd = SquareRootDiffusion('srd', me_srd)
srd_paths = srd.get_instrument_values()

plt.figure()
plt.plot(srd.time_grid, srd.get_instrument_values()[:, :10])
plt.axhline(me_srd.get_constant('theta'), color='r', ls='--', lw=2.0)
plt.xticks(rotation=30)
plt.show()
