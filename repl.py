import numpy as np
from numpy.ma import correlate
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

# ch 19: derivatives valuation
class ValuationClass:
    '''basic class for single-factor valuation'''
    def __init__(self, name, underlying, mar_env, payoff_func=''):
        self.name = name
        self.pricing_date = mar_env.pricing_date
        try:
            self.strike = mar_env.get_constant('strike')
        except:
            pass
        self.maturity = mar_env.get_constant('maturity')
        self.currency = mar_env.get_constant('currency')
        self.frequency = underlying.frequency
        self.paths = underlying.paths
        self.discount_curve = underlying.discount_curve
        self.payoff_func = payoff_func
        self.underlying = underlying
        self.underlying.special_dates.extend([self.pricing_date, self.maturity])

    def update(self, initial_value=None, volatility=None, strike=None, maturity=None):
        if initial_value is not None:
            self.underlying.update(initial_value=initial_value)
        if volatility is not None:
            self.underlying.update(volatility=volatility)
        if strike is not None:
            self.strike = strike
        if maturity is not None:
            self.maturity = maturity
            if maturity not in self.underlying.time_grid:
                self.underlying.special_dates.append(maturity)
                self.underlying.instrument_values = None

    def delta(self, interval=None, accuracy=4):
        if interval is None:
            interval = self.underlying.initial_value / 50.

        # forward difference approximation
        # calculate left value for numerical delta
        value_left = self.present_value(fixed_seed=True)
        # numerical underlying value for right value
        initial_del = self.underlying.initial_value + interval
        self.underlying.update(initial_value=initial_del)
        # calculate right value for numerical delta
        value_right = self.present_value(fixed_seed=True)
        # reset the initial value of the simulation object
        self.underlying.update(initial_value=initial_del - interval)
        delta = (value_right - value_left) / interval

        if delta < -1.0:
            return -1.0
        elif delta > 1.0:
            return 1.0
        else:
            return round(delta, accuracy)

    def vega(self, interval=0.01, accuracy=4):
        if interval < self.underlying.volatility / 50.:
            interval = self.underlying.volatility / 50.

        # forward difference approximation
        value_left = self.present_value(fixed_seed=True)
        vola_del = self.underlying.volatility + interval
        self.underlying.update(volatility=vola_del)
        value_right = self.present_value(fixed_seed=True)
        self.underlying.update(volatility=vola_del - interval)
        vega = (value_right - value_left) / interval
        return round(vega, accuracy)

# The Valuation Class
class ValuationMCSEuropean(ValuationClass):
    '''class to value European options with arbitrary payoff by single-factor Monte Carlo simulation'''
    def generate_payoff(self, fixed_seed=False):
        try:
            strike = self.strike
        except AttributeError:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        time_grid = self.underlying.time_grid
        try:
            time_index = np.where(time_grid == self.maturity)[0]
            time_index = int(time_index)
        except:
            print('Maturity date not in time grid of underlying')

        maturity_value = paths[time_index]
        # average value over whole path
        mean_value = np.mean(paths[:time_index], axis=1)
        # maximum value over whole path
        max_value = np.amax(paths[:time_index], axis=1)[-1]
        # minimum value over whole path
        min_value = np.amin(paths[:time_index], axis=1)[-1]

        try:
            payoff = eval(self.payoff_func)
            return payoff
        except:
            print('Error evaluation payoff function')

    def present_value(self, accuracy=6, fixed_seed=False, full=False):
        cash_flow = self.generate_payoff(fixed_seed=fixed_seed)
        discount_factor = self.discount_curve.get_discount_factors((self.pricing_date, self.maturity))[0, 1]
        result = discount_factor * np.sum(cash_flow) / len(cash_flow)
        if full:
            return round(result, accuracy), discount_factor * cash_flow
        else:
            return round(result, accuracy)
## Use Case
me_gbm = MarketEnvironment('me_gbm', dt.datetime(2020, 1, 1))
me_gbm.add_constant('initial_value', 36.)
me_gbm.add_constant('volatility', 0.2)
me_gbm.add_constant('final_date', dt.datetime(2020, 12, 31))
me_gbm.add_constant('currency', 'EUR')
me_gbm.add_constant('frequency', 'M')
me_gbm.add_constant('paths', 10000)
me_gbm.add_curve('discount_curve', ConstantShortRate('csr', 0.06))

gbm = GeometricBrownianMotion('gbm', me_gbm)

# define the market environment for the option itself
me_call = MarketEnvironment('me_call', me_gbm.pricing_date)
me_call.add_constant('strike', 40.)
me_call.add_constant('maturity', dt.datetime(2020, 12, 31))
me_call.add_constant('currency', 'EUR')

payoff_func = 'np.maximum(maturity_value - strike, 0)'

eur_call = ValuationMCSEuropean('eur_call', underlying=gbm, mar_env=me_call, payoff_func=payoff_func)

eur_call.present_value()  # present value of European call option
eur_call.delta()          # numerical estimate of the delta of the option. Delta is positive for calls
eur_call.vega()           # numerical estimate of the vega for the option. Vega is positive for both calls and puts

def plot_option_stats(s_list, p_list, d_list, v_list):
    '''plot option prices, deltas, and vegas for a set of different initial values of the underlying'''
    plt.figure(figsize=(10, 7))
    sub1 = plt.subplot(311)
    plt.plot(s_list, p_list, 'ro', label='present value')
    plt.plot(s_list, p_list, 'b')
    plt.legend(loc=0)
    plt.setp(sub1.get_xticklabels(), visible=False)
    sub2 = plt.subplot(312)
    plt.plot(s_list, d_list, 'go', label='Delta')
    plt.plot(s_list, d_list, 'b')
    plt.legend(loc=0)
    plt.ylim(min(d_list) - 0.1, max(d_list) + 0.1)
    plt.setp(sub2.get_xticklabels(), visible=False)
    sub3 = plt.subplot(313)
    plt.plot(s_list, v_list, 'yo', label='Vega')
    plt.plot(s_list, v_list, 'b')
    plt.xlabel('initial value of underlying')
    plt.legend(loc=0)
    plt.show()

s_list = np.arange(34., 46.1, 2.)
p_list, d_list, v_list = [], [], []

for s in s_list:
    eur_call.update(initial_value=s)
    p_list.append(eur_call.present_value(fixed_seed=True))
    d_list.append(eur_call.delta())
    v_list.append(eur_call.vega())

# present value, delta, and vega estimates for European call option
plot_option_stats(s_list, p_list, d_list, v_list)

# consider a payoff that is a mixture of a regular and Asian payoff
payoff_func = 'np.maximum(0.33 * (maturity_value + max_value) - 40, 0)'

eur_as_call = ValuationMCSEuropean('eur_as_call', underlying=gbm, mar_env=me_call, payoff_func=payoff_func)

s_list = np.arange(34., 46.1, 2.)
p_list, d_list, v_list = [], [], []

for s in s_list:
    eur_as_call.update(s)
    p_list.append(eur_as_call.present_value(fixed_seed=True))
    d_list.append(eur_as_call.delta())
    v_list.append(eur_as_call.vega())

plot_option_stats(s_list, p_list, d_list, v_list)

# American Exercise
class ValuationMCSAmerican(ValuationClass):
    '''class to value American options with arbitrary payoff by single-factor Monte Carlo simulation'''
    def generate_payoff(self, fixed_seed=False):
        try:
            strike = self.strike
        except AttributeError:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        time_grid = self.underlying.time_grid
        time_index_start = int(np.where(time_grid == self.pricing_date)[0])
        time_index_end = int(np.where(time_grid == self.maturity)[0])
        instrument_values = paths[time_index_start:time_index_end + 1]
        payoff = eval(self.payoff_func)
        return instrument_values, payoff, time_index_start, time_index_end

    def present_value(self, accuracy=6, fixed_seed=False, bf=5, full=False):
        instrument_values, inner_values, time_index_start, time_index_end = self.generate_payoff(fixed_seed=fixed_seed)
        time_list = self.underlying.time_grid[time_index_start:time_index_end + 1]
        discount_factors = self.discount_curve.get_discount_factors(time_list, dtobjects=True)
        V = inner_values[-1]
        for t in range(len(time_list) - 2, 0, -1):
            # derive relevant discount factor for given time interval
            df = discount_factors[t, 1] / discount_factors[t + 1, 1]
            # regression step
            rg = np.polyfit(instrument_values[t], V * df, bf)
            # calculation of continuation values per path
            C = np.polyval(rg, instrument_values[t])
            # optimal decision step
            # if condition (inner value > regression cont value)
            # then take inner value; ow take actual value
            V = np.where(inner_values[t] > C, inner_values[t], V * df)
        df = discount_factors[0, 1] / discount_factors[1, 1]
        result = df * np.sum(V) / len(V)
        if full:
            return round(result, accuracy), df * V
        else:
            return round(result, accuracy)

## a use case
me_gbm = MarketEnvironment('me_gbm', dt.datetime(2020, 1, 1))
me_gbm.add_constant('initial_value', 36.)
me_gbm.add_constant('volatility', 0.2)
me_gbm.add_constant('final_date', dt.datetime(2020, 12, 31))
me_gbm.add_constant('currency', 'EUR')
me_gbm.add_constant('frequency', 'W')
me_gbm.add_constant('paths', 50000)
me_gbm.add_curve('discount_curve', ConstantShortRate('csr', 0.06))

gbm = GeometricBrownianMotion('gbm', me_gbm)
payoff_func = 'np.maximum(strike - instrument_values, 0)'

me_am_put = MarketEnvironment('me_am_put', dt.datetime(2020, 1, 1))
me_am_put.add_constant('maturity', dt.datetime(2020, 12, 31))
me_am_put.add_constant('strike', 40.)
me_am_put.add_constant('currency', 'EUR')

am_put = ValuationMCSAmerican('am_put', underlying=gbm, mar_env=me_am_put, payoff_func=payoff_func)

am_put.present_value(fixed_seed=True, bf=5)

# chapter 20: portfolio valuation

# derivatives positions
class DerivativesPosition:
    '''
    class to model a derivatives position

    name: name of object
    quantity: number of assets/derivatives making up the position
    underlying: name of asset/risk factor for the derivative
    mar_env: instance of market environment
    otype: valuation class to use
    payoff_func: payoff string for the derivative
    '''
    def __init__(self, name, quantity, underlying, mar_env, otype, payoff_func):
        self.name = name
        self.quantity = quantity
        self.underlying = underlying
        self.mar_env = mar_env
        self.otype = otype
        self.payoff_func = payoff_func

    def get_info(self):
        print('NAME')
        print(self.name, '\n')
        print('QUANTITY')
        print(self.quantity, '\n')
        print('UNDERLYING')
        print(self.underlying, '\n')
        print('MARKET ENVIRONMENT')
        print('\n**Constants**')
        for key, value in self.mar_env.constants.items():
            print(key, value)
        print('\n**Lists**')
        for key, value in self.mar_env.lists.items():
            print(key, value)
        print('\n**Curves**')
        for key, value in self.mar_env.curves.items():
            print(key, value)
        print('\nOPTION TYPE')
        print(self.otype, '\n')
        print('PAYOFF FUNCTION')
        print(self.payoff_func)

# a use case
me_gbm = MarketEnvironment('me_gbm', dt.datetime(2020, 1, 1))
me_gbm.add_constant('initial_value', 36.)
me_gbm.add_constant('volatility', 0.2)
me_gbm.add_constant('currency', 'EUR')
me_gbm.add_constant('model', 'gbm')

me_am_put = MarketEnvironment('me_am_put', dt.datetime(2020, 1, 1))
me_am_put.add_constant('maturity', dt.datetime(2020, 12, 31))
me_am_put.add_constant('strike', 40.)
me_am_put.add_constant('currency', 'EUR')

payoff_func = 'np.maximum(strike - instrument_values, 0)'

am_put_pos = DerivativesPosition(name='am_put_pos',
                                 quantity=3,
                                 underlying='gbm',
                                 mar_env=me_am_put,
                                 otype='American',
                                 payoff_func=payoff_func)

am_put_pos.get_info()

# Derivatives Portfolios

# models available for risk factor modeling
models = {'gbm': geometric_brownian_motion,
          'jd': jump_diffusion,
          'srd': square_root_diffusion}

# allowed exercise types
otypes = {'European': valuation_mcs_european,
          'American': valuation_mcs_american}


class DerivativesPortfolio(object):
    ''' Class for modeling and valuing portfolios of derivatives positions.
    Attributes
    ==========
    name: str
        name of the object
    positions: dict
        dictionary of positions (instances of derivatives_position class)
    val_env: market_environment
        market environment for the valuation
    assets: dict
        dictionary of market environments for the assets
    correlations: list
        correlations between assets
    fixed_seed: bool
        flag for fixed random number generator seed
    Methods
    =======
    get_positions:
        prints information about the single portfolio positions
    get_statistics:
        returns a pandas DataFrame object with portfolio statistics
    '''

    def __init__(self, name, positions, val_env, assets,
                 correlations=None, fixed_seed=False):
        self.name = name
        self.positions = positions
        self.val_env = val_env
        self.assets = assets
        self.underlyings = set()
        self.correlations = correlations
        self.time_grid = None
        self.underlying_objects = {}
        self.valuation_objects = {}
        self.fixed_seed = fixed_seed
        self.special_dates = []
        for pos in self.positions:
            # determine earliest starting_date
            self.val_env.constants['starting_date'] = \
                min(self.val_env.constants['starting_date'],
                    positions[pos].mar_env.pricing_date)
            # determine latest date of relevance
            self.val_env.constants['final_date'] = \
                max(self.val_env.constants['final_date'],
                    positions[pos].mar_env.constants['maturity'])
            # collect all underlyings and
            # add to set (avoids redundancy)
            self.underlyings.add(positions[pos].underlying)

        # generate general time grid
        start = self.val_env.constants['starting_date']
        end = self.val_env.constants['final_date']
        time_grid = pd.date_range(start=start, end=end,
                                  freq=self.val_env.constants['frequency']
                                  ).to_pydatetime()
        time_grid = list(time_grid)
        for pos in self.positions:
            maturity_date = positions[pos].mar_env.constants['maturity']
            if maturity_date not in time_grid:
                time_grid.insert(0, maturity_date)
                self.special_dates.append(maturity_date)
        if start not in time_grid:
            time_grid.insert(0, start)
        if end not in time_grid:
            time_grid.append(end)
        # delete duplicate entries
        time_grid = list(set(time_grid))
        # sort dates in time_grid
        time_grid.sort()
        self.time_grid = np.array(time_grid)
        self.val_env.add_list('time_grid', self.time_grid)

        if correlations is not None:
            # take care of correlations
            ul_list = sorted(self.underlyings)
            correlation_matrix = np.zeros((len(ul_list), len(ul_list)))
            np.fill_diagonal(correlation_matrix, 1.0)
            correlation_matrix = pd.DataFrame(correlation_matrix,
                                              index=ul_list, columns=ul_list)
            for i, j, corr in correlations:
                corr = min(corr, 0.999999999999)
                # fill correlation matrix
                correlation_matrix.loc[i, j] = corr
                correlation_matrix.loc[j, i] = corr
            # determine Cholesky matrix
            cholesky_matrix = np.linalg.cholesky(np.array(correlation_matrix))

            # dictionary with index positions for the
            # slice of the random number array to be used by
            # respective underlying
            rn_set = {asset: ul_list.index(asset)
                      for asset in self.underlyings}

            # random numbers array, to be used by
            # all underlyings (if correlations exist)
            random_numbers = sn_random_numbers((len(rn_set),
                                                len(self.time_grid),
                                                self.val_env.constants['paths']),
                                               fixed_seed=self.fixed_seed)

            # add all to valuation environment that is
            # to be shared with every underlying
            self.val_env.add_list('cholesky_matrix', cholesky_matrix)
            self.val_env.add_list('random_numbers', random_numbers)
            self.val_env.add_list('rn_set', rn_set)

        for asset in self.underlyings:
            # select market environment of asset
            mar_env = self.assets[asset]
            # add valuation environment to market environment
            mar_env.add_environment(val_env)
            # select right simulation class
            model = models[mar_env.constants['model']]
            # instantiate simulation object
            if correlations is not None:
                self.underlying_objects[asset] = model(asset, mar_env,
                                                       corr=True)
            else:
                self.underlying_objects[asset] = model(asset, mar_env,
                                                       corr=False)

        for pos in positions:
            # select right valuation class (European, American)
            val_class = otypes[positions[pos].otype]
            # pick market environment and add valuation environment
            mar_env = positions[pos].mar_env
            mar_env.add_environment(self.val_env)
            # instantiate valuation class
            self.valuation_objects[pos] = \
                val_class(name=positions[pos].name,
                          mar_env=mar_env,
                          underlying=self.underlying_objects[
                    positions[pos].underlying],
                payoff_func=positions[pos].payoff_func)

    def get_positions(self):
        ''' Convenience method to get information about
        all derivatives positions in a portfolio. '''
        for pos in self.positions:
            bar = '\n' + 50 * '-'
            print(bar)
            self.positions[pos].get_info()
            print(bar)

    def get_statistics(self, fixed_seed=False):
        ''' Provides portfolio statistics. '''
        res_list = []
        # iterate over all positions in portfolio
        for pos, value in self.valuation_objects.items():
            p = self.positions[pos]
            pv = value.present_value(fixed_seed=fixed_seed)
            res_list.append([
                p.name,
                p.quantity,
                # calculate all present values for the single instruments
                pv,
                value.currency,
                # single instrument value times quantity
                pv * p.quantity,
                # calculate Delta of position
                value.delta() * p.quantity,
                # calculate Vega of position
                value.vega() * p.quantity,
            ])
        # generate a pandas DataFrame object with all results
        res_df = pd.DataFrame(res_list,
                              columns=['name', 'quant.', 'value', 'curr.',
                                       'pos_value', 'pos_delta', 'pos_vega'])
        return res_df


# a use case
me_jd = MarketEnvironment('me_jd', me_gbm.pricing_date)
me_jd.add_constant('lambda', 0.3)
me_jd.add_constant('mu', -0.75)
me_jd.add_constant('delta', 0.1)
me_jd.add_environment(me_gbm)
me_jd.add_constant('model', 'jd')  # needed for portfolio valuation

# a European call based on this new simulation object
me_eur_call = MarketEnvironment('me_eur_call', me_jd.pricing_date)
me_eur_call.add_constant('maturity', dt.datetime(2020, 6, 30))
me_eur_call.add_constant('strike', 38.)
me_eur_call.add_constant('currency', 'EUR')

payoff_func = 'np.maximum(maturity_value - strike, 0)'

eur_call_pos = DerivativesPosition(name='eur_call_pos',
                                   quantity=5,
                                   underlying='jd',
                                   mar_env=me_eur_call,
                                   otype='European',
                                   payoff_func=payoff_func)

# compile a MarketEnvironment for the portfolio valuation
underlyings = {'gbm': me_gbm, 'jd': me_jd}  # relevant risk factors
positions = {'am_put_pos': am_put_pos,      # relevant portfolio positions
             'eur_call_pos': eur_call_pos}

val_env = MarketEnvironment('general', me_gbm.pricing_date)
val_env.add_constant('frequency', 'W')
val_env.add_constant('paths', 25000)
val_env.add_constant('starting_date', val_env.pricing_date)
val_env.add_constant('final_date', val_env.pricing_date)  # final date is not yet known; set pricing_date as preliminary
val_env.add_curve('discount_curve', ConstantShortRate('csr', 0.06))  # unique discounting object for the portfolio valuation

portfolio = DerivativesPortfolio(name='portfolio',
                                 positions=positions,
                                 val_env=val_env,
                                 assets=underlyings,
                                 fixed_seed=False)

portfolio.get_statistics(fixed_seed=False)

portfolio.get_positions()

portfolio.valuation_objects['am_put_pos'].present_value()
portfolio.valuation_objects['eur_call_pos'].delta()

path_no = 888
path_gbm = portfolio.underlying_objects['gbm'].get_instrument_values()[:, path_no]
path_jd = portfolio.underlying_objects['jd'].get_instrument_values()[:, path_no]

# non-correlated risk factors (2 sample paths)
plt.figure()
plt.plot(portfolio.time_grid, path_gbm, 'r', label='gbm')
plt.plot(portfolio.time_grid, path_jd, 'b', label='jd')
plt.xticks(rotation=30)
plt.legend(loc=0)
plt.show()


correlations = [['gbm', 'jd', 0.9]]

port_corr = DerivativesPortfolio(name='portfolio',
                                 positions=positions,
                                 val_env=val_env,
                                 assets=underlyings,
                                 correlations=correlations,
                                 fixed_seed=True)

port_corr.get_statistics()

path_no = 888
path_gbm = port_corr.underlying_objects['gbm'].get_instrument_values()[:, path_no]
path_jd = port_corr.underlying_objects['jd'].get_instrument_values()[:, path_no]

# correlated risk factors (2 sample paths)
plt.figure()
plt.plot(port_corr.time_grid, path_gbm, 'r', label='gbm')
plt.plot(port_corr.time_grid, path_jd, 'b', label='jd')
plt.xticks(rotation=30)
plt.legend(loc=0)
plt.show()

# frequency distribution of the portfolio present value of two options
pv1 = 5 * port_corr.valuation_objects['eur_call_pos'].present_value(full=True)[1]
pv2 = 3 * port_corr.valuation_objects['am_put_pos'].present_value(full=True)[1]

plt.figure()
plt.hist([pv1, pv2], bins=25, label=['European Call', 'American Put'])
plt.axvline(pv1.mean(), color='r', ls='dashed', lw=1.5, label=f'call mean = {pv1.mean():4.2f}')
plt.axvline(pv2.mean(), color='r', ls='dotted', lw=1.5, label=f'put mean = {pv2.mean():4.2f}')
# plt.xlim(0, 80)
# plt.ylim(0, 10000)
plt.legend()
plt.show()

# portfolio frequency distribution of present values
pvs = pv1 + pv2
plt.figure()
plt.hist(pvs, bins=50, label='portfolio')
plt.axvline(pvs.mean(), color='r', ls='dashed', lw=1.5, label=f'mean = {pvs.mean():4.2f}')
plt.legend()
plt.show()

# what impact does the correlation between the 2 risk factors have on the risk of the portfolio?
pvs.std()  # correlated

# not correlated
pv1 = 5 * portfolio.valuation_objects['eur_call_pos'].present_value(full=True)[1]
pv2 = 3 * portfolio.valuation_objects['am_put_pos'].present_value(full=True)[1]

(pv1 + pv2).std()
