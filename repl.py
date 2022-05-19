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

# The Kelly Criterion for Stocks and Indices
raw = pd.read_csv('data/tr_eikon_eod_data.csv')
symbol = '.SPX'

data = raw[symbol].to_frame()
data['returns'] = np.log(data / data.shift(1)).dropna()

mu = data.returns.mean() * 252  # annualized return
sigma = data.returns.std() * 252 ** 0.5  # annualized volatility
r = 0.0  # set the risk-free rate to 0
f = (mu - r) / sigma ** 2  # calculate the optimal Kelly fraction to be invested in the strategy

  /-equs = []
def kelly_strategy(f):
    global equs
    equ = f'equity_{f:.2f}'
    equs.append(equ)
    cap = f'capital_{f:.2f}'
    data[equ] = 1                # a new column for equity initialized to 1
    data[cap] = data[equ] * f    # a new column for capital initialized to f
    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]      # pick the right datetimeindex value for previous states
        # calculate new capital position given the return
        data.loc[t, cap] = data[cap].loc[t_1] * math.exp(data['returns'].loc[t])
        # adjust the equity value according to capital position preference
        data.loc[t, equ] = data[cap].loc[t] - data[cap].loc[t_1] + data[equ].loc[t_1]
        # adjust the capital position given the new equity position and fixed leverage ratio
        data.loc[t, cap] = data[equ].loc[t] * f

kelly_strategy(f * 0.5)    # simulate kelly criterion for 1/2 f
kelly_strategy(f * 0.66)   # 2/3 f
kelly_strategy(f)          # 1 f

print(data[equs].tail())

ax = data['returns'].cumsum().apply(np.exp).plot(legend=True)
data[equs].plot(ax=ax, legend=True)
plt.show()

# ML-Based Trading Strategy

## Vectorized Backtesting
import fxcmpy
api = fxcmpy.fxcmpy(config_file='fxcm.cfg')

data = api.get_candles('EUR/USD', period='m5',
                       start='2018-06-01 00:00:00',
                       stop='2018-06-30 00:00:00')

data = api.get_candles('EUR/USD', period='m5',
                       start='2018-06-01 00:00:00',
                       stop='2018-06-30 00:00:00')

data.iloc[-5:, 4:]

# calculate the average bid-ask spread
spread = (data['askclose'] - data['bidclose']).mean()


# calculate the mid close prices from the ask and bid close prices
data['midclose'] = (data['askclose'] + data['bidclose']) / 2

# calculate the average proportional transaction costs given the average spread
# and average mid close price
ptc = spread / data['midclose'].mean()

data['midclose'].plot(legend=True)
plt.show()

# create binarized data
data['returns'] = np.log(data['midclose'] / data['midclose'].shift(1))

data = data.dropna()

lags = 5

cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    # creates the lagged return data given the number of lags
    data[col] = data['returns'].shift(lag)
    cols.append(col)

data = data.dropna()

# transform the feature values to binary data
data[cols] = np.where(data[cols] > 0, 1, 0)

# transform the return data to directional label data
data['direction'] = np.where(data['returns'] > 0, 1, -1)


data[cols + ['direction']].head()

# fit model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model = SVC(C=1, kernel='linear', gamma='auto')
split = int(len(data) * .80)
train = data.iloc[:split].copy()
test = data.iloc[split:].copy()

model.fit(train[cols], train['direction'])
test['position'] = model.predict(test[cols])

accuracy_score(train['direction'], model.predict(train[cols]))
accuracy_score(test['direction'], test['position'])

# check performance outside of hit rate

# derive log returns for the ML-based algorithmic trading strategy
test['strategy'] = test['position'] * test['returns']

# calculate number of trades implied by the trading strategy based on changes in the position
sum(test['position'].diff() != 0)

# whenever a trade takes place, the proportional transaction costs are subtracted from
# the strategy's log return on that day
test['strategy_tc'] = np.where(test['position'].diff() != 0,
                               test['strategy'] - ptc,
                               test['strategy'])


test[['returns', 'strategy', 'strategy_tc']].sum().apply(np.exp)

test[['returns', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot()
plt.show()

## Optimal Leverage
# annualized mean returns
mean = test[['returns', 'strategy_tc']].mean() * len(data) * 12
mean

# annualized variances
var = test[['returns', 'strategy_tc']].var() * len(data) * 12
var

# annualized volatilities
vol = var ** 0.5
vol

# optimal leverage according to the Kelly criterion (full Kelly)
mean / var

# optional leverage (half Kelly)
mean / var ** 0.5

# performance of algorithmic trading strategy for different leverage values
to_plot = ['returns', 'strategy_tc']

for lev in list(range(10, 60, 10)):
    label = f'lstrategy_tc_{lev}'
    test[label] = test['strategy_tc'] * lev
    to_plot.append(label)

test[to_plot].cumsum().apply(np.exp).plot()
plt.show()

# Risk Analysis
# initial equity
equity = 3333

# relevant log returns time series
risk = pd.DataFrame(test['lstrategy_tc_30'])

# scaled by the initial equity
risk['equity'] = risk['lstrategy_tc_30'].cumsum().apply(np.exp) * equity

# cumulative maximum values over time
risk['cummax'] = risk['equity'].cummax()

# drawdown values over time
risk['drawdown'] = risk['cummax'] - risk['equity']

# max drawdown value
risk['drawdown'].max()

# point in time when it happens
t_max = risk['drawdown'].idxmax()
t_max

# identifies highs for which the drawdown must be 0
temp = risk['drawdown'][risk['drawdown'] == 0]

# calculates the timedelta values between all highs
periods = (temp.index[1:].to_pydatetime() -
           temp.index[:-1].to_pydatetime())

# the longest drawdown period in seconds
t_per = periods.max()


risk[['equity', 'cummax']].plot()
plt.axvline(t_max, c='r', alpha=0.5)
plt.show()

# VaR
import scipy.stats as scs

# define percentiles to check
percs = np.array([0.01, 0.1, 1., 2.5, 5.0, 10.0])
risk['returns'] = np.log(risk['equity'] / risk['equity'].shift(1))

# calculate the VaR values given the percentile values
VaR = scs.scoreatpercentile(equity * risk['returns'], percs)

def print_var():
    print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
    print(33 * '-')
    for pair in zip(percs, VaR):
        # translate percentile values into confidence levels and the VaR values to positive values
        print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))

print_var()

# calculate the VaR values for a time horizon of one hour

# resample the data from 5 minute to 1 hour bars
hourly = risk.resample('1H', label='right').last()
hourly['returns'] = np.log(hourly['equity'] / hourly['equity'].shift(1))

# recalculate the VaR values for the resampled data
VaR = scs.scoreatpercentile(equity * hourly['returns'], percs)

print_var()

# persisting the model object
import pickle

pickle.dump(model, open('algorithm.pkl', 'wb'))

# Online Algorithm
algorithm = pickle.load(open('algorithm.pkl', 'rb'))

sel = ['tradeId', 'amountK', 'currency', 'grossPL', 'isBuy']

def print_positions(pos):
    print('\n\n' + 50 * '=')
    print('Going {}.\n'.format(pos))
    time.sleep(1.5)  # wait for order to be executed and reflected in the open positions
    print(api.get_open_positions()[sel])  # print open positions
    print(50 * '=' + '\n\n')

# set params
symbol = 'EUR/USD'      # instrument symbol to be traded
bar = '15s'             # bar length for resampling
amount = 100            # the amount, in thousands, to be traded
position = 0            # the initial position ('neutral')
min_bars = lags + 1     # minimum # of resampled bars required for the first prediction and trade to be possible
df = pd.DataFrame()     # empty DF to be used later for the resampled data

def automated_strategy(data, dataframe):
    global min_bars, position, df
    # captures the length of the dataframe obj with the tick data
    ldf = len(dataframe)
    # resamples the tick data to the defined bar length
    df = dataframe.resample(bar, label='right').last().ffill()
    if ldf % 20 == 0:
        print('%3d' % len(dataframe), end=',')
    if len(df) > min_bars:
        min_bars = len(df)
        df['Mid'] = df[['Bid', 'Ask']].mean(axis=1)
        df['Returns'] = np.log(df['Mid'] / df['Mid'].shift(1))
        df['Direction'] = np.where(df['Returns'] > 0, 1, -1)
        # picks the relevant feature values for all lags
        features = df['Direction'].iloc[-(lags + 1):-1]
        # reshape them to a form that the model can use for prediction
        features = features.values.reshape(1, -1)
        # generate prediction value (+-1)
        signal = algorithm.predict(features)[0]

        # conditions to enter (or keep) a long position
        if position in [0, -1] and signal == 1:
            api.create_market_buy_order(symbol, amount - position * amount)
            position = 1
            print_positions('LONG')
        # conditions to enter (or keep) a short position
        elif position in [0, -1] and signal == -1:
            api.create_market_sell_order(symbol, amount + position * amount)
            position = -1
            print_positions('SHORT')

    # condition to stop trading and close out any open positions
    if len(dataframe) > 350:
        api.unsubscribe_market_data('EUR/USD')
        api.close_all()


# copied from book, still not using "data" param
# def automated_strategy(data, dataframe):
#     global min_bars, position, df
#     ldf = len(dataframe)
#     df = dataframe.resample(bar, label='right').last().ffill()
#     if ldf % 20 == 0:
#         print('%3d' % len(dataframe), end=',')
#     if len(df) > min_bars:
#         min_bars = len(df)
#         df['Mid'] = df[['Bid', 'Ask']].mean(axis=1)
#         df['Returns'] = np.log(df['Mid'] / df['Mid'].shift(1))
#         df['Direction'] = np.where(df['Returns'] > 0, 1, -1)
#         features = df['Direction'].iloc[-(lags + 1):-1]
#         features = features.values.reshape(1, -1)
#         signal = algorithm.predict(features)[0]
#         if position in [0, -1] and signal == 1:
#             api.create_market_buy_order(
#                 symbol, amount - position * amount)
#             position = 1
#             print_positions('LONG')
#         elif position in [0, 1] and signal == -1:
#             api.create_market_sell_order(
#                 symbol, amount + position * amount)
#             position = -1
#             print_positions('SHORT')
#     if len(dataframe) > 350:
#         api.unsubscribe_market_data('EUR/USD')
#         api.close_all()
