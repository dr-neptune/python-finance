# simple moving averages
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')
plt.style.use('seaborn')

# data import
raw = pd.read_csv('data/tr_eikon_eod_data.csv')
symbol = 'AAPL.O'

data = raw[symbol].dropna().to_frame()

# trading strategy
SMA1 = 42
SMA2 = 252

data['SMA1'] = data[symbol].rolling(SMA1).mean()
data['SMA2'] = data[symbol].rolling(SMA2).mean()

# apple stock price with two simple moving averages
data.plot()
plt.show()

# apply stock price, two SMAs, and resulting positions
data = data.dropna()
data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
data.tail()

ax = data.plot(secondary_y = 'Position')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))
plt.show()

# vectorized backtesting
# calculate the log returns of the Apple stock
data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
# multiply the position values, shifted by a day, by the log
# returns of the Apple stock. This shift is required to avoid a foresight bias
data['Strategy'] = data['Position'].shift(1) * data['Returns']


data = data.dropna()
# sum up the log returns for the strategy and the benchmark invesetment
# and calculate the exponential value to arrive at the absolute performance
np.exp(data[['Returns', 'Strategy']].sum())
# calculate the annualized volatility for the strategy and the benchmark investment
data[['Returns', 'Strategy']].std() * 252 ** 0.5

# performance of Apple stock and SMA-based trading strategy over time
ax = data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot()
data['Position'].plot(ax=ax, secondary_y = 'Position', style='--')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))
plt.show()

# Optimization
# quick lesson on overfitting
