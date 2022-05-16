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
from itertools import product

sma1 = range(20, 61, 4)
sma2 = range(180, 281, 10)

results = pd.DataFrame()
for SMA1, SMA2 in product(sma1, sma2):
    data = pd.DataFrame(raw[symbol]).dropna()
    data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
    data['SMA1'] = data[symbol].rolling(SMA1).mean()
    data['SMA2'] = data[symbol].rolling(SMA2).mean()
    data = data.dropna()
    data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
    data['Strategy'] = data['Position'].shift(1) * data['Returns']
    data = data.dropna()
    perf = np.exp(data[['Returns', 'Strategy']].sum())
    results = results.append(pd.DataFrame({'SMA1': SMA1,
                                           'SMA2': SMA2,
                                           'MARKET': perf['Returns'],
                                           'STRATEGY': perf['Strategy'],
                                           'OUT': perf['Strategy'] - perf['Returns']},
                                           index=[0]), ignore_index=True)

print(results.sort_values('OUT', ascending=False).head(7))

# Random Walk Hypothesis
symbol = '.SPX'
data = pd.DataFrame(raw[symbol])

lags, cols = 5, []

for lag in range(1, lags + 1):
    # defines a column name for the current lag value
    col = f'lag_{lag}'
    # creates a lagged version of the market prices for the current lag value
    data[col] = data[symbol].shift(lag)
    # collects the column names for later reference
    cols.append(col)

data.head(7)
data = data.dropna()

import statsmodels.api as sm

model = sm.OLS(data[data.columns[0]], sm.add_constant(data[data.columns[1:]])).fit()

model.summary()

model.params[1:].plot.bar()
plt.show()

pd.DataFrame(model.predict()).plot(label='Model Prediction')
data[data.columns[0]].plot(label='.SPX')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Linear OLS Regression
## The Data
symbol = 'EUR='
data = pd.DataFrame(raw[symbol])
data['returns'] = np.log(data / data.shift(1))
data = data.dropna()
data['direction'] = np.sign(data['returns']).astype(int)

data.head()

# histogram of log returns for EUR/USD exchange rate
data['returns'].hist(bins=35)
plt.show()

lags = 2

def create_lags(data):
    data = data.copy()
    cols = []
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'
        data[col] = data['returns'].shift(lag)
        cols.append(col)
    return data

data = create_lags(data).dropna()

data.plot.scatter(x='lag_1', y='lag_2', c='returns', cmap='coolwarm', colorbar=True)
plt.axvline(0, c='r', ls='--')
plt.axhline(0, c='r', ls='--')
plt.show()

## Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# implement regression on the log returns directly
data['pos_ols_1'] = model.fit(data[cols], data['returns']).predict(data[cols])

# and on the direction data
data['pos_ols_2'] = model.fit(data[cols], data['direction']).predict(data[cols])

data[['returns', 'pos_ols_1', 'direction', 'pos_ols_2']].head()

# transform real-valued predictions to directional values (+-1)
data[['pos_ols_1', 'pos_ols_2']] = np.where(data[['pos_ols_1', 'pos_ols_2']] > 0, 1, -1)

# the two approaches yield different directional predictions in general
data['pos_ols_1'].value_counts()
data['pos_ols_2'].value_counts()

# Vectorized Backtesting
data['strat_ols_1'] = data['pos_ols_1'] * data['returns']
data['strat_ols_2'] = data['pos_ols_2'] * data['returns']
data[['returns', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp)

(data['direction'] == data['pos_ols_1']).value_counts()
(data['direction'] == data['pos_ols_2']).value_counts()

data[['returns', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot()
plt.show()

# Clustering
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2, random_state=0)
model.fit(data[cols])

data['pos_clus'] = model.predict(data[cols])
data['pos_clus'] = np.where(data['pos_clus'] == 1, -1, 1)

# two clusters identified by the k-means algorithm
plt.figure()
plt.scatter(data[cols].iloc[:, 0], data[cols].iloc[:, 1],
            c=data['pos_clus'], cmap='coolwarm')
plt.show()

data['strat_clus'] = data['pos_clus'] * data['returns']

data[['returns', 'strat_clus']].sum().apply(np.exp)

(data['direction'] == data['pos_clus']).value_counts()

data[['returns', 'strat_clus']].cumsum().apply(np.exp).plot()
plt.show()

# Frequency
def create_bins(data, bins=[0]):
    data = data.copy()
    global cols_bin
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        # digitize the feature values given the bins parameter
        data[col_bin] = np.digitize(data[col], bins=bins)
        cols_bin.append(col_bin)
    return data

data = create_bins(data)

# show the digitized feature values and the label values
data[cols_bin + ['direction']].head()

# show the frequency of the possible movements conditional on the feature value combinations
grouped = data.groupby(cols_bin + ['direction'])

# transform the dataframe object to have the frequencies in columns
res = grouped['direction'].size().unstack(fill_value=0)

def highlight_max(s):
    is_max = s == s.max()
    # highlight the highest-frequency value per feature value combination
    return ['background-color: yellow' if v else '' for v in is_max]

# translate the findings given the frequencies to a trading strategy
data['pos_freq'] = np.where(data[cols_bin].sum(axis=1) == 2, -1, 1)
(data['direction'] == data['pos_freq']).value_counts()

data['strat_freq'] = data['pos_freq'] * data['returns']
data[['returns', 'strat_freq']].sum().apply(np.exp)

data[['returns', 'strat_freq']].cumsum().apply(np.exp).plot()
plt.show()

# Classification

## Two Binary Features
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

C = 1
models = {'log_reg': linear_model.LogisticRegression(C=C),
          'gauss_nb': GaussianNB(),
          'svm': SVC(C=C)}

def fit_models(data):
    """Fit all the models"""
    mfit = {model: models[model].fit(data[cols_bin], data['direction'])
            for model in models.keys()}
    return mfit

models = fit_models(data)

def derive_positions(data):
    """derive all position values from the fitted models"""
    for model in models.keys():
        data['pos_' + model] = models[model].predict(data[cols_bin])

derive_positions(data)

def evaluate_strats(data):
    global sel
    sel = []
    for model in models.keys():
        col = 'strat_' + model
        data[col] = data['pos_' + model] * data['returns']
        sel.append(col)
    sel.insert(0, 'returns')

evaluate_strats(data)

sel.insert(1, 'strat_freq')

data[sel].sum().apply(np.exp)

data[sel].cumsum().apply(np.exp).plot()
plt.show()

## 5 binary features
data = pd.DataFrame(raw[symbol])
data['returns'] = np.log(data / data.shift(1))
data['direction'] = np.sign(data['returns'])

lags = 5
data = create_lags(data)
data = data.dropna()

data = create_bins(data)
cols_bin
data[cols_bin].head()
data = data.dropna()

from cytoolz import juxt

juxt(fit_models, derive_positions, evaluate_strats)(data)

data[sel].sum().apply(np.exp)

data[sel].cumsum().apply(np.exp).plot()
plt.show()

## 5 Digitized Features
mu = data['returns'].mean()
v = data['returns'].std()

bins = [mu - v, mu, mu + v]

cols_bin = [f'lag_{val}_bin' for val in range(1, 6)]

data = create_bins(data, bins)

models = fit_models(data)

derive_positions(data)
evaluate_strats(data)
data[sel].sum().apply(np.exp)

data[sel].cumsum().apply(np.exp).plot()
plt.show()

# Sequential Train-Test split
split = int(len(data) * 0.5)
train = data.iloc[:split].copy()
test = data.iloc[split:].copy()
fit_models(train)
derive_positions(test)
evaluate_strats(test)

test[sel].sum().apply(np.exp)

test[sel].cumsum().apply(np.exp).plot()
plt.show()

# Randomized Train-Test Split
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.5, shuffle=True, random_state=100)
train = train.copy().sort_index()
test = test.copy().sort_index()
fit_models(train)
derive_positions(test)
evaluate_strats(test)

test[sel].sum().apply(np.exp)
test[sel].cumsum().apply(np.exp).plot()
plt.show()

# Deep Neural Networks

## DNNs with sklearn
from sklearn.neural_network import MLPClassifier

# with massive overfitting
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=2 * [250], random_state=1)
model.fit(data[cols_bin], data['direction'])

data['pos_dnn_sk'] = model.predict(data[cols_bin])
data['strat_dnn_sk'] = data['pos_dnn_sk'] * data['returns']

data[['returns', 'strat_dnn_sk']].sum().apply(np.exp)

data[['returns', 'strat_dnn_sk']].cumsum().apply(np.exp).plot()
plt.show()

# with a train-test split
train, test = train_test_split(data, test_size=0.5, random_state=100)
train = train.copy().sort_index()
test = test.copy().sort_index()

model = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=500, hidden_layer_sizes=3 * [500], random_state=1)
model.fit(train[cols_bin], train['direction'])

test['pos_dnn_sk'] = model.predict(test[cols_bin])
test['strat_dnn_sk'] = test['pos_dnn_sk'] * test['returns']

test[['returns', 'strat_dnn_sk']].sum().apply(np.exp)

test[['returns', 'strat_dnn_sk']].cumsum().apply(np.exp).plot()
plt.show()

# DNNs with Tensorflow
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from cytoolz import juxt

def to_binary(value):
    if value == 1.0:
        return 1
    else:
        return 0


x_train, y_train = train[cols_bin], train['direction'].apply(to_binary)
x_test, y_test = test[cols_bin], test['direction'].apply(to_binary)

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.fit_transform(x_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model.fit(x_train_s, y_train, epochs=500)

plt.plot(np.arange(1, 501), history.history['loss'], label='Loss')
plt.plot(np.arange(1, 501), history.history['accuracy'], label='Accuracy')
plt.plot(np.arange(1, 501), history.history['precision'], label='Precision')
plt.plot(np.arange(1, 501), history.history['recall'], label='Recall')
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()
plt.show()

preds = model.predict(x_test_s)
pred_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(preds)]

juxt(confusion_matrix, accuracy_score, precision_score, recall_score)(y_test, pred_classes)

test['pos_dnn_tf'] = np.where(preds > 0, 1, -1)
test['strat_dnn_tf'] = test['pos_dnn_tf'] * test['returns']
test[['returns', 'strat_dnn_sk', 'strat_dnn_tf']].sum().apply(np.exp)

test[['returns', 'strat_dnn_sk', 'strat_dnn_tf']].cumsum().apply(np.exp).plot()
plt.show()
