import numpy as np
import pandas as pd
from pylab import plt, mpl
from pathlib import Path

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

csv_path = Path('csv_here.csv')
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

df = (df['.SPX']
      .dropna())

# view the dataframe
df.info()

# get returns
df['rets'] = np.log(df / df.shift(1))
df['vola'] = df['rets'].rolling(252).std() * np.sqrt(252)
df[['.SPX', 'vola']].plot(subplots=True, figsize=(10,6))


import eikon as ek
ek.set_app_key('too expensive')

data = ek.get_timeseries()
