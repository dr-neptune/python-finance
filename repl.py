import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')

path = 'data/temp/'

# working with SQL databases
import sqlite3 as sq3

con = sq3.connect(path + 'nums.db')
query = 'CREATE TABLE nums (Date date, No1 real, No2 real)'
con.execute(query)
con.commit()
q = con.execute

q('SELECT * FROM sqlite_master').fetchall()

# populate it with data
import datetime

now = datetime.datetime.now()

q('INSERT INTO nums VALUES(?, ?, ?)', (now, 0.12, 7.3))

np.random.seed(8888)

df = np.random.standard_normal((10000, 2)).round(4)

for row in df:
    now = datetime.datetime.now()
    q('INSERT INTO nums VALUES(?, ?, ?)', (now, row[0], row[1]))

con.commit()

q('SELECT * FROM nums').fetchmany(4)

q('DROP TABLE IF EXISTS nums')
con.close()

# another one
df = np.random.standard_normal((10000000, 5)).round(4)
fname = path + 'numbers'
con = sq3.Connection(fname + '.db')
query = 'CREATE TABLE numbers (No1 real, No2 real, No3 real, No4 real, No5 real)'

q = con.execute
qm = con.executemany

q(query)
qm('INSERT INTO numbers VALUES (?, ?, ?, ?, ?)', df)
con.commit()

temp = q('SELECT * FROM numbers').fetchall()
print(temp[:3])

query = 'SELECT * FROM numbers WHERE No1 > 0 AND No2 < 0'
res = np.array(q(query).fetchall()).round(3)

res = res[::100]

plt.figure()
plt.plot(res[:, 0], res[:, 1], 'ro')
plt.show()

h5s = pd.HDFStore(fname + '.h5s', 'w')
h5s['df'] = pd.DataFrame(df)
h5s.close()

h5s = pd.HDFStore(fname + '.h5s', 'r')
data_ = h5s['df']
data_ is df
(data_ == df).all()

# I/O with PyTables
# PyTables is a Python binding for the HDF5 database standard
# It is designed to optimize the performance of I/O operations and make best use of the available hardware
import tables as tb
import datetime as dt

fname = path + 'pytab.h5'
h5 = tb.open_file(fname, 'w')

row_des = {
    'Date': tb.StringCol(26, pos=1),
    'No1': tb.IntCol(pos=2),
    'No2': tb.IntCol(pos=3),
    'No3': tb.Float64Col(pos=4),
    'No4': tb.Float64Col(pos=5)
    }
rows = 2000000

filters = tb.Filters(complevel=0)  # specify compression level

tab = h5.create_table('/', 'ints_floats',
                      row_des,
                      title='Integers and Floats',
                      expectedrows=rows,
                      filters=filters)

type(tab)
tab

# populate the table with numerical data
pointer = tab.row
ran_int = np.random.randint(0, 10000, size=(rows, 2))
ran_flo = np.random.standard_normal((rows, 2)).round(4)

for i in range(rows):
    pointer['Date'] = dt.datetime.now()
    pointer['No1'] = ran_int[i, 0]
    pointer['No2'] = ran_int[i, 1]
    pointer['No3'] = ran_int[i, 0]
    pointer['No4'] = ran_int[i, 1]
    pointer.append()

tab.flush()
tab

# pytables also provides tools to query data
query = '((No3 < -0.5) | (No3 > 0.5)) & ((No4 < -1) | (No4 > 1))'
iterator = tab.where(query)

# [(row['No3'], row['No4']) for row in iterator]
