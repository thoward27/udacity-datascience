#python3

"""
working through the sneak peek at feature scaling section..
"""
# imports
import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.append("../tools/")
import dos2unix as d2u
# set options
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# get data
DATA = d2u.pickle_load("../final_project/final_project_dataset.pkl")
DATA = pd.DataFrame.from_dict(DATA, orient="index")
DATA = DATA.drop(['TOTAL'], axis=0)
DATA = DATA.convert_objects(convert_dates=True, convert_numeric=True)
DATA = DATA['exercised_stock_options']
DATA.dropna(axis=0, how='any', inplace=True)
# build preprocessing scaler
MIN_MAX = MinMaxScaler()
MIN_MAX.fit(DATA)
PRED = MIN_MAX.transform([1000000.])
# summarize 
print(PRED)
print(DATA.describe())
