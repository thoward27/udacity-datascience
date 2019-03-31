#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import dos2unix as d2u


### read in data dictionary, convert to numpy array
data_dict = d2u.pickle_load("../final_project/final_project_dataset.pkl")
## cleaning
data_dict.pop('TOTAL')
print(data_dict)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

x, y, label = [], [], []
for key, value in data_dict.items():
    x.append(value['salary'])
    y.append(value['bonus'])
    label.append(key)

trace = go.Scatter(
    x = x,
    y = y,
    text = label,
    mode = "markers"
)

plotly.offline.plot([trace], filename="scatter.html")