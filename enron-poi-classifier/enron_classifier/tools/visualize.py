"""
Visualize performance.
"""
from plotly.offline import plot
from plotly.graph_objs import Scatter

import csv
import pandas as pd

def progress():
    """ prints all runs and progress """
    x = []
    y = []

    plot([Scatter(x=x, y=x)])
    