#!/usr/bin/python

import numpy as np
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []
    ### your code goes here
    errors = abs(predictions - net_worths)
    i_cutoff = int(len(predictions) * .1)
    for i, val in enumerate(predictions):
        row = (ages[i, 0], net_worths[i, 0], errors[i, 0])
        cleaned_data.append(row)
    print(cleaned_data)
    print(sorted(cleaned_data, key=lambda row: row[2], reverse=False)[:-i_cutoff])
    return sorted(cleaned_data, key=lambda row: row[2], reverse=False)[:-i_cutoff]