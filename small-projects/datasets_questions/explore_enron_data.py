#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import sys
import re
import os
import pandas as pd
import numpy as np
os.system("cls")
sys.path.append("../tools")
import dos2unix
file = "../final_project/final_project_dataset.pkl"
enron_data = dos2unix.pickle_load(file)

enron_frame = pd.DataFrame.from_dict(enron_data, 'index')
enron_frame.replace("NaN", np.NaN, inplace=True)
#enron_frame.dropna(inplace=True)
print(enron_frame.head())
print(enron_frame[enron_frame.poi == True].count())

'''
count = 0
for person, deets in enron_data.items():
    match = bool(re.search(r"(skilling)|(lay)|(fastow)", person, re.I))
    if match:
        print(person)
        print(deets)
print(count)
'''