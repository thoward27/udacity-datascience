#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import dos2unix as d2u

data_dict = d2u.pickle_load("../final_project/final_project_dataset.pkl")

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)


### it's all yours from here forward!  
tree_clf = DecisionTreeClassifier()
t0 = time.time()
tree_clf.fit(x_train, y_train)
t1 = time.time()
score = tree_clf.score(x_test, y_test)
t2 = time.time()
string = f"Decision tree scored: {score:0.2}, and took: {t1-t0:0.2} to fit. {t2-t1:0.2} to score."
print(string)
