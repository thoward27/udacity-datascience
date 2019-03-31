#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import itertools
import pickle
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import dos2unix as d2u

# functions
def check_all_negatives(y_test):
    """ check if all 0s is better:
    """
    pred = [0 for i in range(0, len(y_test))]
    score = accuracy_score(y_test, pred)
    print(score)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# variables
np.set_printoptions(precision=2)
data_dict = d2u.pickle_load("../final_project/final_project_dataset.pkl")
features_list = ["poi", "salary"]

# splitting data apart
data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson14_keys.pkl')
y, x = targetFeatureSplit(data)
class_labels = list(set(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

report = []

### fitting and scoring the decision tree: 
tree_clf = DecisionTreeClassifier()
t0 = time.time()
tree_clf.fit(x_train, y_train)
t1 = time.time()
score = tree_clf.score(x_test, y_test)
t2 = time.time()
report.append(f"Decision tree scored: {score:0.2}, and took: {t1-t0:0.2} to fit. {t2-t1:0.2} to score.")

### checking POI count in test set
pred = tree_clf.predict(x_test)
report.append(f"Predicted {sum(pred)} POIs, out of {len(pred)}")

# build confusion matrix
confusion = (confusion_matrix(y_test, pred, labels=[0, 1]))
#plt.figure()
#plot_confusion_matrix(confusion, classes=class_labels,
#                      title='Confusion matrix, without normalization')

precision = precision_score(
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0], 
    [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
)
recall = recall_score(
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0], 
    [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
)
report.append(f"Finished with precision: {precision}, and recall {recall}")

# print out our results
for string in report:
    print(string)

#plt.show()
