#!/usr/bin/python
import os
from time import time
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################

def test_compatability():
	import inspect
	from sklearn.utils.testing import all_estimators
	for name, clf in all_estimators():
		print(name)
	return True

### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
class TrainedModel:
	def __init__(self, features_train, labels_train, features_test, labels_test, model):
		self.model = model
		try:
			self.clf = model(random_state=42)
		except Exception as err:
			self.clf = model()
		self.score = 0
		self.fit_time = 0
		self.score_time = 0
		self.build(features_train, labels_train, features_test, labels_test)

	def build(self, features_train, labels_train, features_test, labels_test):
		t1 = time()
		self.clf.fit(features_train, labels_train)
		t2 = time()
		self.score = self.clf.score(features_test, labels_test)
		t3 = time()
		self.fit_time = t2 - t1
		self.score_time = t3 - t2
		
	
def test_many(features_train, labels_train, features_test, labels_test, models):
	trained_models = []
	for model in models:
		m = TrainedModel(features_train, labels_train, features_test, labels_test, model)
		trained_models.append(m)
	print(trained_models)
	os.system('cls')
	clf = None
	for model in sorted(trained_models, key=lambda model: model.score, reverse=True):
		if not clf: clf = model.clf
		print(f"\nModel: {model.model}")
		print(f"Finished with a score of: {model.score}")
		print(f"Took {model.fit_time} to fit, and {model.score_time} to score.")
	print(f"returning {clf}")
	return clf

################################################################################
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
#test_compatability()
# clf = test_many(
	# features_train, 
	# labels_train, 
	# features_test, 
	# labels_test, 
	# [
		# RandomForestClassifier, 
		# AdaBoostClassifier,
		# ExtraTreesClassifier
	# ]
# )
#prettyPicture(clf, features_test, labels_test)
from sklearn.model_selection import GridSearchCV
extrees = ExtraTreesClassifier()
params = {
	'n_estimators': [1, 5, 10, 15, 20],
	'max_depth': [None, 1, 5, 10, 15, 25, 35, 45, 75, 100],
	'min_samples_split': [2, 8, 10, 15, 20],
	'min_samples_leaf': [1, 2, 5, 10, 15, 20, 25],
	'max_leaf_nodes': [None, 10, 50, 100, 1000],
	'bootstrap': [False, True],
	'random_state': [42],
	'warm_start': [False, True]
}
clf = GridSearchCV(extrees, params)
clf.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print(score)
print("done!")
