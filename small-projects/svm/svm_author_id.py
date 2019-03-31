#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from collections import Counter


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
try:
	from sklearn import svm
	
	'''# strip down data to 1% for working
	features_i = int(len(features_train)/100)
	labels_i = int(len(labels_train)/100)
	features_train = features_train[:features_i] 
	labels_train = labels_train[:labels_i] 
	'''
	# testing c values
	Cs = [10000]
	scores = []
	for c in Cs:
		print(f"Building model for c: {c}")
		clf = svm.SVC(
			kernel="rbf", 
			C=c
		)
		
		print('fitting')
		t1 = time()
		clf.fit(features_train, labels_train)
		t2 = time()
		
		print("predicting")
		t3 = time()
		pred = clf.predict(features_test)
		t4 = time()
		score = clf.score(features_test, labels_test)
		scores.append(f"sum chris: {Counter(pred).values()}")
		scores.append(f"(10, {pred[10]}) (26, {pred[26]}) (50, {pred[50]})")
		scores.append(f"score for C: {c}; score: {score}; fitting: {t2-t1}; predicting: {t4-t3}")
	for score in scores:
		print(score)
	input("done!")
except Exception as err:
	print(err, sys.exc_info())
	input("what?")
#########################################################