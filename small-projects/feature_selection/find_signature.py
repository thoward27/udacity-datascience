#!/usr/bin/python

import pickle
import numpy
import sys
sys.path.append('../tools')
import dos2unix as d2u
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = d2u.pickle_load(words_file)
authors = d2u.pickle_load(authors_file) 



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn.tree import DecisionTreeClassifier
DECISION_TREE = DecisionTreeClassifier()
DECISION_TREE.fit(features_train, labels_train)
print(f"scored: {DECISION_TREE.score(features_test, labels_test)}")

IMPORTANCES = DECISION_TREE.feature_importances_
INDICIES = numpy.argsort(IMPORTANCES)[::-1]
VOCAB = vectorizer.vocabulary_

print("Top 10 features:")
i = 0
for f in range(features_train.shape[1]):
    if i > 10: break
    string = (
        f"feature: {INDICIES[f]}, "
        f"importance: {IMPORTANCES[INDICIES[f]]:.2%}, "
        f"word: {list(VOCAB.keys())[list(VOCAB.values()).index(INDICIES[f])]}"
    )
    print(string)
    i += 1

