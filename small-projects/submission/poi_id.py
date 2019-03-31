#!python3

import sys
import shelve

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tools.dos2unix import pickle_load
from tools.feature_format import featureFormat, targetFeatureSplit
from tools.tester import test_classifier


# Constants
# ##################################################
SEED = 42
TEST_SIZE = 0.3
FEATURES = [
    'poi',
    #'salary',
    #'to_messages',
    #'deferral_payments',
    #'total_payments',
    #'loan_advances',
    #'bonus',
    #'restricted_stock_deferred',
    #'deferred_income',
    #'total_stock_value',
    #'expenses',
    #'from_poi_to_this_person',
    #'exercised_stock_options',
    #'from_messages',
    #'other',
    'from_this_person_to_poi',
    #'long_term_incentive',
    'shared_receipt_with_poi',
    #'restricted_stock',
    #'director_fees'
]

# dict of classifiers, each offering defaults and current tunings
BASE_CLFS = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoost": GradientBoostingClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    #"SVC": SVC(),
}

TUNED = {
    "TUNED:DecisionTree": DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.0,
        max_features=2,
        random_state=SEED,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        class_weight=None,
        presort=False
        ),
    "TUNED:RandomForest": RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_features="auto",
        max_depth=50,
        min_samples_split=5,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0001,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=SEED,
        warm_start=False,
        class_weight=None,
        ),
    "TUNED:AdaBoost": AdaBoostClassifier(
        algorithm='SAMME.R',
        base_estimator=DecisionTreeClassifier(min_samples_split=5, random_state=SEED),
        learning_rate=0.1,
        n_estimators=400,
        random_state=SEED,
        ),
    "TUNED:GradientBoost": GradientBoostingClassifier(
        loss='deviance',
        learning_rate=0.1,
        n_estimators=200,
        max_depth=4,
        criterion='friedman_mse',
        min_samples_split=5,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.001,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0001,
        random_state=SEED
        ),
    "TUNED:ExtraTrees": ExtraTreesClassifier(
        n_estimators=20,
        criterion="gini",
        max_features="auto",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        bootstrap=False,
        random_state=SEED,
        ),
}

GRID = {
    "GRID:DecisionTree": GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        {
            'criterion': ['gini'],
            'max_depth': [None, 4, 5, 6],
            'min_samples_split': [4, 5, 6, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'min_weight_fraction_leaf': [0., 0.01, 0.02, 0.03, 0.04, 0.05],
            'max_features': [None, 2, 3, 4, 5, "auto", "sqrt"],
            'max_leaf_nodes': [None, 5, 10, 20, 30, 40, 50, 100, 500, 750, 1000]
        },
        scoring=None,
        cv=None,
        error_score=0
        ),
    "GRID:RandomForest": GridSearchCV(
        RandomForestClassifier(random_state=SEED),
        param_grid = {
            'max_depth': [40, 45, 50, 55, 60],
            'max_leaf_nodes': [None, 5, 10, 50, 100, 200, 300, 400],
            'min_samples_leaf': [1, 2],
            'min_samples_split': [2, 3, 4],
            'n_estimators': [90, 95, 100, 105, 110]
        },
        scoring=None,
        cv=None,
        error_score=0
        ),
    "GRID:AdaBoost": GridSearchCV(
        AdaBoostClassifier(random_state=SEED),
        {
            'base_estimator': [
                None,
            ],
            'n_estimators': [390, 395, 400, 405, 410, 450, 500],
            'learning_rate': [0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.09, 0.1, 0.105],
        },
        scoring=None,
        cv=None,
        error_score=0,
        ),
    "GRID:GradientBoost": GridSearchCV(
        GradientBoostingClassifier(random_state=SEED),
        {
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 75, 100, 150, 200],
            'max_depth': [2, 3, 4, 6, 8, 10],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'min_weight_fraction_leaf': [0.0, 0.001, 0.01, 0.05],
            'max_features': ["auto", None],
            'max_leaf_nodes': [None, 2, 4, 6, 10, 50, 100],
            'min_impurity_decrease': [0.0, 0.05, 0.1],
        },
        scoring=None,
        cv=None,
        error_score=0
        ),
    "GRID:ExtraTrees": GridSearchCV(
        ExtraTreesClassifier(random_state=SEED),
        {
            'n_estimators':[5, 10, 15, 20, 40],
            'max_features':["auto", None],
            'max_depth':[None, 2, 4, 6, 8],
            'min_samples_split':[2, 3, 4],
            'min_samples_leaf':[1, 2, 3],
            'min_weight_fraction_leaf':[0., 0.001, 0.01, 0.015],
            'max_leaf_nodes':[None, 4, 6, 8, 10],
            'min_impurity_decrease':[0., 0.001, 0.0015, 0.01],
            'bootstrap':[False, True],
        },
        scoring=None,
        cv=None,
        error_score=0
        ),
}

# Custom ensemble methods:
VOTING = {
    "Tuned:HardVote": VotingClassifier(
        estimators=[(key, value) for key, value in TUNED.items()],
        voting='hard'
    ),
    "Tuned:SoftVote": VotingClassifier(
        estimators=[(key, value) for key, value in TUNED.items()],
        voting='soft'
    )
}

# Private Functions
# ##################################################
def _print_dict(data_dict):
    """ prints dictionary of data for examination """
    for key, value in data_dict.items():
        print(key)
        for k, v in value.items():
            print(f"{k}, {v}")
        break
    return True


def _precision_recall(clf, x_test, y_test):
    """ Tests, prints and returns precision, recall """
    y_pred = clf.predict(x_test)
    precision = precision_score(y_test, y_pred, labels=[0, 1])
    recall = recall_score(y_test, y_pred, labels=[0, 1])
    print(f"untuned precision: {precision}, recall: {recall}")


def _feature_importances(clf, FEATURES):
    """ finds and prints feature importances for given clf """
    results = {}
    params = zip(FEATURES, clf.feature_importances_)
    for param, importance in sorted(params, key=lambda x: x[1], reverse=True):
        param = param.replace("_", " ")
        results[param] = importance
        print(f'{param:>27}  {importance:.3f}')
    return results


def _get_dataset(data_dir="./data/final_project_dataset.pkl"):
    """ Loading / cleaning the data """
    data = pickle_load(data_dir)
    # remove outliers
    del data['TOTAL']
    return data


def _features(data, FEATURES):
    """ Create new feature(s) """
    # break it open add new components
    df = pd.DataFrame.from_dict(data, orient="index")
    df.replace(to_replace="NaN", value=np.NaN, inplace=True)
    df['portion_from_poi'] = df['from_poi_to_this_person'] / \
        (df['from_messages'] + 0.001)
    df['portion_to_poi'] = df['from_this_person_to_poi'] / \
        (df['to_messages'] + 0.001)

    # put everything back together
    df.replace(to_replace=np.NaN, value="NaN", inplace=True)
    df.to_csv('data/data.csv', index_label="index")
    data = df.to_dict(orient='index')
    FEATURES.append('portion_from_poi')
    FEATURES.append('portion_to_poi')
    return data


def _split_data(data, FEATURES, sort_keys=True, TEST_SIZE=TEST_SIZE, SEED=SEED, scale=False):
    """ extracting features, split data """
    data = featureFormat(data, FEATURES, sort_keys=sort_keys)
    y, x = targetFeatureSplit(data)
    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)


def _test_classifiers(CLASSIFIERS, x_train, x_test, y_train, y_test, my_dataset):
    """ Try a varity of classifiers """
    try:
        results = {}
        for group in CLASSIFIERS:
            for name, clf in group.items():
                print(name)
                clf.fit(x_train, y_train)
                
                if "GRID" in str(name):
                    clf = clf.best_estimator_
                
                if hasattr(clf, "_feature_importances"):
                    features_dict = _feature_importances(clf, FEATURES)
                    results[name].update(features_dict)
                
                try:
                    test_results = test_classifier(clf, my_dataset, FEATURES, folds=1000)
                except Exception as err:
                    print(err)
                    continue
                else:
                    results[name] = {}
                    clf_results[name]['clf'] = clf
                    clf_results[name]['pred'] = clf.predict(x_test)
                    results[name].update(test_results)
    except KeyboardInterrupt:
        print("aborting the classifier tests, please wait while I return the results")
        pass
    return results


# Public Functions
# ##################################################
def main():
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    my_dataset = _get_dataset()
    my_dataset = _features(my_dataset, FEATURES)
    x_train, x_test, y_train, y_test = _split_data(
        my_dataset, 
        FEATURES, 
        sort_keys=True
    )
    CLFS = [
        BASE_CLFS,
        TUNED,
        VOTING,
        GRID
    ]
    results = _test_classifiers(CLFS, x_train, x_test, y_train, y_test, my_dataset)

    print(""" analyzing results """)
    dataframe = pd.DataFrame.from_dict(results, orient="index")

    print(""" exporting results """)
    # There are certain fields we do not want to pass into the dataframe. 
    dataframe.drop(['pred'], axis=1, inplace=True)
    with shelve.open('results/current_performance', flag="c", writeback=True) as shelf:
        shelf['0'] = dataframe
        dataframe.to_csv(f'results/{timestamp}.csv', index_label="name")

    print(""" picking winner """)
    # TO BUILD

    print(""" final test """)
    #test_classifier(clf, my_dataset, FEATURES, folds=1000)
    #dump_classifier_and_data(clf, my_dataset, FEATURES)



# Main
# ##################################################
if __name__ == "__main__":
    main()
