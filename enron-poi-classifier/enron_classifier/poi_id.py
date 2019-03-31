#!python3

import sys
import shelve
import os
from itertools import compress
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

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
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from tools.dos2unix import pickle_load
from tools.feature_format import featureFormat, targetFeatureSplit
from tools.tester import test_classifier, dump_classifier_and_data


# Constants
# ##################################################
SEED = 42
TEST_SIZE = 0.3
DECOMPOSE = False
SCALE = False
KBEST = False
FEATURES_DICT = {
    'salary': False,
    'to_messages': False,
    'deferral_payments': False,
    'total_payments': False,
    'loan_advances': False,
    'bonus': False,
    'restricted_stock_deferred': False,
    'deferred_income': False,
    'total_stock_value': False,
    'expenses': False,
    'from_poi_to_this_person': False,
    'exercised_stock_options': False,
    'from_messages': False,
    'other': False,
    'from_this_person_to_poi': True,
    'long_term_incentive': False,
    'shared_receipt_with_poi': True,
    'restricted_stock': False,
    'director_fees': False,
    'compute': {
        'portion_from_poi': {
            'use': True,
            'formula': "df['from_poi_to_this_person'] / (df['from_messages'] + 0.001)",
        },
        'portion_to_poi': {
            'use': True, 
            'formula': "df['from_this_person_to_poi'] / (df['to_messages'] + 0.001)",
        }
    }
}

CLFS_DICT = {
    'DecisionTree': {
        'base': {
            'use': False,
            'clf': DecisionTreeClassifier(random_state=SEED)
        },
        'tuned': {
            'use': True,
            'clf': DecisionTreeClassifier(
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
        },
        'grid': {
            'use': False,
            'clf': GridSearchCV(
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
        },
    },
    "RandomForest": {
        'base': {
            'use': True,
            'clf': RandomForestClassifier(random_state=SEED),
        },
        'tuned': {
            'use': False,
            'clf': RandomForestClassifier(
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
        },
        'grid': {
            'use': False,
            'clf': GridSearchCV(
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
        }
    },
    "AdaBoost": {
        'base': {
            'use': False,
            'clf': AdaBoostClassifier(random_state=SEED),
        },
        'tuned': {
            'use': True,
            'clf': AdaBoostClassifier(
                algorithm='SAMME.R',
                base_estimator=DecisionTreeClassifier(min_samples_split=5, random_state=SEED),
                learning_rate=0.1,
                n_estimators=400,
                random_state=SEED,
            ),
        },
        'grid': {
            'use': False,
            'clf': GridSearchCV(
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
        }
    },
    "GradientBoost": {
        'base': {
            'use': False,
            'clf': GradientBoostingClassifier(random_state=SEED),
        },
        'tuned': {
            'use': False,
            'clf': GradientBoostingClassifier(
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
        },
        'grid': {
            'use': False,
            'clf': GridSearchCV(
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
        },
    },
    "ExtraTrees": {
        'base': {
            'use': False,
            'clf': ExtraTreesClassifier(random_state=SEED),
        },
        'tuned': {
            'use': False,
            'clf': ExtraTreesClassifier(
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
        },
        'grid': {
            'use': False,
            'clf': GridSearchCV(
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
        },
    },
}

CLFS_DICT.update({
    "HardVote": {
        'tuned': {
            'use': True,
            'clf': VotingClassifier(
                estimators=[(f"{key}:tuned", payload['tuned']['clf']) for key, payload in CLFS_DICT.items() if payload['tuned']['use']],
                voting='hard'
            ),
        },
    },
    "SoftVote": {
        'tuned': {
            'use': True,
            'clf': VotingClassifier(
                estimators=[(f"{key}:tuned", payload['tuned']['clf']) for key, payload in CLFS_DICT.items() if payload['tuned']['use']],
                voting='soft'
            ),
        },
    },
})


# Private Functions
# ##################################################
def _set_features(data, FEATURES_DICT):
    """ Interprets the features dictionary, converting it to a list of used features """
    FEATURES = ['poi']
    df = pd.DataFrame.from_dict(data, orient="index")
    df.replace(to_replace="NaN", value=np.NaN, inplace=True)
    
    for feature, use in FEATURES_DICT.items():
        if feature == "compute":
            for new_feature, payload in use.items():
                if payload['use']:
                    df[new_feature] = eval(payload['formula'])
                    FEATURES.append(new_feature)
        elif use:
            FEATURES.append(feature)
    
    df.replace(to_replace=np.NaN, value="NaN", inplace=True)
    df.to_csv('data/data.csv', index_label="index")
    data = df.to_dict(orient='index')
    return data, FEATURES


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
    if hasattr(clf, "feature_importances_"):
        results = {}
        params = zip(FEATURES, clf.feature_importances_)
        for param, importance in sorted(params, key=lambda x: x[1], reverse=True):
            param = param.replace("_", " ")
            results[param] = importance
            print(f'{param:>27}  {importance:.3f}')
        return results
    else:
        return False


def _get_dataset(data_dir="./data/final_project_dataset.pkl"):
    """ Loading / cleaning the data """
    data = pickle_load(data_dir)
    # remove outliers
    del data['TOTAL']
    del data['THE TRAVEL AGENCY IN THE PARK']
    del data['LOCKHART EUGENE E']
    return data


def _split_train_test(x, y, TEST_SIZE=TEST_SIZE, SEED=SEED):
    """ extracting features, split data """    
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)


def _set_x_y_features(data, FEATURES, sort_keys=True, TEST_SIZE=TEST_SIZE, SEED=SEED, scale=SCALE, decompose=DECOMPOSE, kbest=KBEST):
    """ extracting features, split data """
    data = featureFormat(data, FEATURES, sort_keys=sort_keys)
    y, x = targetFeatureSplit(data)
    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    if decompose:
        pca = PCA(n_components=decompose)
        x = pca.fit_transform(x)
        print(f"Inputs decomposed importance: {pca.explained_variance_ratio_}")
    if kbest:
        select = SelectKBest(k=kbest)
        x = select.fit_transform(x, y)
        FEATURES = [feature for feature in compress(FEATURES, select.get_support())]
    return x, y, FEATURES


def _test_classifiers(CLASSIFIERS, x, y, FEATURES):
    """ Try a varity of classifiers """
    try:
        results = {}
        for name_, options in CLASSIFIERS.items():
            for flag, payload in options.items():
                if not payload['use']: 
                    continue
                name = f"{name_}:{flag}"
                clf = payload['clf']

                print(name)
                if flag == 'grid': # prevents the grid from being run at every test
                    x_train, y_train, x_test, y_test = _split_train_test(x, y)
                    clf.fit(x_train, y_train)
                    clf = clf.best_estimator_
                
                try:
                    print("\tTesting")
                    test_results = test_classifier(clf, x, y, feature_list=FEATURES, folds=1000)
                except KeyboardInterrupt:
                    print("aborting the classifier tests, please wait while I return the results")
                    return results
                except Exception as err:
                    print(err)
                    continue
                else:
                    results[name] = {}
                    results[name]['clf'] = clf
                    results[name].update(test_results)
                    results[name]['scaled_inputs'] = SCALE
                    results[name]['PCA'] = DECOMPOSE
                    results[name]['FEATURES'] = str(FEATURES)
                    features_dict = _feature_importances(clf, FEATURES)
                    if features_dict: results[name].update(features_dict)
    except KeyboardInterrupt:
        print("aborting the classifier tests, please wait while I return the results")
        pass
    return results


# Public Functions
# ##################################################
def main():
    # first thing we want to do is get a timestamp to log results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    # then we get the dataset
    # immediately after, we set the features to use from the features_dict
    my_dataset = _get_dataset()
    my_dataset, FEATURES = _set_features(my_dataset, FEATURES_DICT)
    x, y, FEATURES = _set_x_y_features(my_dataset, FEATURES)
    # generate results by testing each classifier against training and testing data
    results = _test_classifiers(
        CLFS_DICT, 
        x, y,
        FEATURES=FEATURES
    )
    print(""" analyzing results """)
    dataframe = pd.DataFrame.from_dict(results, orient="index")
    #print(dataframe.describe())

    print(""" exporting results """)
    with shelve.open('results/current_performance', flag="c", writeback=True) as shelf:
        shelf['0'] = dataframe
        dataframe.to_csv(f'results/{timestamp}.csv', index_label="name")

    print(""" picking winner """)
    best_f1 = dataframe['f1'].idxmax(axis=1)
    clf = dataframe.get_value(best_f1, 'clf')

    print(""" final test """)
    test_classifier(clf, x, y, FEATURES, folds=1000)
    dump_classifier_and_data(clf, my_dataset, FEATURES)



# Main
# ##################################################
if __name__ == "__main__":
    main()
