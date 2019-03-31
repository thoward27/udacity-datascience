# Enron POI Classifier

## How to use this
To use this script properly please duplicate the Anaconda Environment by running: `conda env create -f environment.yml`. This method should resolve Windows / Mac discrepancies, however, if you are still having trouble there is a environment_mac file with premade changes to account for the different OS. 

**PLEASE DO NOT RUN THIS CODE OUTSIDE OF THE PROVIDED ENVIRONMENT**
I've worked with a Udacity coach to resolve issues with the running of this script, it comes down to using the provided environment.


### Here are the dependencies for those determined to not use Anaconda
Package | Version | Python version
--------|---------|---------------
| decorator | 4.1.2 | py36_0
| ipython_genutils | 0.2.0 | py36_0
| jsonschema | 2.6.0 | py36_0
| jupyter_core | 4.3.0 | py36_0
| mkl | 2017.0.3 | 0
| nbformat | 4.4.0 | py36_0
| numpy | 1.13.1 | py36_0
| pandas | 0.20.3 | py36_0
| pip | 9.0.1 | py36_1
| plotly | 2.0.11 | py36_0
| python | 3.6.2 | 0
| python-dateutil | 2.6.1 | py36_0
| pytz | 2017.2 | py36_0
| requests | 2.14.2 | py36_0
| scikit-learn | 0.19.0 | np113py36_0
| scipy | 0.19.1 | np113py36_0
| setuptools | 27.2.0 | py36_1 *on mac use py36\_0*
| six | 1.10.0 | py36_0
| traitlets | 4.3.2 | py36_0
| vs2015_runtime | 14.0.25420 | *windows only*
| wheel | 0.29.0 | py36_0
| ipython-genutils | 0.2.0 | 
| jupyter-core | 4.3.0 | 

## Goal
My goal for this project has been to correctly identify Persons of Interest (POI) from the [Enron Corporation][], based solely on financial and email data. Without machine learning, this would be nearly impossible, requiring immense investments in hand-crafted algorithms that use a multitude of features. With machine learning, I've achieved a precision of 0.31625 and recall of 0.35800 with a Decision Tree Classifier of POI. This process has been elating for me, and I'm glad to be sharing it with you now.

## Data Exploration and Investigation
The Enron dataset is comprised of both financial and email data on many employees within the company. This data can be leveraged through the thoughtful use of machine learning to make predictions as to whether someone is a POI. 

During my initial investigation of the data I found just one outlier, the "TOTAL" column had been left in. My first reviewer noted two other outliers present in the data that I had not noticed: TRAVEL AGENCY IN THE PARK, which is not a real person; and LOCKHART EUGENE E, who has no non NaN values.

With all aforementioned outliers removed, there are 143 data points in the set, with 18 Persons of Interest. This means we're working with a rather small data set, with POI accounting for only %13 of the data points representing true values. Fewer data points typically mean more tuning will be necessary to achieve good results, as the model won't have many points to learn from. 

## Features
I've chosen to use only email features in my final classifier. I found that the financial features were seemingly too random to be of use. Whenever I introduced financial-based metrics the performance always suffered. From the standard email features, I've kept "from this person to poi", and "shared receipt with poi". Adding to this I've created two new features, "portion from poi" and "portion to poi": these both represent the fraction of emails from and to a POI. Without the two engineered features, precision drops to: 0.13446 and recall to 0.16700.

To select these features, I began by introducing all of them, then slowly paired away those I found not helpful for classification. To do this I created a small function to print feature importance. Unfortunately, this didn't work all that well for me. I found erratic results, sometimes my scores going up, other times going down for no explainable reason. This led to me removing all fields, then reintroducing them one by one to try and find the key balance.

In my efforts to craft the best model possible, I've also tried [PCA][], [Scaling][] and [SelectKBest][] of features, all resulted in worse results.

Since I trained a SVC Model at some point during this project, I'm obligated to discuss scaling. Scaling is when features are standardized, normally around 0, and it allows models such as SVC, or linear, to perform much better. This improvement comes from the fact that the data points will be more evenly spaced out after normalization. Since they have more space, SVCs can build larger decision boundaries, and linear models can find better fits through the data. 

Some selected results: **(For full results see the CSVs contained in the results folder)**

### ALL FEATURES USED:
#### With SelectKBest: 4
Decision Tree -> Precision: 0.373 Recall: 0.215
#### With SelectKBest: 5
Decision Tree -> Precision: 0.412 Recall: 0.236
#### With PCA: 6
Decision Tree -> Precision: 0.237 Recall: 0.163
#### With PCA: 5
Decision Tree -> Precision: 0.238 Recall: 0.150

### Manual Selection without Engineered Features
#### FEATURES  |  [ all ] 
Decision Tree -> Precision 0.42575 Recall: 0.22650

#### FEATURES  |  [ all financial data, no email ] 
Decision Tree -> Precision 0.34954 Recall: 0.115

#### FEATURES  |  [ all email, no financial ] 
Decision Tree -> Precision 0.086 Recall: 0.074

#### FEATURES  |  ['from_this_person_to_poi', 'shared_receipt_with_poi'] 
Decision Tree -> Precision 0.134 Recall: 0.167

### Manual Selection with Engineered Features
#### FEATURES  |  [ all ] 
Decision Tree -> Precision: 0.373 Recall: 0.215

#### FEATURES  |  ['from_this_person_to_poi', 'shared_receipt_with_poi', 'from_poi_to_this_person']
Decision Tree -> Precision: 0.17189 Recall: 0.19200

#### FEATURES  |  ['from_this_person_to_poi', 'shared_receipt_with_poi'] 
Decision Tree -> Precision: 0.31625 Recall: 0.35800
**This is the model and features that were used for the final**

## Algorithm
All algorithms I've tried are still present in my code; they include: [Decision Trees][], [Random Forests][], [AdaBoost models][], [Gradient Boosting][], [Extra Trees][], [SVC][], and both a [hard-voting][voting] and [soft-voting][voting] ensemble of all the models. For this project, I intentionally focused on [tree-based models][] as they have always fascinated me with their complex simplicity. At the heart of it, they're a collection of yes/no or true/false and the fact that these simple questions, compounded, can make such accurate answers delights me. 

## Parameters
All the algorithms I've tested were tuned to some degree. To perform this tuning, I used the Scikit-Learn [GridSearchCV][] function, for an exhaustive search of the parameters. This was pivotal in getting my precision and recall scores above the 0.3 requirement for this project, as the defaults averaged around 0.2 recall even under voting conditions. As an aside, the voting classifiers may have performed better had the original algorithms been more varied, I well understood this, but decided against broadening my scope as I really wanted to dive into Decision Trees as much as possible. As for why broadening can help, just remember that similar algorithms have similar inherent defects, aggregating similar defects can amplify those negative attributes, it is variety of attributes that helps mitigate this. 

In general terms, parameter tuning can be done in two main ways, exhaustive grid searches, or random searches. While there are some algorithms and models that can be solved through optimization, the ones which I've dealt with so far have all required parameter searching. Personally, I prefer the exhaustive approach, as I enjoy thoroughness. If I had access to unlimited computational power, I think I would start with a randomized search and move towards a grid search from there, to avoid any potential local minima that may be present. 

As for my winner, the Decision Tree, the parameters tuned include: max depth, a measure for how deep the decision tree will go which can prevent overfitting of the data; minimum split sample, which again can reduce overfitting by required a certain number of points to create a split; minimum samples per leaf, which controls how many samples each leaf must have, a very similar parameter to the minimum split value; max features, or the maximum features used by the decision tree; and max leaf nodes, which simply controls the maximum amount of leaves a tree is allowed to have. Again, these were tuned using a GridSearchCV, an exhaustive method. To tune all required parameters would require leaving my computer running overnight, sometimes into the next day. 

We tune parameters to prevent overfitting; especially in decision trees. If left unchecked decision trees can create immensely complex decision boundaries, splitting at every turn, which can look great on initial testing, but fails when the model is tested on data it hasn't seen. In other words, models created without some sort of tuning will not generalize well. Optimization of models is one of the more abstract arts to machine learning, as most aren't inherently solvable, there is no best solution. Even under good circumstances, optimization methods can fall into local-minima causing a false sense of success. This is why random search methods, combined with grid search methods, tends to perform the best. 

## Evaluation
To evaluate my models, I leveraged the provided testing code, with slight modifications. The modifications were: altering it to return the results, as well as print them. I then wrote these results to a CSV for further examination and charting of model progress. Although I chose to save all metrics, the ones I focused on more heavily were precision and recall. These metrics are outstanding to determine performance of a classifier, and were very helpful for model selection.

Precision measures whether the people marked POI are actually POI, as in, if five are marked POI, and all five are POI, precision is 1. Recall, balances precision, by measuring how many are missed. To walkthrough a full example: if there are ten points in our set, and five are POI; our model identifies 4 POI, of which all are actually POI. Thus, our results are: Precision 4/4, and recall 4/5. Clearly the benefits of using this approach to scoring classifiers is obvious, as it captures both sides of the problem, are people going to get away? And, are we going to falsely identify anyone?

Validation is the practice of holding a small subset of data outside of the classifiers training. This validation set checks the bias and variance of the model, making sure it is not overfitting the testing data. To validate my results I've used a stratified testing split, which signifies that the testing and training sets will have the same proportion of POI within them. This is especially important, as the dataset is not very large. Purely random splitting methods may result in sets entirely composed of a single class, which would be impossible to train on (as there's nothing to really classify at that point). This is why stratified splits are generally preferred for smaller datasets.

## Final Model
My top-performing model was a decision tree (darn you simplicity). Although I desperately wanted a more complex model to fair better this is realistically the best result. Decision trees are extremely simple to visualize, and replicate human decision making. Being able to predict at this level with a decision tree means conveying findings to those interested in the particulars, is much easier than if I had gone with something more similar to a neural network. 

Sources are contained in source.

[markdown]: https://daringfireball.net/projects/markdown/syntax "Markdown syntax"
[Enron Corporation]: https://en.wikipedia.org/wiki/Enron "Enron Wikipedia"
[Decision Trees]: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html "Decision Tree Model Details"
[Random Forests]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html "Random Forest Model Details"
[AdaBoost models]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html "AdaBoost Classifier Model Details"
[Gradient Boosting]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html "Gradient Boosting Model Details"
[Extra Trees]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html "Extra Trees Model Details"
[svc]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html "SVC Model Details"
[voting]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html "Voting Classifier"
[tree-based models]: https://en.wikipedia.org/wiki/Decision_tree_learning "Wikipedia Decision Trees"
[GridSearchCV]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html "GridSearchCV Details"
[PCA]:  http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html "PCA Details"
[scaler]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html "Standard Scaler Details"
[SelectKBest]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html "SelectKBest Details"
