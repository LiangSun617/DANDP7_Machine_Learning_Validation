
#### Why Use Training & Testing Data?

- Gives estimate of performance on an independent dataset.
- Serves as check on overfitting


http://scikit-learn.org/stable/modules/cross_validation.html

#### Train/Test Split in Sklearn

```PYTHON

#!/usr/bin/python

"""
PLEASE NOTE:
The api of train_test_split changed and moved from sklearn.cross_validation to
sklearn.model_selection(version update from 0.17 to 0.18)

The correct documentation for this quiz is here:
http://scikit-learn.org/0.17/modules/cross_validation.html
"""

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
features = iris.data
labels = iris.target

###############################################################
### YOUR CODE HERE
###############################################################

### import the relevant code and make your train/test split
### name the output datasets features_train, features_test,
### labels_train, and labels_test
# PLEASE NOTE: The import here changes depending on your version of sklearn
from sklearn import cross_validation # for version 0.17
# For version 0.18
# from sklearn.model_selection import train_test_split


### set the random_state to 0 and the test_size to 0.4 so
### we can exactly check your result
features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels, \
test_size = 0.4, random_state = 0)

###############################################################
# DONT CHANGE ANYTHING HERE
clf = SVC(kernel="linear", C=1.)
clf.fit(features_train, labels_train)

print clf.score(features_test, labels_test)
##############################################################
def submitAcc():
    return clf.score(features_test, labels_test)

```

Here's your output:
0.966666666667

#### Where to use training vs. testing data?

Train:

pca.fit(training_features)

pca.transform(training_features)

svc.fit(training_features)


Test:

pca.transform(test_features)

svc.predict(test_features)

####  K-Fold CV in sklearn

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

There's a simple way to randomize the events in sklearn k-fold CV: set the shuffle flag to true.

Then you'd go from something like this:

cv = KFold( len(authors), 2 )

To something like this:

cv = KFold( len(authors), 2, shuffle=True )

##### Practical Advice For K-Fold In Sklearn
If our original data comes in some sort of sorted fashion, then we will want to first shuffle the order of the data points before splitting them up into folds, or otherwise randomly assign data points to each fold. If we want to do this using KFold(), then we can add the "shuffle = True" parameter when setting up the cross-validation object.

If we have concerns about class imbalance, then we can use the StratifiedKFold() class instead. Where KFold() assigns points to folds without attention to output class, StratifiedKFold() assigns data points to folds so that each fold has approximately the same number of data points of each output class. This is most useful for when we have imbalanced numbers of data points in your outcome classes (e.g. one is rare compared to the others). For this class as well, we can use "shuffle = True" to shuffle the data points' order before splitting into folds.

#### GridSearchCV in sklearn

GridSearchCV is a way of systematically working through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. The beauty is that it can work through many combinations in only a couple extra lines of code.

Here's an example from the sklearn documentation:

```PYTHON
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)

```

Let's break this down line by line.

```PYTHON
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
```
A dictionary of the parameters, and the possible values they may take. In this case, they're playing around with the kernel (possible choices are 'linear' and 'rbf'), and C (possible choices are 1 and 10).

Then a 'grid' of all the following combinations of values for (kernel, C) are automatically generated:
```
('rbf', 1)	('rbf', 10)
('linear', 1)	('linear', 10)
```

Each is used to train an SVM, and the performance is then assessed using cross-validation.
```PYTHON
svr = svm.SVC()
```
This looks kind of like creating a classifier, just like we've been doing since the first lesson. But note that the "clf" isn't made until the next line--this is just saying what kind of algorithm to use. Another way to think about this is that the "classifier" isn't just the algorithm in this case, it's algorithm plus parameter values. Note that there's no monkeying around with the kernel or C; all that is handled in the next line.
```PYTHON
clf = grid_search.GridSearchCV(svr, parameters)
```
This is where the first bit of magic happens; the classifier is being created. We pass the algorithm (svr) and the dictionary of parameters to try (parameters) and it generates a grid of parameter combinations to try.
```PYTHON
clf.fit(iris.data, iris.target)
```
And the second bit of magic. The fit function now tries all the parameter combinations, and returns a fitted classifier that's automatically tuned to the optimal parameter combination. You can now access the parameter values via  $ clf.best_params_ $.

### Validation Mini-Project

In this mini-project, you’ll start from scratch in making a training-testing split in the data. This will be the first step toward your final project, of building a POI identifier.


```python

#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!
    Start by loading/formatting the data
    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

```

You’ll start by building the simplest imaginable (unvalidated) POI identifier. The starter code (validation/validate_poi.py) for this lesson is pretty bare--all it does is read in the data, and format it into lists of labels and features. Create a decision tree classifier (just use the default parameters), train it on all the data (you will fix this in the next part!), and print out the accuracy. THIS IS AN OVERFIT TREE, DO NOT TRUST THIS NUMBER! Nonetheless, what’s the accuracy?

#### Your First (Overfit) POI Identifier

From Python 3.3 forward, a change to the order in which dictionary keys are processed was made such that the orders are randomized each time the code is run. This will cause some compatibility problems with the graders and project code, which were run under Python 2.7. To correct for this, add the following argument to the **featureFormat** call on line 25 of **validate_poi.py**:
```PYTHON
sort_keys = '../tools/python2_lesson13_keys.pkl'
```
This will open up a file in the **tools** folder with the Python 2 key order.

Note: If you are not getting the results expected by the grader, then you may want to check the file **tools/feature_format.py**. Due to changes in the final project, some file changes have affected the numbers output on this assignment as written. Check that you have the most recent version of the file from the repository, such that the **featureFormat** has a default parameter for **sort_keys = False** and that **keys = dictionary.keys()** results.


```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
pred = clf.predict(features)
```


```python
from sklearn.metrics import accuracy_score
accuracy_score(pred, labels)
```




    0.98947368421052628




```python
clf.score(features,labels)
```




    0.98947368421052628



Pretty high accuracy, huh?  Yet another case where testing on the training data would make you think you were doing amazingly well, but as you already know, that's exactly what holdout test data is for...

#### Deploying a Training/Testing Regime

Now you’ll add in training and testing, so that you get a trustworthy accuracy number. Use the train_test_split validation available in sklearn.cross_validation; hold out 30% of the data for testing and set the random_state parameter to 42 (random_state controls which points go into the training set and which are used for testing; setting it to 42 means we know exactly which events are in which set, and can check the results you get). What’s your updated accuracy?


```python
#split the data
from sklearn import cross_validation   
features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels, test_size = 0.3, random_state = 42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
clf.score(features_test, labels_test)
```




    0.72413793103448276



Aaaand the testing data brings us back down to earth after that 99% accuracy in the last quiz.

### Evaluation Mini-Project

#### Applying Metrics to Your POI Identifier

Go back to your code from the last lesson, where you built a simple first iteration of a POI identifier using a decision tree and one feature. Copy the POI identifier that you built into the skeleton code in evaluation/evaluate_poi_identifier.py. Recall that at the end of that project, your identifier had an accuracy (on the test set) of 0.724. Not too bad, right? Let’s dig into your predictions a little more carefully.

From Python 3.3 forward, a change to the order in which dictionary keys are processed was made such that the orders are randomized each time the code is run. This will cause some compatibility problems with the graders and project code, which were run under Python 2.7. To correct for this, add the following argument to the **featureFormat** call on line 25 of **evaluate_poi_identifier.py*:

**
sort_keys = '../tools/python2_lesson14_keys.pkl'
**

This will open up a file in the **tools** folder with the Python 2 key order.

#### Number of POIs in Test Set

How many POIs are predicted for the test set for your POI identifier?

(Note that we said test set! We are not looking for the number of POIs in the whole dataset.)

How many people total are in your test set?


```python
sum(labels_test)
```




    4.0




```python
len(labels_test)
```




    29



If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?

25/29 = 0.862

#### Number of True Positives

Look at the predictions of your model and compare them to the true test labels. Do you get any true positives? (In this case, we define a true positive as a case where both the actual label and the predicted label are 1)


```python
pred = clf.predict(features_test)
pred
```




    array([ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.])




```python
import numpy as np
np.asarray(labels_test)
```




    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,
            0.,  1.,  0.])



There is no true positives.

#### Unpacking into Precision and Recall

As you may now see, having imbalanced classes like we have in the Enron dataset (many more non-POIs than POIs) introduces some special challenges, namely that you can just guess the more common class label for every point, not a very insightful strategy, and still get pretty good accuracy!

Precision and recall can help illuminate your performance better. Use the precision_score and recall_score available in sklearn.metrics to compute those quantities.

What’s the precision?


```python
precision = 0  # because true positive = 0
```

What’s the recall?

(Note: you may see a message like UserWarning: The precision and recall are equal to zero for some labels. Just like the message says, there can be problems in computing other metrics (like the F1 score) when precision and/or recall are zero, and it wants to warn you when that happens.)

Obviously this isn’t a very optimized machine learning strategy (we haven’t tried any algorithms besides the decision tree, or tuned any parameters, or done any feature selection), and now seeing the precision and recall should make that much more apparent than the accuracy did.


```python
recall = 0  # because true positive = 0
```

#### How Many True Positives? True Negatives? False Positives? False Negatives? Precision? Recall?

In the final project you’ll work on optimizing your POI identifier, using many of the tools learned in this course. Hopefully one result will be that your precision and/or recall will go up, but then you’ll have to be able to interpret them.

Here are some made-up predictions and true labels for a hypothetical test set; fill in the following boxes to practice identifying true positives, false positives, true negatives, and false negatives. Let’s use the convention that “1” signifies a positive result, and “0” a negative.

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]

true labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

How many true positives are there?  6

How many true negatives are there in this example?  9

How many false positives are there? 3

How many false negatives are there? 2

What's the precision of this classifier? 6/(6+3) = 0.67

What's the recall of this classifier? 6/(6+2) = 0.75

#### Making Sense of Metrics 1

Fill in the blank:

“My true positive rate is high, which means that when a ___ is present in the test data, I am good at flagging him or her.”

POI.


“My identifier doesn’t have great _, but it does have good _. That means that, nearly every time a POI shows up in my test set, I am able to identify him or her. The cost of this is that I sometimes get some false positives, where non-POIs get flagged.”

Precision/Recall.


“My identifier doesn’t have great _, but it does have good __. That means that whenever a POI gets flagged in my test set, I know with a lot of confidence that it’s very likely to be a real POI and not a false alarm. On the other hand, the price I pay for this is that I sometimes miss real POIs, since I’m effectively reluctant to pull the trigger on edge cases.”

Recall/Precision.


“My identifier has a really great _.

This is the best of both worlds. Both my false positive and false negative rates are _, which means that I can identify POI’s reliably and accurately. If my identifier finds a POI then the person is almost certainly a POI, and if the identifier does not flag someone, then they are almost certainly not a POI.”

F1 score/low.

#### Metrics for Your POI Identifier

There’s usually a tradeoff between precision and recall--which one do you think is more important in your POI identifier? There’s no right or wrong answer, there are good arguments either way, but you should be able to interpret both metrics and articulate which one you find most important and why.
