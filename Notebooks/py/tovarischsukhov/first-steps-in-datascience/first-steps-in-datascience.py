#!/usr/bin/env python
# coding: utf-8

# First steps in DataScience with Titanic.csv

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score

#matplotlib magic
get_ipython().magic(u'matplotlib inline')

#importing data
df = pd.read_csv('../input/train.csv', index_col='PassengerId')


# In[ ]:


from statistics import median

x_labels = ['Pclass', 'Fare', 'Age', 'Sex','Embarked', 'Parch', 'SibSp', 'Survived']
X = df[x_labels]

print(len(X[pd.isnull(X.Age)]))

hlp = X[X.Age.notnull()]

X.fillna(median(hlp['Age'].tolist()), inplace=True)
print(None in X)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X['Sex'] = lb.fit_transform(X['Sex'])

#all possible features
colds = ['Pclass', 'Fare', 'Age', 'Sex', 'Embarked', 'Parch', 'SibSp']
#features we use ('Parch', 'SibSp' dropp the accuracy)
cols = ['Pclass', 'Fare', 'Age', 'Sex']

#parameters of the model, that we will fit; Tried different ones, theese gives best fitting
parameters = [
    {'min_samples_split': range(2,5), 'max_features': [4], 'min_samples_leaf': [4]}
]

#divide features from answers
labels = X["Survived"].values
features = X[list(cols)].values

#classifier we use
dtc = DecisionTreeClassifier()

#grid search model to scroll parameters
clf = GridSearchCV(dtc, parameters, n_jobs=-1)

#arays to stock answers
y_pred = []
y_true = []

#cross-validation model
skf = StratifiedKFold(n_splits=3)

#loop that fits the grid search model using clross-validation
for train, test in skf.split(features, labels):
    clf.fit(features[train], labels[train])
    #colect answers
    y_pred = np.append(y_pred, clf.predict(features[test]))
    y_true = np.append(y_true, labels[test])

#print firs estimation of accuracy of the model
classif_rate = np.mean(y_pred == y_true) * 100
print("Classification rate : %f" % classif_rate)

#print report
target_names = ('Dead', 'Survived')
print(classification_report(y_true, y_pred, target_names=target_names))

print(f1_score(y_true, y_pred, average='weighted'))

print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(features[train], labels[train])
target_names = ('Dead', 'Survived')

y_pred = np.append(y_pred, rf.predict(features[test]))
y_true = np.append(y_true, labels[test])
print(classification_report(y_true, y_pred, target_names=target_names))


# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(features[train], labels[train])
target_names = ('Dead', 'Survived')

y_pred = np.append(y_pred, lr.predict(features[test]))
y_true = np.append(y_true, labels[test])
print(classification_report(y_true, y_pred, target_names=target_names))


# In[ ]:


test_df = pd.read_csv('../input/test.csv')
X_test = test_df[cols]
#print(X_test)
#print(len(X_test[pd.isnull(X_test.Age)]))

hlp = X_test[X_test.Age.notnull()]
X_test.fillna(median(hlp['Age'].tolist()), inplace=True)
#print(X_test[pd.isnull(X_test.Age)])

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X_test['Sex'] = lb.fit_transform(X_test['Sex'])
#print(X_test)
y_pred_on_test = rf.predict(X_test)
print(y_pred_on_test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_pred_on_test
    })
submission.to_csv('submission.csv', index=False)

