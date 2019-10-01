#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import math
import seaborn as sns
from six.moves import cPickle as pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

get_ipython().magic(u'matplotlib inline')


# ## From previous result
# 
# From previous result, we decide to use the following models to do submission. They are:
# 1. Polynomial SVM Accuracy: 83.96%
# 2. XGboost Accuracy: 83.58%
# 3. RBF SVM Accuracy: 82.84%
# 4. Bernoulli Naive bayes Accuracy: 82.46%
# 5. Logistic Accuracy: 81.72%
# 6. Random forest Accuracy: 81.34%
# 7. Linear SVM Accuracy: 81.34%
# 8. Neural Network Accuracy: 81.34%
# 9. Extra tree Accuracy: 79.85%
# 10. Guassian Naive bayes Accuracy: 60.45%
# 
# In this section, we will run a grid search to find the best parameter for the validation set.

# In[ ]:


train_ds_file = '../input/cleansedtitanicdataset/train_dataset.pickle'
train_lb_file = '../input/cleansedtitanicdataset/train_label.pickle'
test_ds_file = '../input/cleansedtitanicdataset/test_dataset.pickle'

with open(train_ds_file, 'rb') as f:
    train_dataset = pickle.load(f)
    
with open(train_lb_file, 'rb') as f:
    train_label = pickle.load(f)
    
with open(test_ds_file, 'rb') as f:
    test_dataset = pickle.load(f)
    
def transform_ds_to_input(dataset):
    columns = ["Pclass", "Embarked_enc", "Salutation_enc", "CabinArea_enc"]
    ds_onehot = dataset[["Pclass", "Sex_enc", "SibSp", "Parch", "Fare", "CabinArea_enc",                                       "Embarked_enc", "Salutation_enc", "FamilyMember"]]
    ds_onehot = pandas.get_dummies(ds_onehot, sparse=True, columns=columns)
    scaler = StandardScaler().fit(ds_onehot)
    ds_onehot_scaled = scaler.transform(ds_onehot) 
    return ds_onehot_scaled

full_dataset = pandas.concat([train_dataset, test_dataset])
full_dataset_onehot = transform_ds_to_input(full_dataset)
train_dataset_onehot= full_dataset_onehot[:len(train_dataset)]
test_dataset_onehot = full_dataset_onehot[len(train_dataset):]

print(pandas.DataFrame(train_dataset_onehot[0:10]))


# In[ ]:


def get_train_test_set(test_size):
    X_train, X_test, y_train, y_test =         train_test_split(train_dataset_onehot, train_label, test_size=test_size)
    
    return X_train, X_test, y_train, y_test


# ## Tuning logistic regression
# 
# The model parameter to optimized are
# 1. C - The regularization term

# In[ ]:


from sklearn.linear_model import LogisticRegression

#parameters = {'C':[0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5]}
parameters = {'C':[0.001, 0.0025, 0.005, 0.0075, 0.01]}
lr = LogisticRegression()
clf = GridSearchCV(lr, parameters)
clf.fit(train_dataset_onehot, train_label)

print(clf.best_params_)


# In[ ]:


for i in range(10):
    X_train, X_test, y_train, y_test = get_train_test_set(0.3)
    lr = LogisticRegression(C = 0.075)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("%s Accuracy: %.2f%%" % ("Logistic regression, C:0.3", accuracy * 100.0))


# # Tuning RBF SVM
# 
# The model parameter to optimized are
# 
# 1. C - regularization term
# 2. Gamma - the influence of a single training example reaches

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

#parameters = {'C':[0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5], 
#              'gamma':[0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, \
#                   0.005, 0.0075, 0.01, 0.025, 0.05, 0.075]}
parameters = {'C':[0.1, 0.25, 0.5, 0.75], 
              'gamma':[0.01, 0.025, 0.05, 0.075]}

rbf = SVC()
clf = GridSearchCV(rbf, parameters, n_jobs=8)
clf.fit(train_dataset_onehot, train_label)

print(clf.best_params_)


# In[ ]:


for i in range(10):
    X_train, X_test, y_train, y_test = get_train_test_set(0.3)
    lr = SVC(gamma=0.05, C=0.75)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("%s Accuracy: %.2f%%" % ("RBF SVM, gamma:0.05, C=0.75", accuracy * 100.0))


# ## Tuning XGBoost
# 
# The model will use tree instead of linear
# 
# The parameter to be tuned are:
# 1. learning_rate
# 2. max_depth
# 3. min_child_weight
# 4. gamma
# 5. subsample
# 6. colsample_bytree
# 7. objective
# 8. learning_rate

# In[ ]:


parameters = {
    'min_child_weight':range(2,6,1),
    'max_depth':range(3,7,1),
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
    'learning_rate':[0.01, 0.1, 1]
}

#n_iter=5000
n_iter_search=10
xgb = XGBClassifier()
clf = RandomizedSearchCV(xgb, parameters, n_jobs=8, n_iter=n_iter_search)
clf.fit(train_dataset_onehot, train_label)

print(clf.best_params_)


# In[ ]:


for i in range(10):
    X_train, X_test, y_train, y_test = get_train_test_set(0.3)
    lr = XGBClassifier(learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, gamma=0.2,
                       max_depth=5, reg_alpha=0.01, min_child_weight=3, objective= 'binary:logistic')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("%s Accuracy: %.2f%%" % ("XGBoost", accuracy * 100.0))


# ## Tuning Random Forest
# 
# The parameter to be tuned are:
# 1. max_depth
# 2. max_features
# 3. min_samples_split
# 4. min_samples_leaf
# 5. bootstrap
# 6. criterion

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint

param_dist = {"n_estimators" : sp_randint(3, 20),
              "max_depth": [1, 2, 3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
#n_iter_search = 5000
n_iter_search = 10
# build a classifier
clf = RandomForestClassifier()
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, n_jobs=8)
random_search.fit(train_dataset_onehot, train_label)

print(random_search.best_params_)


# In[ ]:


for i in range(10):
    X_train, X_test, y_train, y_test = get_train_test_set(0.3)
    lr = RandomForestClassifier(max_features=9, bootstrap=True, min_samples_split=9, n_estimators=16, criterion='gini',
                       min_samples_leaf=4, max_depth=None)  
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("%s Accuracy: %.2f%%" % ("Random forest", accuracy * 100.0))


# # Submission, select the best model for Kaggle

# In[ ]:


# Submission score is 0.77990, better than gender classifier 0.76555

clf1 = RandomForestClassifier(max_features=9, bootstrap=True, min_samples_split=9, n_estimators=16, criterion='gini',
                       min_samples_leaf=4, max_depth=None)  
clf1.fit(train_dataset_onehot, train_label)
r_pred = clf1.predict(test_dataset_onehot)
r_predictions = [int(round(value)) for value in r_pred]

submission_df = pandas.DataFrame(index=test_dataset.index, columns=["Survived"])
submission_df["Survived"] = r_predictions
submission_df.to_csv("submission_best_rf.csv", sep=',')


# In[ ]:


# Submission score is 0.78469, better than gender classifier 0.76555

clf2 = XGBClassifier(learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, gamma=0.2,
                       max_depth=5, reg_alpha=0.01, min_child_weight=3, objective= 'binary:logistic')
clf2.fit(train_dataset_onehot, train_label)
r_pred = clf2.predict(test_dataset_onehot)
r_predictions = [int(round(value)) for value in r_pred]

submission_df = pandas.DataFrame(index=test_dataset.index, columns=["Survived"])
submission_df["Survived"] = r_predictions
submission_df.to_csv("submission_best_xg.csv", sep=',')


# In[ ]:


# Submission score is 0.78469, better than gender classifier 0.76555

clf3 = SVC(gamma=0.05, C=0.75)
clf3.fit(train_dataset_onehot, train_label)
r_pred = clf3.predict(test_dataset_onehot)
r_predictions = [int(round(value)) for value in r_pred]

submission_df = pandas.DataFrame(index=test_dataset.index, columns=["Survived"])
submission_df["Survived"] = r_predictions
submission_df.to_csv("submission_best_svc.csv", sep=',')


# In[ ]:


# Submission score is 0.77512, better than gender classifier 0.76555

clf4 = LogisticRegression(C = 0.075)
clf4.fit(train_dataset_onehot, train_label)
r_pred = clf4.predict(test_dataset_onehot)
r_predictions = [int(round(value)) for value in r_pred]

submission_df = pandas.DataFrame(index=test_dataset.index, columns=["Survived"])
submission_df["Survived"] = r_predictions
submission_df.to_csv("submission_best_lr.csv", sep=',')


# In[ ]:


# Submission score is 0.81818, better than gender classifier 0.76555

from sklearn.ensemble import VotingClassifier

eclf2 = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2), ('svm', clf3), ('lr', clf4)], voting='hard')
eclf2.fit(train_dataset_onehot, train_label) 
r_pred = eclf2.predict(test_dataset_onehot)
r_predictions = [int(round(value)) for value in r_pred]

submission_df = pandas.DataFrame(index=test_dataset.index, columns=["Survived"])
submission_df["Survived"] = r_predictions
submission_df.to_csv("submission_best_voting.csv", sep=',')


# In[ ]:




