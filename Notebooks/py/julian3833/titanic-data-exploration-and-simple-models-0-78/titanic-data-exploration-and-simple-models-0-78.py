#!/usr/bin/env python
# coding: utf-8

# # 0. Introduction

# This notebook goes through:
# 1. A very quick data exploration and preprocessing, plotting some simple plots with seaborn and pandas (2 and 3) 
# 2. A simple RandomForestClassifier with 0.73 and a grid search over a RFC model which obtains a 0.78 score (3).
# 
# *It's currently under construction. *
# 
# 1. [Imports and data load](#1)
# 2. [Basic data exploration and nulls & categoricals handling](#2)
# 3. [Plots](#3)
# 4. [Simple RandomForest models (0.73 and 0.78)](#4)
# 5. [More advanced feature handling](#5) (0.76)
# 
# 
# Please, upvote it if you found it useful!

# <a id='1'></a>

# # 1. Imports and data load

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[ ]:


train = pd.read_csv("../input/train.csv", index_col='PassengerId')
test = pd.read_csv("../input/test.csv", index_col='PassengerId')
example = pd.read_csv("../input/gender_submission.csv")


# <a id='2'></a>

# # 2. Basic data exploration and nulls & categoricals handling

#  ## Check general information of the dataframes

# In[ ]:


print("Train dataframe shape: {}".format(train.shape))
train.head()


# In[ ]:


print("Test dataframe shape: {}".format(test.shape))
test.head()


# In[ ]:


print("Submission example dataframe shape: {}".format(example.shape))
example.head()


# In[ ]:


train.info()
train.describe()


# **Check survival rate**

# In[ ]:


train.Survived.value_counts()


# ## Check and handle nulls

# ### **Check for nulls**

# In[ ]:


train.isnull().sum()


# The train dataframe has three columns with nulls:
# 1. Age, which has 177 nulls (20% of the rows)
# 2. Cabin, which has 687 (77% of the rows)
# 3. Embarked, which has only 2 nulls

# In[ ]:


test.isnull().sum()


# The test dataframe has three columns with nulls too:
# 1. Age, with 86 (20% again)
# 2. Cabin, which has 327 (78%, again)
# 3. Fare, which has only 1

# ### ** Handle nulls **

# **Drop train rows with null in Embarked since they are just 2 **

# In[ ]:


train.dropna(subset=['Embarked'], inplace=True)


# **Impute the unique test row with Fare null since we can't drop it**

# In[ ]:


fare_imputer = Imputer()
fare_imputer.fit(train[['Fare']])
test['Fare'] = fare_imputer.transform(test[['Fare']]) # Impute it only for the test set

print("Fare mean: {:.2f}".format(fare_imputer.statistics_[0]))


# **Assign special value for null Cabins, since the information may be useful**

# In[ ]:


train.loc[train['Cabin'].isnull(), 'Cabin'] = 'U0'
test.loc[test['Cabin'].isnull(), 'Cabin'] = 'U0'


# **Impute null ages**

# In[ ]:


# Copies to use later on, with a different approach...
original_train_age = train['Age'].copy()
original_test_age = train['Age'].copy()

age_imputer = Imputer()  
train['Age'] = age_imputer.fit_transform(train[['Age']])
test['Age'] = age_imputer.transform(test[['Age']])
print("The mean known age: {}".format(age_imputer.statistics_[0]))


# **Check there are no more nulls left**

# In[ ]:


print(train.isnull().sum())
print(test.isnull().sum())


# ## Handle categorical data

# ** Check the amount of unique values per 'object' column**

# In[ ]:


full_df = pd.concat([train.drop("Survived", axis=1), test])
print("Full dataset contains {} rows".format(full_df.shape[0]))
for c in full_df.dtypes[full_df.dtypes == 'object'].index.tolist():
    print("Unique values for '{}': {}".format(c, full_df[c].nunique()))


# It seems like there are 2 repeated names :P ...

# In[ ]:


full_df.Name.value_counts()[full_df.Name.value_counts() > 1]


# **Sex can be label-encoded because it's binary**

# In[ ]:


sex_encoder = LabelEncoder()
train['Sex'] = sex_encoder.fit_transform(train.Sex)
test['Sex'] = sex_encoder.transform(test.Sex)


# **Embarked has 3 values, since it has no order, we one-hot-encode it**

# In[ ]:


embarked = pd.concat([train, test])['Embarked']
train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])


# <a id='3'><a>

# # 3. Plots

#  ## Correlations

# In[ ]:


f,ax = plt.subplots(figsize=(16, 10))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);


# ## Feature values distribution

# In[ ]:


fig = pd.concat([train, test])[['Age', 'Fare', 'Parch', 'SibSp']].hist(figsize=(14, 10), alpha=0.7, layout=(2, 2), color='r');
fig[1][1].set_title("# Siblings and Spouse");
fig[1][0].set_title("# Parents and Children");


# In[ ]:


sex_fig = pd.concat([train, test])['Sex'].value_counts().plot(figsize=(14, 7), alpha=0.7, color=['g', 'r'], kind='bar', width=1);
sex_fig.set_xticklabels(['Male', 'Female'])
sex_fig.set_title("Sex")
plt.show()

class_fig = pd.concat([train, test])['Pclass'].plot(figsize=(14, 7), alpha=0.7, kind='hist');
class_fig.set_xlim([.5, 3.5])
class_fig.set_xticks([1,2,3])
class_fig.set_title("Social class")
class_fig.set_xticklabels(['High', 'Middle', 'Low'])
plt.show()

encoder = LabelEncoder()
fig = pd.Series(encoder.fit_transform(embarked)).plot(kind='hist', figsize=(14, 7), alpha=0.7, color='g')
fig.set_title("Port of Embarkation")
fig.set_xticks([0, 1, 2])
fig.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton']);


# <a id='4'></a>

# # 4. Simple RandomForest models (0.73 and 0.78)

# ## Trivial model without scaling nor feature engineering: 0.73684

# In[ ]:


X = train.drop(["Survived", "Name", "Ticket", "Cabin"], axis=1)
X_delivery = test.drop(["Name", "Ticket", "Cabin"], axis=1)
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X.head()


# In[ ]:


model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Accuracy over train set: {:.2f}".format(model.score(X_train, y_train)))
print("Accuracy over test set: {:.2f}".format(model.score(X_test, y_test)))


# **Dump model to random_forest_no_cv.csv**

# In[ ]:


results = model.fit(X, y).predict(X_delivery)
pd.DataFrame(list(zip(X_delivery.index.tolist(), results)), columns=["PassengerId", "Survived"]).to_csv("random_classifier_no_cv.csv", index=False)


# ## Trivial RandomForest with Grid Search: 0.78468

# In[ ]:


# Uncomment this to run the grid search
# use a full grid over all parameters
# param_grid = {"max_depth": [3, None],
#               "max_features": [1, 3, 9],
#               "min_samples_split": [2, 3, 9],
#               "min_samples_leaf": [1, 3, 9],
#               "bootstrap": [True, False],
#               "n_estimators": [10, 100, 500],
#               "criterion": ["gini", "entropy"]}

# clf = RandomForestClassifier()
# grid_search = GridSearchCV(clf, param_grid=param_grid,  scoring='accuracy', verbose=10)
# grid_search.fit(X_train, y_train)
#grid_search.best_params_
#grid_search.best_score_ ==> 0.84234234234234229 
best_params = {'bootstrap': False,
             'criterion': 'entropy',
             'max_depth': None,
             'max_features': 3,
             'min_samples_leaf': 3,
             'min_samples_split': 9,
             'n_estimators': 500}


# In[ ]:


clf = RandomForestClassifier(**best_params)
clf.fit(X_train, y_train)
print("Accuracy over train set: {:.2f}".format(clf.score(X_train, y_train)))
print("Accuracy over test set: {:.2f}".format(clf.score(X_test, y_test)))


# ** Dump results to random_classifier_cv.csv **

# In[ ]:


results = clf.fit(X, y).predict(X_delivery)
pd.DataFrame(list(zip(X_delivery.index.tolist(), results)), columns=["PassengerId", "Survived"]).to_csv("random_classifier_cv.csv", index=False)


# <a id='5'></a>

# # 5. More advanced feature handling

# ## More complex treatment of Age field: learn it from other fields
# Idea taken from here: 
# http://www.ultravioletanalytics.com/2014/11/03/kaggle-titanic-competition-part-ii-missing-values/

# In[ ]:


### Populate missing ages using RandomForestClassifier
def set_missing_ages(train, test):
    
    # Grab all the features that can be included in a Random Forest Regressor
    df = pd.concat([train, test])
    age_df = df[['Age','Pclass','Fare', 'Parch', 'SibSp', 'Sex',  'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    
    train_age = train[['Age','Pclass','Fare', 'Parch', 'SibSp', 'Sex',  'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    test_age = test[['Age','Pclass','Fare', 'Parch', 'SibSp', 'Sex',  'Embarked_C', 'Embarked_Q', 'Embarked_S']]

    # Split into sets with known and unknown Age values
    known_age = age_df.loc[ (df.Age.notnull()) ]
    unknown_age = age_df.loc[ (df.Age.isnull()) ]
    
    # All age values are stored in a target array
    y = known_age.values[:, 0]
    
    # All the other values are stored in the feature array
    X = known_age.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    #rtr.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    #rtr.fit(X, y)
    rtr.fit(X_train, y_train)

    print(rtr.score(X_train, y_train))
    print(rtr.score(X_test, y_test))
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predicted_ages_train = rtr.predict(train_age[train_age.Age.isnull()].values[:, 1::])
    predicted_ages_test = rtr.predict(test_age[test_age.Age.isnull()].values[:, 1::])
    
    # Assign those predictions to the full data set
    #df.loc[ (df.Age.isnull()), 'Age' ] = predicted_ages 
    
    train.loc[train.Age.isnull(), 'Age'] = predicted_ages_train
    test.loc[test.Age.isnull(), 'Age'] = predicted_ages_test
    
    return train, test


# In[ ]:


train['Age'] = original_train_age
test['Age'] = original_test_age
train.Age.isnull().sum()


# In[ ]:


train, test = set_missing_ages(train, test)
train.Age.isnull().sum()


# ## Use the RFC with previously grid-searched params with new Age feature. Wasn't an improvement: 0.76

# In[ ]:


X = train.drop(["Survived", "Name", "Ticket", "Cabin"], axis=1)
X_delivery = test.drop(["Name", "Ticket", "Cabin"], axis=1)
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X.head()


# In[ ]:


clf = RandomForestClassifier(**best_params)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# In[ ]:


results = clf.fit(X, y).predict(X_delivery)
pd.DataFrame(list(zip(X_delivery.index.tolist(), results)), columns=["PassengerId", "Survived"]).to_csv("random_classifier_predicted_age.csv", index=False)


# ## Try with new grid search with new Age feature: same as previous one: same, 0.76

# In[ ]:


# Uncomment this to run the grid search
# use a full grid over all parameters
# param_grid = {"max_depth": [3, None],
#               "max_features": [1, 3, 9],
#               "min_samples_split": [2, 3, 9],
#               "min_samples_leaf": [1, 3, 9],
#               "bootstrap": [True, False],
#               "n_estimators": [10, 100, 500],
#               "criterion": ["gini", "entropy"]}

# clf = RandomForestClassifier()
# grid_search = GridSearchCV(clf, param_grid=param_grid,  scoring='accuracy', verbose=10)
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_) # ==>0.843843843844
best_params = {'bootstrap': False,
 'criterion': 'entropy',
 'max_depth': None,
 'max_features': 3,
 'min_samples_leaf': 3,
 'min_samples_split': 2, # This changes from 9 to 2
 'n_estimators': 500}


# In[ ]:


clf_new = RandomForestClassifier(**best_params)
clf_new.fit(X_train, y_train)
clf_new.score(X_test, y_test)


# In[ ]:


results = clf_new.fit(X, y).predict(X_delivery)
pd.DataFrame(list(zip(X_delivery.index.tolist(), results)), columns=["PassengerId", "Survived"]).to_csv("random_classifier_predicted_age_re_cv.csv", index=False)

