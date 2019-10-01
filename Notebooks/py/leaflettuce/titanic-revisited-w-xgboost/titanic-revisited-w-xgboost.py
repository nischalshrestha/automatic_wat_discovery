#!/usr/bin/env python
# coding: utf-8

# # Titanic Revisited (w/ XGBoost)
# 
# ### Goal
# I originally worked with this dataset about 2.5 years ago when working through Udacity's Nanodegree in Data Analytics. I has, more or less, no idea what I was doing then. I thought it would be nice to revisit this dataset and see if I could get a better accuracy than my first time through (which was around 74% if I recall). Particularly, I've been wanting to work with XGBoost for awhile and this seems an appropriate classification problem to give it a go on!

# In[ ]:



import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import math
import xgboost

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/gender_submission.csv')

train.head()


# ## Feature Generation and Removal
# 
# Time to generate a few features which may be of use in classification.

# In[ ]:


#Does the passanger have a cabin?
train['cabin_binary'] = train["Cabin"].apply(lambda i: 0 if str(i) == "nan" else 1)

#Family Size
train['family_size'] = 1 + train['SibSp'] + train['Parch']
train['solo'] = train["family_size"].apply(lambda i: 1 if i == 1 else 0)

#Fix Nulls
train['Embarked'] = train['Embarked'].fillna('S')
train['Age'] = train['Age'].fillna(int(np.mean(train['Age'])))

#A few age specific Binaries
train['Child'] = train["Age"].apply(lambda i: 1 if i <= 17 and i > 6 else 0)
train['toddler'] = train["Age"].apply(lambda i: 1 if i <= 6 else 0)
train['Elderly'] = train["Age"].apply(lambda i: 1 if i >= 60 else 0)

# Fancy fancy
train['fancy'] = train['Fare'].apply(lambda i: 1 if i >= 100 else 0)

# standard
train['standard_fare'] = train['Fare'].apply(lambda i: 1 if i <= 10.0 else 0)

#No requirement to standardize in DT models, but might as well
fare_scaler = StandardScaler()
fare_scaler.fit(train['Fare'].values.reshape(-1, 1))
train['fare_std'] = fare_scaler.transform(train['Fare'].values.reshape(-1, 1))

#get status of passanger
train['title'] = 'default'

for i in train.values:
    name = i[3] #First checks for rare titles (Thanks Anisotropic's wonderful Kernel for inspiration//help here!)
    for e in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        if e in name:
            train.loc[train['Name'] == name, 'title'] = 'rare'
    if 'Miss' in name or  'Mlle' in name or 'Ms' in name or 'Mme' in name or 'Mrs' in name:
        train.loc[train['Name'] == name, 'title'] = 'Ms'
    if 'Mr.' in name or 'Master' in name:
        train.loc[train['Name'] == name, 'title'] = 'Mr'


train.head(10)


# Lets send the test data through the same pipeline!

# In[ ]:


#Does the passanger have a cabin?
test['cabin_binary'] = test["Cabin"].apply(lambda i: 0 if str(i) == "nan" else 1)

#Family Size
test['family_size'] = 1 + test['SibSp'] + test['Parch']
test['solo'] = test["family_size"].apply(lambda i: 1 if i == 1 else 0)

#Fix Nulls
test['Embarked'] = test['Embarked'].fillna('S')
test['Age'] = test['Age'].fillna(int(np.mean(test['Age'])))

#A few age specific Binaries
test['Child'] = test["Age"].apply(lambda i: 1 if i <= 17 and i > 6 else 0)
test['toddler'] = test["Age"].apply(lambda i: 1 if i <= 6 else 0)
test['Elderly'] = test["Age"].apply(lambda i: 1 if i >= 60 else 0)

# Fancy fancy
test['fancy'] = test['Fare'].apply(lambda i: 1 if i >= 100 else 0)
test['standard_fare'] = test['Fare'].apply(lambda i: 1 if i <= 10.0 else 0)

#standardize
test['fare_std'] = fare_scaler.transform(test['Fare'].values.reshape(-1, 1))

#get status of passanger
test['title'] = 'default'

for i in test.values:
    name = i[2] #First checks for rare titles (Thanks Anisotropic's wonderful Kernel for inspiration//help here!)
    for e in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        if e in name:
            test.loc[test['Name'] == name, 'title'] = 'rare'
    if 'Miss' in name or  'Mlle' in name or 'Ms' in name or 'Mme' in name or 'Mrs' in name:
        test.loc[test['Name'] == name, 'title'] = 'Ms'
    if 'Mr.' in name or 'Master' in name:
        test.loc[test['Name'] == name, 'title'] = 'Mr'


test.head(10)


#  Remove Unneccesary Features and Encode Categorical

# In[ ]:


train = pd.get_dummies(train, columns=["Sex", "Embarked", "title"])
test = pd.get_dummies(test, columns=["Sex", "Embarked", "title"])

train = train.drop(['Name','PassengerId', 'Ticket', 'Cabin', 'Fare', 'SibSp'], axis = 1)
test = test.drop(['Name','PassengerId', 'Ticket', 'Cabin', 'Fare', 'SibSp'], axis = 1)

train.head()


# ## Quick EDA Visuals
# Lets do just a little bit of EDA with Seaborn:

# In[ ]:


#COR MATRIX OF numerical vars
plt.figure(figsize=(14,12))
plt.title('Corelation Matrix', size=12)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)


# In[ ]:


#Family size histo
sns.distplot(train['family_size'])


# In[ ]:


#boxplot of family size and survival
sns.boxplot("Survived", y="family_size", data = train)


# In[ ]:


#Fare to Age relationship?
sns.lmplot('fare_std', 'Age', data = train, 
           fit_reg=False,scatter_kws={"marker": "D", "s": 20}) 


# In[ ]:


#boxplot of age and survival
sns.boxplot("Survived", "Age", data = train)


# In[ ]:


#bar chart of age and survival
sns.lmplot("Survived", "fare_std", data = train, fit_reg=False)


# ## Classifing with XGBoost
# 
# I'll be comparing XG to AdaBoost, GradientBoost, RandomForest, andddd maybe a SVC or something else.
# 
# First off to split the train data into a train/test for cross validation testing

# In[ ]:


X = train.drop(['Survived'], axis = 1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Alright, first letsfirst try some more traditional models.. AKA: Random Forest and Standard Tree
# 

# In[ ]:


#Random Forest Setup
ranfor = RandomForestClassifier()
parameters = {'n_estimators':[10,50,100], 'random_state': [42, 138],               'max_features': ['auto', 'log2', 'sqrt']}
ranfor_clf = GridSearchCV(ranfor, parameters)
ranfor_clf.fit(X_train, y_train)


'''CROSS VALIDATE'''
cv_results = cross_validate(ranfor_clf, X_train, y_train)
cv_results['test_score']  

y_pred = ranfor_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


# In[ ]:


##Decision Tree Go
dt = DecisionTreeClassifier()
parameters = {'random_state': [42, 138],'max_features': ['auto', 'log2', 'sqrt']}
dt_clf = GridSearchCV(dt, parameters)
dt_clf.fit(X_train, y_train)

y_pred = dt_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


# At 81% with a random forest.. I'm stoked as this is already better than my last attempt.  Lets keep pushing on with it and give some boosting models  a go.

# In[ ]:


ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())
parameters = {'n_estimators':[10,50,100], 'random_state': [42, 138], 'learning_rate': [0.1, 0.5, 0.8, 1.0]}
ada_clf = GridSearchCV(ada, parameters)
ada_clf.fit(X_train, y_train)

cv_results = cross_validate(ada_clf, X_train, y_train)
cv_results['test_score']  

y_pred = ada_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


# In[ ]:


gradBoost = GradientBoostingClassifier()
parameters = {'n_estimators':[10,50,100], 'random_state': [42, 138], 'learning_rate': [0.1, 0.5, 0.8, 1.0],              'loss' : ['deviance', 'exponential']}
gb_clf = GridSearchCV(gradBoost, parameters)
gb_clf.fit(X_train, y_train)

cv_results = cross_validate(gb_clf, X_train, y_train)
cv_results['test_score']  

y_pred = gb_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


# ![](http://)So GradientBoosting has given me the best score at 82% so far..  lets finally make our way to XG:

# In[ ]:


xg = xgboost.XGBClassifier(max_depth = 3, n_estimators = 600, learning_rate = 0.05)
xg.fit(X_train, y_train)

cv_results = cross_validate(xg, X_train, y_train)
cv_results['test_score']  

y_pred = xg.predict(X_test)
print(accuracy_score(y_test, y_pred))


# In[ ]:


'''Confusion Matrix'''
y_pred = xg.predict(X_test)
# TN, FP, FN, TP
confusion_matrix(y_test, y_pred)


# So.. A liiiittttleee better than the Gradient boost. I'm happy with it for a quick project like this.  Lets write it out and submit.

# In[ ]:


predictions = xg.predict(test)

sub['Survived'] = predictions
sub.to_csv("first_submission_xgb.csv", index=False)

