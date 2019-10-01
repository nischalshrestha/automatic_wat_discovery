#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


########################## READ THIS arthurtok/introduction-to-ensembling-stacking-in-python


# In[ ]:


########################## LOAD DATA
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
KeyId = test['PassengerId']
train.head(3)


# In[ ]:


############################ MODIFY DATA
import re
full_data = [train, test]
train['NameLength'] = train['Name'].apply(len)
test['NameLength'] = test['Name'].apply(len)
train['HasCabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['HasCabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)

X_train = train.dropna()[["Pclass","Age", "SibSp", "Parch", "Fare", "Sex", "Embarked", "NameLength", "HasCabin", "FamilySize", "Title"]]
Y_train = train.dropna()["Survived"]

train.head(3)


# In[ ]:


############################ OPTIMIZE MODEL
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from time import time

alg = RandomForestClassifier()
alg.get_params()

param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 7],
              "min_samples_leaf": [1, 2, 3, 10],
              "min_samples_split": [2, 3, 4, 10],
              "bootstrap": [True, False],
              "criterion": ["gini"]}

grid_search = GridSearchCV(alg, param_grid=param_grid)
start = time()
grid_search.fit(X_train, Y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))

print(grid_search.best_params_)
print(grid_search.best_score_)


# In[ ]:


############################ CLASSIFY DATA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

alg = RandomForestClassifier(
    random_state=1,
    n_estimators=150,
    min_samples_split=4,
    min_samples_leaf=2
)

alg.fit(X_train, Y_train)
print(alg.score(X_train, Y_train))

scores = cross_val_score(
    alg,    X_train,     Y_train,      cv=5
)
print(scores, scores.mean())
len(X_train)


# In[ ]:


############################ MAKE SUBMISSIONS
X_test = test.dropna()[["Pclass","Age", "SibSp", "Parch", "Fare", "Sex", "Embarked", "NameLength", "HasCabin", "FamilySize", "Title"]]
predictions = alg.predict(X_test)
submission = pd.DataFrame({ 'PassengerId': KeyId,
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)
submission.head(3)

