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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import re as re
import pandas as pd
import numpy as np
from scipy.stats import norm
import os as os
from sklearn import preprocessing
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# In[ ]:


# Data cleanup
# TRAIN DATA
train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv', dtype={"Age": np.float64}, )
test = pd.read_csv('../input/train.csv', dtype={"Age": np.float64}, )
le = preprocessing.LabelEncoder()


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def get_lastname(name):
    title_search = re.search('(?:Mrs\.|Major\.|Mr\.|Miss\.|Master\.|Dr\.|Lady\.|Countess\.|Don\.|Rev\.|'
                             'Jonkheer\.|Dona\.|Mme\.) (\(([A-Za-z0-9_]+)\)|[a-zA-Z]+)', name)

    # If the title exists, extract and return it.
    if title_search:
        final_result = re.sub('(?:Mr\.|Major\.|Mrs\.|Miss\.|Master\.|Dr\.|Lady\.|Countess\.|Don\.|'
                              'Rev\.|Jonkheer\.|Dona\.|Mme\.| )', '', title_search.group(0))
        return final_result
    return ""


def df_freq_table(df, var):
    df_freq = pd.value_counts(df[var]).to_frame().reset_index()
    df_freq.columns = [var, 'Count']
    return df_freq


def my_func(b):
    return max(0, b)

vmy_func = np.vectorize(my_func)

train['data_label'] = 'train'
test['data_label'] = 'test'
test['Survived'] = -1

all_data = train.append(test)

all_age = all_data['Age'].loc[~pd.isnull(all_data['Age'])]
mu, std = norm.fit(all_age)

all_names = all_data.loc[~pd.isnull(all_data['Name']), ['PassengerId', 'Name']]
all_names['LastName'] = ''
all_names.loc[:, 'LastName'] = all_names.loc[:, 'Name'].apply(get_lastname)
all_names.loc[all_names['LastName'] == '', 'LastName'] = 'MISC'
lastname_df = df_freq_table(all_names, 'LastName')
lastname_dict = lastname_df.set_index('LastName')['Count'].to_dict()

SEED = 0
np.random.seed(5)
full_data = [train, test]


def get_cabin_type(cabin):
    if pd.isnull(cabin):
        return 'Z'
    else:
        letter_list = " ".join(re.findall("[a-zA-Z]+", cabin))
        unique_cabin_letter = list(set(letter_list))
        unique_cabin_letter = [x for x in unique_cabin_letter if x != ' ']
        if unique_cabin_letter.__len__() != 1:
            unique_cabin_letter.sort()
            print('multiple cabin for one passenger, only keeps the higher cabin')
            return unique_cabin_letter[0]
        else:
            return unique_cabin_letter[0]

# recode cabin type
all_data['Cabin_type'] = all_data['Cabin'].apply(get_cabin_type)
all_data['Cabin_type'] = all_data['Cabin_type'].map(lambda x: ord(x) - 64 if x.isalpha() else x)

flag_P1_NA = ((all_data['Cabin_type'] == 26) & (all_data['Pclass'] == 1))
P1_NA_Size = pd.value_counts(all_data['Cabin_type'].loc[flag_P1_NA]).values
all_data.loc[flag_P1_NA, 'Cabin_type'] = np.random.randint(0, 5, P1_NA_Size)

# Create Family Size
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

# Create Flag isalone
all_data['IsAlone'] = 0
all_data.loc[all_data['FamilySize'] == 1, 'IsAlone'] = 1

# filling empty embarked
all_data['Embarked'] = all_data['Embarked'].fillna('S')

# filling empty Fare
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())

#  ID
all_data['Title'] = ''
all_data.loc[:, 'Title'] = all_data['Name'].apply(get_title)
all_data['LastName'] = ""
all_data.loc[:, 'LastName'] = all_data.loc[:, 'Name'].apply(get_lastname)
all_data.loc[all_data['LastName'] == '', 'LastName'] = 'MISC'
all_data['LastNameFreq'] = all_data['LastName'].map(lastname_dict)
all_data['FamilyID'] = ''
all_data.loc[:, 'FamilyID'] = all_data.apply(lambda row: (str(row['FamilySize']) + row['LastName']), axis=1)
all_data.loc[all_data['FamilySize'] < 3, 'FamilyID'] = 'Small'
le.fit(all_data['FamilyID'].values)
all_data.loc[:, 'FamilyID_Code'] = le.transform(all_data['FamilyID'].values)

# mapping tile to code
all_data['Title'] = all_data['Title'].replace(
    ['Lady', 'Countess', 'Don', 'Dr', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Noble')
all_data['Title'] = all_data['Title'].replace(
    ['Capt', 'Col', 'Major'], 'Military')
all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
all_data.loc[(np.isnan(all_data['Age'])) & (all_data['Title'] == 'Master'), 'Age'] = 8
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 6, "Military": 5, "Noble": 4}
all_data['Title'] = all_data['Title'].map(title_mapping)
all_data['Title'] = all_data['Title'].fillna(0)

# code Gender
all_data['Sex'] = all_data['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Mapping Embarked
all_data['Embarked'] = all_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Mapping Fare
all_data.loc[all_data['Fare'] <= 7.91, 'Fare'] = 0
all_data.loc[(all_data['Fare'] > 7.91) & (all_data['Fare'] <= 14.454), 'Fare'] = 1
all_data.loc[(all_data['Fare'] > 14.454) & (all_data['Fare'] <= 31), 'Fare'] = 2
all_data.loc[all_data['Fare'] > 31, 'Fare'] = 3
all_data['Fare'] = all_data['Fare'].astype(int)

# Mapping Age
age_null_count = all_data['Age'].isnull().sum()
#age_null_random_list = vmy_func(np.random.normal(mu, std, age_null_count))
age_null_random_list = np.random.randint(mu - std, mu + std, size=age_null_count)
all_data.loc[np.isnan(all_data['Age']), 'Age'] = age_null_random_list
all_data['Age'] = all_data['Age'].astype(int)
all_data.loc[all_data['Age'] <= 16, 'Age'] = 0
all_data.loc[(all_data['Age'] > 16) & (all_data['Age'] <= 24), 'Age'] = 1
all_data.loc[(all_data['Age'] > 24) & (all_data['Age'] <= 28), 'Age'] = 2
all_data.loc[(all_data['Age'] > 28) & (all_data['Age'] <= 32), 'Age'] = 3
all_data.loc[(all_data['Age'] > 32) & (all_data['Age'] <= 48), 'Age'] = 4
all_data.loc[(all_data['Age'] > 48) & (all_data['Age'] <= 64), 'Age'] = 5
all_data.loc[all_data['Age'] > 64, 'Age'] = 6

# save all output to file
test_clean = all_data.loc[all_data['data_label'] == 'test', :]
train_clean = all_data.loc[all_data['data_label'] == 'train', :]

#var_drop_list = ['LastNameFreq', 'LastName', 'IsAlone', 'FamilyID', 'Embarked', 'Parch']
var_drop_list = ['LastName', 'FamilyID']
test_clean = test_clean.drop(['data_label', 'Survived', 'Name', 'Ticket', 'Cabin'] + var_drop_list, axis=1)
train_clean = train_clean.drop(['data_label', 'Name', 'Ticket', 'Cabin'] + var_drop_list, axis=1)


# In[ ]:


# import numpy as np
# import pandas as pd
# import numpy as np
# import re as re

# train_set = pd.read_csv('../input/train.csv')
# test_set = pd.read_csv('../input/test.csv')
# full_data = [train_set, test_set]

# train_set.info()
# for dataset in full_data:
#     dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1
# print(train_set[['Family_Size', 'Survived']].groupby(['Family_Size'], as_index=False).mean())
# for dataset in full_data:
#     dataset['isAlone'] = 0
#     dataset.loc[dataset['Family_Size'] == 1, 'isAlone'] = 1
# print(train_set[['isAlone', 'Survived']].groupby(['isAlone'], as_index=False).mean())
# for dataset in full_data:
#     dataset['Embarked'].fillna('S')
# print(train_set[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
# train_set['Fare'].fillna(dataset['Fare'].median())
# train_set['Cat_Fare'] = pd.qcut(dataset['Fare'], 4)
# print(train_set[['Cat_Fare', 'Survived']].groupby(['Cat_Fare'], as_index=False).mean())
# for dataset in full_data:
#     mean = dataset['Age'].mean()
#     std = dataset['Age'].std()
#     null_count = dataset['Age'].isnull().sum()

#     age_null_list = np.random.randint(mean - std, mean + std, size=null_count)
#     dataset['Age'][np.isnan(dataset['Age'])] = age_null_list
#     dataset['Age'] = dataset['Age'].astype(int)

# train_set['CatAge'] = pd.qcut(train_set['Age'], 5)
# print(train_set[['CatAge', 'Survived']].groupby(['CatAge'], as_index=False).mean())


# def get_title(name):
#     search_t = re.search(' ([A-Za-z]+)\.', name)
#     if (search_t):
#         return search_t.group(1)
#     return ""


# for dataset in full_data:
#     dataset['Title'] = dataset['Name'].apply(get_title)
# print(pd.crosstab(train_set['Title'], train_set['Sex']))
# for dataset in full_data:
#     dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
#                                                  'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
#     dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
#     dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
#     dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print(train_set[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
# for dataset in full_data:
#     # Mapping Sex
#     dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
#     title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
#     dataset['Title'] = dataset['Title'].map(title_mapping)
#     dataset['Title'] = dataset['Title'].fillna(0)

#     # Mapping Embarked
#     dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0).astype(int)

#     # Mapping Fare
#     dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
#     dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
#     dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
#     dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
#     dataset['Fare'] = dataset['Fare'].fillna(0).astype(int)

#     # Mapping Age
#     dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
#     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
#     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
#     dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# # Feature Selection
# drop_elements = ['Name', 'SibSp', 'Ticket', 'Cabin', 'Parch', 'Family_Size']
# train_set = train_set.drop(drop_elements, axis=1)
# train_set = train_set.drop(['PassengerId'], axis=1)
# train_set = train_set.drop(['CatAge', 'Cat_Fare'], axis=1)

# test_set = test_set.drop(drop_elements, axis=1)

# print(train_set.head(10))
# print(test_set.head(10))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

train_clean = train_clean.drop(['PassengerId'], axis=1)
train = train_clean.values
test = test_clean.drop(['PassengerId'], axis=1).values


# In[ ]:


best_param = {'min_samples_split': 2,
              'criterion': 'gini',
              'max_depth': 4,
              'oob_score': False,
              'min_weight_fraction_leaf': 0.0,
              'verbose': 0,
              'max_features': 'sqrt',
              'n_estimators': 500,
              'max_leaf_nodes': None,
              'n_jobs': 1,
              'min_impurity_split': None,
              'min_impurity_decrease': 0.0,
              'bootstrap': True,
              'class_weight': None,
              'warm_start': False,
              'random_state': 55,
              'min_samples_leaf': 2}

RandomForestclf = RandomForestClassifier(**best_param)
RandomForestclf.fit(train[0::, 1::], train[0::, 0])
y_result = RandomForestclf.predict(test)
submission = pd.DataFrame({
    "PassengerId": test_set["PassengerId"],
    "Survived": y_result
})


# In[ ]:


submission.to_csv('submission_RF.csv', index=False)

