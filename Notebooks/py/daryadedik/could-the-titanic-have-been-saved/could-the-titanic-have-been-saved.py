#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import collections as cln

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


survived_color = '#6699ff'
died_color = '#ff6666'

na_string = 'NA'
na_number = -1
width = 0.35
embarked_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown', na_string: 'N/A'}
pclass_map = {1: 'First class', 2: 'Second class', 3: 'Third class'}


# In[ ]:


def ensure_na(d):
    if not na_string in d:
        d[na_string] = 0
    return d


# In[ ]:


titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
idx_survived = titanic_df['Survived'] == 1
idx_died = np.logical_not(idx_survived)

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)

titanic_df["Embarked"] = titanic_df["Embarked"].fillna(na_string)
titanic_df["Fare"] = titanic_df["Fare"].fillna(na_number)
titanic_df["Age"] = titanic_df["Age"].fillna(na_number)


# In[ ]:


# Embarked

survived_embarked_counts = ensure_na(titanic_df[idx_survived].Embarked.value_counts())
died_embarked_counts = ensure_na(titanic_df[idx_died].Embarked.value_counts())
print(survived_embarked_counts)
print(died_embarked_counts)
assert(len(survived_embarked_counts) == len(died_embarked_counts))

N = len(survived_embarked_counts)
ind = np.arange(N) 
plot1 = plt.bar(ind, survived_embarked_counts, width, color=survived_color)
plot2 = plt.bar(ind + width, died_embarked_counts, width, color=died_color)

plt.ylabel('Number of people')
plt.xlabel('Port of Embarkation')
plt.xticks(ind + width, (embarked_map[k] for k in survived_embarked_counts.keys()))
plt.legend((plot1[0], plot2[0]), ('survived', 'died'))
plt.show()


# In[ ]:


# Embarked train/test get dummies

embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)

titanic_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)


# In[ ]:


# Fare
fare_survived = titanic_df[idx_survived].Fare
fare_died = titanic_df[idx_died].Fare

minFare, maxFare = min(titanic_df.Fare), max(titanic_df.Fare)
bins = np.linspace(minFare, maxFare, 25)

fare_survived_counts, _ = np.histogram(fare_survived, bins)
fare_died_counts, _ = np.histogram(fare_died, bins)

plt.figure()
plt.bar(bins[:-1], np.log10(fare_survived_counts), width=20, color=survived_color, label='Survived')
plt.bar(bins[:-1], -np.log10(fare_died_counts), width=20, color=died_color, label='Died')
plt.ylabel('Number of people')
plt.xlabel('Ticket fare')
plt.yticks(range(-3,4), (10**abs(k) for k in range(-3,4)))
plt.legend()
plt.show()


# In[ ]:


# Pclass: 1 = 1st; 2 = 2nd; 3 = 3rd

pclass_survived = titanic_df[idx_survived].Pclass
pclass_died = titanic_df[idx_died].Pclass
pclass_survived_counts = ensure_na(titanic_df[idx_survived].Pclass.value_counts())
pclass_died_counts = ensure_na(titanic_df[idx_died].Pclass.value_counts())

# we get no NA values fro Pclass feature
# so we remove NA from plots and sort the rest of values by index
pclass_survived_sorted = pclass_survived_counts[0:3].sort_index()
pclass_died_sorted = pclass_died_counts[0:3].sort_index()

N = len(pclass_survived_sorted)
ind = np.arange(N)

plot1 = plt.bar(ind, pclass_survived_sorted, width, color=survived_color, label='Survived')
plot2 = plt.bar(ind + width, pclass_died_sorted, width, color=died_color, label='Died')

plt.xlabel('Passenger Classes', fontsize=18)
plt.ylabel('Number of people', fontsize=16)
plt.legend(loc='upper center')
plt.xticks(ind + width, (pclass_map[l] for l in pclass_survived_sorted.keys()))
plt.show()


# In[ ]:


# make dummies from Pclass feature

pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)

titanic_df.drop(['Pclass'], axis=1, inplace=True)
test_df.drop(['Pclass'], axis=1, inplace=True)


# In[ ]:


# Age
age_survived = titanic_df[idx_survived].Age
age_died = titanic_df[idx_died].Age

minAge, maxAge = min(titanic_df.Age), max(titanic_df.Age)
bins = np.linspace(minAge, maxAge, 100)

age_survived_counts, _ = np.histogram(age_survived, bins)
age_died_counts, _ = np.histogram(age_died, bins)

plt.bar(bins[:-1], np.log10(age_survived_counts), color=survived_color, label='Survived')
plt.bar(bins[:-1], -np.log10(age_died_counts), color=died_color, label='Died')
plt.yticks(range(-3,4), (10**abs(k) for k in range(-3,4)))
plt.legend(loc='upper right')
plt.xlabel('Age', fontsize=18)
plt.ylabel('Number of people', fontsize=16)
plt.show()


# In[ ]:


# New feature for Age
# 0/1 - depending on whether Age exists or not

titanic_df['AgeExists'] = titanic_df['Age']

#titanic_df[titanic_df['Age'] > -1]['AgeExists'] = 1.0
#titanic_df[titanic_df['Age'] == -1]['AgeExists'] = 0.0

titanic_df.AgeExists.loc[titanic_df['AgeExists'] > -1] = 1.0

#titanic_df.AgeExists[titanic_df.AgeExists > -1] = 1.0
#titanic_df.AgeExists[titanic_df.AgeExists == -1] = 0.0
print(titanic_df)

