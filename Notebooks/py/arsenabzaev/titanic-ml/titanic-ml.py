#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import read_csv, DataFrame, Series
data = read_csv('../input/train.csv')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import pylab as pl
from sklearn import tree

data.Age[data.Age.isnull()] = data.Age.mean()
MaxPassEmbarked = data.groupby('Embarked').count()['PassengerId'] # число пассажиров с определенного порта
data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0] # присваиваем букву самого популярного порта (index[0] - буква)

data = data.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked', 'Parch', 'SibSp'],axis=1)

label = LabelEncoder()
dicts = {}

label.fit(data.Sex.drop_duplicates())
dicts['Sex'] = list(label.classes_)
data.Sex = label.transform(data.Sex)


test = read_csv('../input/test.csv')
test.Age[test.Age.isnull()] = test.Age.mean()
test.Fare[test.Fare.isnull()] = test.Fare.median()
MaxPassEmbarked = test.groupby('Embarked').count()['PassengerId']
test.Embarked[test.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
result = DataFrame(test.PassengerId)
result1 = DataFrame(test.PassengerId)
test = test.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked', 'Parch', 'SibSp'],axis=1)

label.fit(dicts['Sex'])
test.Sex = label.transform(test.Sex)


target = data.Survived
train = data.drop(['Survived'], axis=1)

ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.25)

model_rfc = RandomForestClassifier(n_estimators = 80, max_features = 'auto', criterion = 'entropy', max_depth = 4) #в параметре передаем кол-во деревьев

model_rfc.fit(train, target)
result.insert(1,'Survived', model_rfc.predict(test))
result.to_csv('RandomFor2.csv', index=False)


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

