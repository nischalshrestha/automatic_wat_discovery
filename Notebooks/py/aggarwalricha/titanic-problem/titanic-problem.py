#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv", sep=',', usecols=['PassengerId','Pclass','Sex','Age', 'Survived'])
train['Age'] = train['Age'].apply(lambda x: 1 if x < 15 else 0)
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'female' else 0)

from sklearn import svm 
from sklearn.model_selection import GridSearchCV
X = train[['Pclass','Sex','Age',]]
Y = train['Survived']

Cs =[0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
kernels = ['linear', 'rbf']
paramGrid = [{'C': Cs , 'gamma': gammas, 'kernel':['rbf']}, {'C': Cs ,'kernel':['linear']}]

gridSearch = GridSearchCV(svm.SVC(), paramGrid, cv=4 )
gridSearch.fit(X, Y)

#print(gridSearch.best_params_)
clf = svm.SVC(C=1, gamma = 0.1, kernel='rbf')
clf.fit(X, Y)
#print('training score : ', clf.score(X,Y))

test = pd.read_csv("../input/test.csv", sep= ',', usecols=['PassengerId','Pclass','Sex','Age'])

test['Age'] = test['Age'].apply(lambda x: 1 if x < 15 else 0)
test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'female' else 0)

X_test = test[['Pclass','Sex','Age',]]
Y_pred = clf.predict(X_test)

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': Y_pred})
print(output.head())
print(output.shape)
output.to_csv("gender_submission.csv")


# In[ ]:




