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


dataTrain = pd.read_csv("../input/train.csv").drop('Ticket',axis=1)
dataTest = pd.read_csv("../input/test.csv").drop('Ticket', axis=1)


# In[ ]:


len(dataTrain), len(dataTest)


# In[ ]:


''' How many NaN Age exist '''
len(dataTrain[dataTrain.Age.isnull()]), len(dataTest[dataTest.Age.isnull()])


# In[ ]:


''' filling NaN with mean in train and test sets '''
dataTrain.Age.fillna(dataTrain.Age.mean(), inplace = True)
dataTest.Age.fillna(dataTest.Age.mean(), inplace = True)


# In[ ]:


''' we check how many nan Cabin values exist'''
len(dataTrain[dataTrain.Cabin.isnull()])


# In[ ]:


''' we delete Cabin feature'''
dataTrain.drop('Cabin', axis = 1, inplace = True)
dataTest.drop('Cabin', axis = 1, inplace = True)


# In[ ]:


dataTrain.head()


# In[ ]:


''' we divide the Age feature in quantiles'''
cutted= pd.qcut(dataTrain.Age.values, [0, 0.20, 0.4, 0.6, 0.8, 1.])
pd.value_counts(cutted, sort = False)


# In[ ]:


dataTrain['Age Quantili'] = cutted
dataTest['Age Quantili'] = pd.cut(dataTest.Age, [0, 20, 28, 29.699, 38, 80 ])


# In[ ]:


''' we delete also Name feature '''

dataTrain.drop('Name', axis = 1, inplace = True)
dataTest.drop('Name', axis = 1, inplace = True)


# In[ ]:


dataTrain.head()


# In[ ]:


dataTrain['Family number'] = dataTrain.SibSp + dataTrain.Parch
dataTest['Family number'] = dataTest.SibSp + dataTest.Parch


# In[ ]:


dataTest['Fare'].fillna(dataTest['Fare'].mean(), inplace = True)


# In[ ]:


fare_qbin= pd.qcut(dataTrain.Fare.values,5)


# In[ ]:


fare_qbin.value_counts()


# In[ ]:


dataTrain['Fare q_bins'] = fare_qbin
dataTest['Fare q_bins'] = pd.cut(dataTest.Fare, [-0.001, 7.854, 10.5, 21.679, 39.688, 520 ])


# In[ ]:


dataTrain = pd.get_dummies(dataTrain, columns=['Sex','Embarked'], drop_first = True)
dataTest = pd.get_dummies(dataTest, columns=['Sex','Embarked'], drop_first = True)


# In[ ]:


dataTest.head()


# In[ ]:


dataTrain.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()
dataTrain['Fare bins_dummies'] = le.fit_transform(dataTrain['Fare q_bins'])


# In[ ]:


dataTrain.head()


# In[ ]:


''' check out the correlation between Pclass and Fare bins_dummies'''

dataTrain['Pclass'].corr(dataTrain['Fare bins_dummies'])


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


le = LabelEncoder()
dataTest['Fare bins_dummies'] = le.fit_transform(dataTest['Fare q_bins'])


# In[ ]:


dataTest.head()


# In[ ]:


dataTrain.head()


# In[ ]:


le = LabelEncoder()
dataTrain['Age bins_dummies'] = le.fit_transform(dataTrain['Age Quantili'])


# In[ ]:


le = LabelEncoder()
dataTest['Age bins_dummies'] = le.fit_transform(dataTest['Age Quantili'])


# In[ ]:


dataTrain.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


dataTrain_copy = dataTrain.copy()


# In[ ]:


scaler = MinMaxScaler()
pclass_scaled = scaler.fit_transform(dataTrain_copy.Pclass.values.reshape(-1,1))


# In[ ]:


dataTrain_copy['Pclass'] = pclass_scaled


# In[ ]:


scaler2 = MinMaxScaler()
age_dummies_scaled = scaler2.fit_transform(dataTrain_copy['Age bins_dummies'].values.reshape(-1,1))


# In[ ]:


dataTrain_copy['Age bins_dummies'] = age_dummies_scaled
dataTrain_copy.head(10)


# In[ ]:


X = dataTrain_copy[['Pclass', 'Family number', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Age bins_dummies']]


# In[ ]:


X.head(10)


# In[ ]:


y = dataTrain.Survived


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# In[ ]:


''' check out the accuracy for the LogisticRegression'''
logreg = LogisticRegression()
scores = cross_val_score(logreg, X, y, cv = 10, scoring = 'accuracy')


# In[ ]:


scores.mean()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


k_range = range(1,25)
knn_scores = []
for i in k_range:
    knn =  KNeighborsClassifier(n_neighbors = i)
    score_array = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    knn_scores.append((i,score_array.mean()))


# In[ ]:


sorted(knn_scores, key = lambda x: x[1], reverse= True)[0]


# In[ ]:


dataTest.head()


# In[ ]:


scaler = MinMaxScaler()
pclass_scaled_test = scaler.fit_transform(dataTest.Pclass.values.reshape(-1,1))
dataTest['Pclass'] = pclass_scaled_test
scaler2 = MinMaxScaler()
age_dummies_scaled = scaler2.fit_transform(dataTest['Age bins_dummies'].values.reshape(-1,1))
dataTest['Age bins_dummies'] = age_dummies_scaled


# In[ ]:


dataTest.head()


# In[ ]:


X_test = dataTest[['Pclass', 'Family number', 'Sex_male', 'Embarked_Q', 'Embarked_S','Age bins_dummies']]


# In[ ]:


logreg.fit(X,y)
LG_predictions = logreg.predict(X_test)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(X,y)
knn_predictions = knn.predict(X_test)


# In[ ]:


titanic_submission = pd.DataFrame(dict(PassengerId = dataTest['PassengerId'].values, LG_predictions = LG_predictions, KNN_predictions = knn_predictions))


# In[ ]:


titanic_submission.head(10)


# In[ ]:


''' we choose the KNN model'''


# In[ ]:


final = titanic_submission.drop(['LG_predictions'], axis = 1)
final.rename(columns = {'KNN_predictions':'Survived'}, inplace = True)
final.head()


# In[ ]:


final.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




