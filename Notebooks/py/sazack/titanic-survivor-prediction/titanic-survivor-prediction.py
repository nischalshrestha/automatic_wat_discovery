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


dataset = pd.read_csv("../input/train.csv")
dataset.head()


# In[ ]:


class_survived = dataset[dataset['Survived']==1]['Pclass'].value_counts()
class_died = dataset[dataset['Survived']==0]['Pclass'].value_counts()

#dataset[dataset['Survived']==1]['Pclass'].value_counts()
status_dataframe = pd.DataFrame([class_survived,class_died])
status_dataframe.index =[ 'Survived','Died']
status_dataframe.columns=['Class 1', 'Class 2', 'Class 3']
status_dataframe.plot(kind='bar', title='Survival Status based on class')


# In[ ]:


sex_survived = dataset[dataset['Survived'] ==1]['Sex'].value_counts()
sex_died = dataset[dataset['Survived'] ==0]['Sex'].value_counts()

status_by_sex_DF = pd.DataFrame([sex_survived,sex_died])
status_by_sex_DF.index=['Survived','Died']
status_by_sex_DF.plot(kind="bar", stacked=True, title="Survival Status based on Sex")


# In[ ]:


embark_survived = dataset[dataset['Survived'] ==1]['Embarked'].value_counts()
embark_died = dataset[dataset['Survived'] ==0]['Embarked'].value_counts()

status_by_embark_DF = pd.DataFrame([embark_survived,embark_died])
status_by_embark_DF.index=['Survived','Died']
status_by_embark_DF.plot(kind="bar", stacked=True, title="Survival Status based on Sex")


# In[ ]:


X = dataset.drop(['PassengerId','Cabin','Ticket','Fare','Parch','SibSp'],axis=1)
y = X.Survived
X = X.drop(['Survived'], axis=1)

X.head(50)


# In[ ]:


#Preprocessing for string data

#Encoding Sex
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X.Sex = labelEncoder_X.fit_transform(X.Sex)
# X.head(50)


# Encoding Embark for NaN

# print("Total Null values in Embarked: ", sum(X.Embarked.isnull()))
null_index = X.Embarked.isnull()
X.loc[null_index,'Embarked'] ='S'
Embarked = pd.get_dummies(X.Embarked, prefix="Embarked")
X.head(20)
X = X.drop(['Embarked'],axis=1)
X = pd.concat([X,Embarked],axis=1)
X = X.drop(['Embarked_S'],axis=1)

X.head()


# In[ ]:


#handling missing age data

n_rows = X.shape[0]
print(n_rows)
mean_age = X.Age.mean()
print(mean_age)
for i in range(0,n_rows):
    if np.isnan(X.Age[i]) == True:
        X.Age[i] = mean_age
for i in range(0,n_rows):
    if X.Age[i]>18:
        age= 0
    else:
        age = 1
    X.Age[i] = age
X=X.drop(['Name'], axis=1)
X.head(20)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
y= dataset.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)
# print(prediction)
accuracy = accuracy_score(prediction, y_test)
print("The Accuracy is :",accuracy*100,"%")

