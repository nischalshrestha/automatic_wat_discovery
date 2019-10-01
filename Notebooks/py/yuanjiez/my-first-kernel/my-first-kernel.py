#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/train.csv")



# In[ ]:


#show the info of train data
train.info()


# In[ ]:


#show top5 samples of data
train.head()


# In[ ]:


#show total null values in data
train.isnull().sum()


# In[ ]:


#show the heatmap for null values in data
sns.heatmap(train.isnull(),cmap="YlGnBu",cbar=False,yticklabels=False)


# In[ ]:


#fill null values of age by median
age_median = train.Age.median()
print(age_median)
train["Age"]=train.Age.fillna(age_median)
train.Age.describe()



# In[ ]:


bins=[0,10,18,30,60,np.inf]
labels=['Child','Teenager','Young','Adult','Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)


# In[ ]:


#too many null values, so drop it
train=train.drop("Cabin",axis=1)
train.head()


# In[ ]:


train.Embarked.value_counts()


# In[ ]:


#use the mostly value to fill the embarked
train=train.fillna({"Embarked":"S"})
train[train.Embarked.isnull()].Embarked


# In[ ]:


#show survived
sns.countplot(data=train,x="Survived")


# In[ ]:


#relationshio between Pclass and Fare.
sns.barplot(data=train,x="Pclass",y="Fare",ci=None)


# In[ ]:


#show the relationship between class and survived.The survived of 1st is the highest
sns.barplot(data=train,x="Pclass",y="Survived",ci=None)


# In[ ]:


#show the relationship between class and survived.The survived of female is the highest
sns.barplot(data=train,x="Sex",y="Survived",ci=None)


# In[ ]:


#show the relationship for class,sex and survived
sns.pointplot(data=train,x="Pclass",y="Survived",hue="Sex",ci=None)


# In[ ]:


#show the relationship for age and survived
plt.figure(figsize= (10 ,5))
sns.set_style("whitegrid")
sns.barplot(data=train,x="AgeGroup",y="Survived",ci=None)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


#show the relationship for age,sex and survived
plt.figure(figsize= (10 ,5))
sns.set_style("whitegrid")
sns.pointplot(data=train,x="AgeGroup",y="Survived",hue="Sex",ci=None)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


train.head()


# In[ ]:


age_mapping = {'Child':1,'Teenager':2,'Young':3,'Adult':4,'Senior':5}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis = 1)
train.head()


# In[ ]:


train=train.drop(['Name','Ticket'],axis=1)
train.head()


# In[ ]:


sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
train.head()


# In[ ]:


embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
train.head()


# In[ ]:


test=test.drop(['Name','Ticket','Cabin'],axis=1)

test.head()


# In[ ]:


#fill null values of age by median
age_median = test.Age.median()
print(age_median)
test["Age"]=test.Age.fillna(age_median)
test.Age.describe()


# In[ ]:


bins=[0,10,18,30,60,np.inf]
labels=['Child','Teenager','Young','Adult','Senior']
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)
age_mapping = {'Child':1,'Teenager':2,'Young':3,'Adult':4,'Senior':5}
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
test = test.drop(['Age'], axis = 1)
test.head()


# In[ ]:


sex_mapping = {"male": 0, "female": 1}
test['Sex'] = test['Sex'].map(sex_mapping)
test.head()


# In[ ]:


test=test.fillna({"Embarked":"S"})
test[test.Embarked.isnull()].Embarked


# In[ ]:


embarked_mapping = {"S": 1, "C": 2, "Q": 3}
test['Embarked'] = test['Embarked'].map(embarked_mapping)
test.head()


# In[ ]:


#mechine learning
from sklearn.model_selection import train_test_split
features = train.drop(['Survived', 'PassengerId'], axis=1)
target_labels=train["Survived"]
X_train, X_val, Y_train, Y_val = train_test_split(features, target_labels, test_size = 0.2, random_state = 0)


# In[ ]:



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,f1_score
LogisticReg =LogisticRegression()
LogisticReg.fit(X_train, Y_train)
y_pred = LogisticReg.predict(X_val)
print("ACC:",accuracy_score(Y_val,y_pred))
print("REC:",recall_score(Y_val,y_pred))
print("F1:",f1_score(Y_val,y_pred))


# **Sources:**
# 
# 1.https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner
# 2.http://coffee.pmcaff.com/article/1013168772722816/pmcaff?utm_source=forum&from=related&pmc_param%5Bentry_id%5D=924376977711232
