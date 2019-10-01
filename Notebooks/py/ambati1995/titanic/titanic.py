#!/usr/bin/env python
# coding: utf-8

# In[15]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[6]:


#loading Train data
titanic_train = pd.read_csv('../input/train.csv')
#loading test data
titanic_test = pd.read_csv('../input/test.csv')


# In[7]:


titanic_train.head()


# In[8]:


titanic_train.info()


# In[9]:


titanic_test.info()


# In[10]:


#cleaning Train data
titanic_train["Age"]=titanic_train["Age"].fillna(titanic_train["Age"].median())
titanic_train.loc[titanic_train["Sex"] == "male", "Sex"] = 0
titanic_train.loc[titanic_train["Sex"] == "female", "Sex"] = 1
titanic_train["Embarked"] = titanic_train["Embarked"].fillna('S')
titanic_train.loc[titanic_train["Embarked"] == 'S',"Embarked"] = 0
titanic_train.loc[titanic_train["Embarked"] == 'C',"Embarked"] = 1
titanic_train.loc[titanic_train["Embarked"] == 'Q',"Embarked"] = 2


# In[12]:


#Cleaning test data
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_train["Age"].median())
titanic_test.loc[titanic_test["Sex"] == "male","Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female","Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna('S')
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Embarked"] == 'S',"Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == 'C',"Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == 'Q',"Embarked"] = 2


# In[13]:


# Features to train
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


# In[16]:


sur_model = RandomForestClassifier(random_state=1)


# In[17]:


sur_model.fit(titanic_train[features],titanic_train['Survived'])


# In[18]:


predictions = sur_model.predict(titanic_test[features])


# In[22]:


titanic_test = pd.read_csv("../input/test.csv")
titanic_test.insert(len(titanic_test.columns),"Survived", predictions.astype(int))
#print(predictions.astype(int))

submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
    
#print(submission)


# In[27]:


submission.to_csv("output.csv", sep=',', encoding='utf-8',index = False,index_label=False)


# In[ ]:




