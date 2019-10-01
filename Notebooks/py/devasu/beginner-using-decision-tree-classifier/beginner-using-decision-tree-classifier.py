#!/usr/bin/env python
# coding: utf-8

# # In this version I'll be using imputing which I've never used before

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


# # Including all the Required Libraries 

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


y = train['Survived'].copy()


# In[ ]:


features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']  #All the features we might need in the model


# In[ ]:


X = train[features].copy()


# In[ ]:


X = pd.get_dummies(X) #one hot code encoding
X.head()


# In[ ]:


first_imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
X = pd.DataFrame(first_imputer.fit_transform(X))


# In[ ]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=1,test_size=0.33)


# In[ ]:


def AccuracyTracker(Xtrain,Xtest,ytrain,ytest,n):
    model = DecisionTreeClassifier(max_leaf_nodes=n,random_state=1)
    model.fit(Xtrain,ytrain)
    print(n,accuracy_score(ytest,model.predict(Xtest)))
for i in range(2,50):
    AccuracyTracker(Xtrain,Xtest,ytrain,ytest,i)


# ## accuracy is maximum for n=12 which is 0.7864406779661017

# In[ ]:


model = RandomForestClassifier(n_estimators=100)
model.fit(Xtrain,ytrain)
accuracy_score(ytest,model.predict(Xtest))


# # This is even less than the last time, so we will use DT

# In[ ]:





# # Now we have all the features(X) and correspoding output (y) to create a trained Model

# In[ ]:


modeltree = DecisionTreeClassifier(max_leaf_nodes=12,random_state=1)  #hollow tree created now let's put the data
modeltree.fit(X,y)


# # Now we will process the test data in the same way

# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


pretest = test[features].copy()
pretest = pd.get_dummies(pretest)


# In[ ]:


imputedpretest = pd.DataFrame(first_imputer.fit_transform(pretest))


# In[ ]:


imputedpretest


# ## We have all set testdata now let's predict using this data

# In[ ]:


res = modeltree.predict(imputedpretest)


# In[ ]:


ansdic = {'PassengerId':test['PassengerId'],'Survived':res}
ans = pd.DataFrame(ansdic)
ans.head()


# In[ ]:


ans.to_csv("answer.csv",index=False)


# In[ ]:





# In[ ]:




