#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# In[ ]:


trainingData = pd.read_csv("../input/train.csv")
testingData = pd.read_csv("../input/test.csv")


# In[ ]:


testingData.head()


# In[ ]:


trainingData.head()


# # Preprocessing

# In[ ]:


Features = ["Pclass","Sex","Age","Fare","Parch","SibSp","Embarked"]
train_X = trainingData[Features]
train_y = trainingData["Survived"]
test_X = testingData[Features]


# In[ ]:


train_X.isnull().sum()


# In[ ]:


test_X.isnull().sum()


# In[ ]:


train_X['Age'].fillna(train_X['Age'].median(),inplace=True)
test_X['Age'].fillna(train_X['Age'].median(),inplace=True)
test_X['Fare'].fillna(test_X['Fare'].median(),inplace=True)


# In[ ]:


train_X['Embarked'].fillna(train_X['Embarked'].value_counts().index[0], inplace=True)


# In[ ]:


p = {1:'1st',2:'2nd',3:'3rd'} 
train_X['Pclass'] = train_X['Pclass'].map(p)
test_X['Pclass'] = test_X['Pclass'].map(p)


# In[ ]:


categorical_df = train_X[['Pclass','Sex','Embarked']]
one_hot_encode = pd.get_dummies(categorical_df,drop_first=True) 
train_X = train_X.drop(['Pclass','Sex','Embarked'],axis=1)
train_X = pd.concat([train_X,one_hot_encode],axis=1)


# In[ ]:


categorical_df = test_X[['Pclass','Sex','Embarked']]
one_hot_encode = pd.get_dummies(categorical_df,drop_first=True) 
test_X = test_X.drop(['Pclass','Sex','Embarked'],axis=1)
test_X = pd.concat([test_X,one_hot_encode],axis=1)


# In[ ]:





# # Training Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(train_X,train_y)


# In[ ]:


pred = clf.predict(test_X)


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': testingData.PassengerId, 'Survived': pred})


# In[ ]:


my_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




