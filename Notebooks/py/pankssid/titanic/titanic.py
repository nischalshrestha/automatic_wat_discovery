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


X_train=pd.read_csv("../input/train.csv")
X_test=pd.read_csv("../input/test.csv")
X_train.head()


# In[ ]:


cols_to_remove=['Name','Ticket','Cabin','Embarked','Fare']
df=X_train.drop(cols_to_remove,axis=1)
df_1=X_test.drop(cols_to_remove,axis=1)
print(df.head())
print(df_1.head())


# In[ ]:


sex_mapping={
    'male':0,
    'female':1,
}
df.Sex=df.Sex.map(sex_mapping)
df_1.Sex=df_1.Sex.map(sex_mapping)
df=df.dropna(axis=0)
df_1=df_1.dropna(axis=0)


# In[ ]:


df.head()
Y_train=df['Survived']
Y_train=Y_train.dropna(axis=0)
print(Y_train.shape)


# In[ ]:


X_train=df.dropna(axis=0)
X_test=df_1.dropna(axis=0)
X_train=X_train.drop('Survived',1)

print(X_train.shape)

print(X_test.shape)


# In[ ]:


X_test.head()




# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
selector=SelectKBest(chi2,k=4)
X_new=selector.fit_transform(X_train,Y_train)
print(X_new.shape)
scores=selector.scores_
print(scores)
print (scores.shape)



# In[ ]:


from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
lm.fit(X_train,Y_train)


# In[ ]:


Y_pred=lm.predict(X_test)


# In[ ]:


Submission=pd.DataFrame({
    'PassengerId':X_test['PassengerId'],
    'Survived'  : Y_pred
      
})
Submission.to_csv('titanic.csv',index=False)


# In[ ]:


Submission.head()


# In[ ]:




