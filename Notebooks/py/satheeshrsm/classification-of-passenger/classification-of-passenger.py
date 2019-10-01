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


# In[ ]:


dataset.head()


# In[ ]:


dataset[['class1','class2']] = pd.get_dummies(dataset['Pclass'],drop_first=True)


# In[ ]:


dataset.head()


# In[ ]:


import seaborn as sns
dataset.notnull()
sns.heatmap(dataset.notnull())


# In[ ]:



def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age,axis = 1)


# In[ ]:


sns.heatmap(dataset.isnull())


# In[ ]:


dataset.drop('Cabin',inplace = True,axis = 1)


# In[ ]:


dataset['Sex'] = pd.get_dummies(dataset['Sex'],drop_first=True) 


# In[ ]:


dataset.head()


# In[ ]:


embark = pd.get_dummies(dataset['Embarked'],drop_first = True)


# In[ ]:



dataset.head()


# In[ ]:


dataset.dropna(inplace = True)


# In[ ]:


sns.heatmap(dataset.isnull())


# In[ ]:


dataset = pd.concat([dataset,embark],axis = 1)


# In[ ]:


dataset.head()


# In[ ]:


dataset.drop(['PassengerId','Name','Ticket','Embarked'],axis = 1,inplace = True)


# In[ ]:


dataset.head()


# In[ ]:


x = dataset.drop('Survived',axis = 1)
x = x.drop('Pclass',axis = 1)
y = dataset['Survived']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators = 300,criterion = 'entropy')


# In[ ]:


x.dropna(inplace = True)


# In[ ]:


y.dropna(inplace = True)


# In[ ]:


rfc.fit(x,y)


# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


test.head()


# In[ ]:


embark = pd.get_dummies(test['Embarked'],drop_first=True)


# In[ ]:


sex = pd.get_dummies(test['Sex'],drop_first=True)


# In[ ]:


test.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],inplace = True,axis = 1)


# In[ ]:


new_test = pd.concat([test,sex,embark],axis = 1)


# In[ ]:


new_test.head()


# In[ ]:


new_test['Age'] = new_test[['Age','Pclass']].apply(impute_age,axis = 1)


# In[ ]:


sns.heatmap(new_test.isnull())


# In[ ]:


new_test['Sex'] = new_test['male']


# In[ ]:


new_test.drop('male',axis = 1,inplace = True)


# In[ ]:


new_test.head()


# In[ ]:


x.head()


# In[ ]:


new_test[['class1','class2']] = pd.get_dummies(new_test['Pclass'],drop_first = True)


# In[ ]:


new_test.head()


# In[ ]:


new_test = new_test.drop('Pclass',axis = 1)


# In[ ]:


sns.heatmap(new_test.notna())


# In[ ]:


x.head()


# In[ ]:


test_set = new_test[['Sex','Age','SibSp','Parch','Fare','class1','class2','Q','S']]


# In[ ]:


sns.heatmap(test_set.isna())


# In[ ]:


np.where(test_set.isna())


# In[ ]:


new_test = test_set.fillna(test_set['Fare'].mean())


# In[ ]:


y_pred = rfc.predict(new_test)


# In[ ]:


y_pred


# In[ ]:




