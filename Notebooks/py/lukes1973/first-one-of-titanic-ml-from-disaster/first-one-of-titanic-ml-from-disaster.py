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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
test=pd.read_csv('../input/test.csv')
train=pd.read_csv('../input/train.csv')



# In[ ]:


import sklearn


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


train['Age']=train['Age'].fillna(train['Age'].median())


# In[ ]:


train['Sex'][train['Sex']=='male']=0
train['Sex'][train['Sex']=='female']=1


# In[ ]:


train['Embarked']=train['Embarked'].fillna('S')


# In[ ]:


train['Embarked'][train['Embarked']=='S']=0
train['Embarked'][train['Embarked']=='C']=1
train['Embarked'][train['Embarked']=='Q']=2


# In[ ]:


features_forest=train[['Pclass','Age','Sex','Fare','SibSp','Parch','Embarked']].values


# In[ ]:


forest=RandomForestClassifier(max_depth=10,min_samples_split=2,n_estimators=100,random_state=1)


# In[ ]:


target=train['Survived'].values


# In[ ]:


my_forest=forest.fit(features_forest,target)


# In[ ]:


test['Fare'][152]=test.Fare.median()


# In[ ]:


test['Age']=test['Age'].fillna(test['Age'].median())


# In[ ]:


test['Embarked']=test['Embarked'].fillna('S')


# In[ ]:


test['Sex'][test['Sex']=='male']=0
test['Sex'][test['Sex']=='female']=1


# In[ ]:


test['Embarked'][test['Embarked']=='S']=0
test['Embarked'][test['Embarked']=='C']=1
test['Embarked'][test['Embarked']=='Q']=2


# In[ ]:


test_features=test[['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']].values


# In[ ]:


my_first_prediction=my_forest.predict(test_features)


# In[ ]:


PassengerId=np.array(test['PassengerId']).astype(int)


# In[ ]:


my_solution=pd.DataFrame(my_first_prediction,PassengerId,columns=['Survived'])


# In[ ]:


my_solution.to_csv('my_first_submission.csv',index_label=['PassengerId'])


# In[ ]:




