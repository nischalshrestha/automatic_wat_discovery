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
""
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/train.csv")


# In[ ]:


data.head()


# In[ ]:


data['Cabin'].isnull().value_counts()


# In[ ]:


data.drop(['Name','Ticket','Cabin'],inplace=True,axis=1)


# In[ ]:


data['Age'].isnull().value_counts()


# In[ ]:


import seaborn as sns
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:



sns.barplot(x='Pclass',y='Age',data=data)


# In[ ]:


def age_fill(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


data['Age']=data[['Age','Pclass']].apply(age_fill,axis=1)


# In[ ]:


data.head(10)


# In[ ]:


data['Sex']=data['Sex'].apply(lambda x : 1 if x=='male' else 0 )


# In[ ]:


Embarked=pd.get_dummies(data['Embarked'],drop_first=True)


# In[ ]:


data.drop('Embarked',axis=1,inplace=True)


# In[ ]:


data=pd.concat([data,Embarked],axis=1)


# In[ ]:


data.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model=LogisticRegression() 


# In[ ]:


X=data.drop('Survived',axis=1)
y=data['Survived']


# In[ ]:


model.fit(X,y)


# In[ ]:


test_data=pd.read_csv('../input/test.csv')


# In[ ]:


test_data.head()


# In[ ]:


test_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


test_data['Sex']=test_data['Sex'].apply(lambda x : 1 if x=='male' else 0 )


# In[ ]:


sns.heatmap(test_data.isnull(),cbar=False)


# In[ ]:


test_data['Age']=test_data[['Age','Pclass']].apply(age_fill,axis=1)


# In[ ]:


test_data.head()


# In[ ]:


Embarked=pd.get_dummies(test_data['Embarked'],drop_first=True)  


# In[ ]:


test_data.drop('Embarked',axis=1,inplace=True)


# In[ ]:


test_data=pd.concat([test_data,Embarked],axis=1)


# In[ ]:


test_data["Fare"]=test_data["Fare"].fillna(value=test_data['Fare'].mean()) 


# In[ ]:


predictions=model.predict(test_data)


# In[ ]:


gender=pd.read_csv("../input/gender_submission.csv")  


# In[ ]:


gender


# In[ ]:




