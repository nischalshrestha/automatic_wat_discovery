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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


df=pd.read_csv("../input/train.csv")
dft=pd.read_csv("../input/test.csv")
df.tail()
dft.head()


# In[ ]:


df.describe()


# In[ ]:


df.Embarked.value_counts()


# In[ ]:


df[df.Embarked=='Q'][df.Survived==1].shape[0]


# In[ ]:


df[df.Sex=='female'][df.Survived==1].shape[0]


# In[ ]:


df['gender']=df['Sex'].map({'male':1,'female':0}).astype(int)
df.Embarked.fillna('S',inplace=True)
dft['gender']=dft['Sex'].map({'male':1,'female':0}).astype(int)
dft.Embarked.fillna('S',inplace=True)


# In[ ]:


df['emb']=df['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)
del df['Sex']
del df['Embarked']
df.rename(columns={'gender':'Sex'},inplace=True)
df.rename(columns={'emb':'Embarked'},inplace=True)
dft['emb']=dft['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)
del dft['Sex']
del dft['Embarked']
dft.rename(columns={'gender':'Sex'},inplace=True)
dft.rename(columns={'emb':'Embarked'},inplace=True)


# In[ ]:


df.isna().sum()


# In[ ]:


del df['Name']
del df['Ticket']
del dft['Name']
del dft['Ticket']


# In[ ]:


df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


df[df.Cabin.isna()][df.Survived==1].shape[0]
dft.isna().sum()


# In[ ]:


def con(str):
    if str=='':
        return 0
    else:
        return 1
df['cabin']=df['Cabin'].apply(con)
del df['Cabin']
dft['cabin']=dft['Cabin'].apply(con)
del dft['Cabin']


# In[ ]:


df.rename(columns={'cabin':'Cabin'},inplace=True)
dft.rename(columns={'cabin':'Cabin'},inplace=True)
del df['SibSp']
del df['Parch']
del dft['SibSp']
del dft['Parch']
dft.isna().sum()


# In[ ]:


meanS=df[df.Survived==1].Age.mean()
df["Age"]=np.where(df.Age.isna() & df.Survived==1 , meanS,df["Age"])
df.isna().sum()
meanSt=dft.Age.mean()
dft["Age"]=np.where(dft.Age.isna()  , meanSt,dft["Age"])
dft.isna().sum()


# In[ ]:


meanNS=df[df.Survived==0].Age.mean()
df["Age"].fillna(meanNS,inplace=True)


# In[ ]:


dft.isna().sum()


# In[ ]:


del df['Fare']
del dft['Fare']


# In[ ]:


dft.head()


# In[ ]:


x_train=np.array(df.iloc[:,2:9])
y_train=np.array(df.iloc[:,1])
x_test=np.array(dft.iloc[:,1:8])


# In[ ]:


clf=LogisticRegression(C=0.3,max_iter=1000000)
#clf=SVC()
#clf = RandomForestClassifier(n_estimators=100)
#clf=KNeighborsClassifier()
clf.fit(x_train,y_train)


# In[ ]:


clf.score(x_train,y_train)


# In[ ]:


y_pred=clf.predict(x_test)


# In[ ]:


f=np.c_[np.array(dft.PassengerId),y_pred]
d=pd.DataFrame(f)


# In[ ]:


d.to_csv('pred.csv',index=False,header=['PassengerId','Survived'])


# In[ ]:




