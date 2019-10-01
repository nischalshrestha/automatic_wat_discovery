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


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train_set= pd.read_csv("../input/train.csv")
#test_set= pd.read_csv("../input/test.csv")
train_set.shape


# In[ ]:


data=train_set.copy()


# In[ ]:


#frames = [train_set,test_set]
#data = pd.concat(frames,sort=False)
#data.shape


# In[ ]:


data


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe(include="all")


# In[ ]:


data.drop(["PassengerId","Ticket","Fare","Cabin"],axis=1,inplace=True)
data.head()


# In[ ]:


mean1=int(data[data.Pclass==1]["Age"].mean())
print("mean of 1st class",mean1)
mean2=int(data[data.Pclass==2]["Age"].mean())
print("mean of 2nd class",mean2)
mean3=int(data[data.Pclass==3]["Age"].mean())
print("mean of 3rd class",mean3)


# In[ ]:


def impute_age(a):
    Age= a[0]
    Pclass=  a[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return mean1
        elif Pclass==2:
            return mean2
        elif Pclass ==3:
            return mean3
    else:
        return data.Age


# In[ ]:


data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)
data.Age.isnull().sum()


# In[ ]:


#filling missing Age with interpolate
#data.Age.interpolate(inplace=True)
#data.Age=data.Age.astype(int)


# In[ ]:


#filling nan of column Embarked with S
data.Embarked.fillna(data.Embarked.value_counts().index[0],inplace=True)
#Embarked is filled
data.Embarked.isnull().sum()


# In[ ]:


data.info()


# In[ ]:


#sibsp+parch is alone or with family
data["family_or_alone"]= data.SibSp + data.Parch
data["family_type"]= ["alone" if x==0 else "small_family" if x<5 else "big_family" for x in data.family_or_alone ]


# In[ ]:


data.head()


# In[ ]:


sns.countplot(x="Survived",data=data,hue="family_type")
plt.show()


# In[ ]:


#converting age to children, young people,old people
data["age_type"]=["children" if x<17  else "adult" if x<81 else "none" for x in data.Age]
        


# In[ ]:


sns.countplot(x="Survived",data=data,hue="age_type")
plt.show()


# In[ ]:


sns.countplot(x="Survived",data=data,hue="Pclass")
plt.show()


# In[ ]:


data.Survived.value_counts()


# In[ ]:


data.Survived.value_counts(normalize=True)


# In[ ]:


sns.countplot(x="Survived",data=data,hue="Sex")
plt.show()


# In[ ]:


sns.countplot(x="Survived",data=data,hue="family_or_alone")
plt.show()


# In[ ]:


sns.countplot(x="Embarked",data=data,hue="Survived")


# In[ ]:


data.Pclass.value_counts()


# In[ ]:


pd.crosstab(data.Survived,data.Pclass,margins=True)


# In[ ]:


data.drop(["Name","family_or_alone","family_type","age_type"],axis=1,inplace=True)
data=pd.get_dummies(data,drop_first=True)


# In[ ]:


data.head()


# In[ ]:


X=data.drop("Survived",axis=1)
y=data.Survived


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(accuracy_score(y_test,y_pred))

