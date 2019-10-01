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
import seaborn as sns
from matplotlib import pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


titanic=pd.read_csv("../input/train.csv")
titanic.head()


# In[ ]:


titanic.describe()


# From the count of every data , i can conclude that some Age data are missing  and 1 fare data  is missing.

# In[ ]:


sns.heatmap(titanic.isnull())


# From the Heat map it is ibvious that there are many null values in Age and Cabin. 
# So my task will be flling those null value with sob 

# In[ ]:


titanic.info()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x="Age",hue="Sex",data=titanic)


# From the above count plot we can see that no. of male is greater than no. of female.

# In[ ]:


sns.distplot(titanic[titanic["Survived"]==1]['Age'].dropna())


# Distplot shows that maximum people of age group 20 to 40 survived,
# there can be 2 reason no. of people in this age group was much much more or survival team was biased on helping these young ones only.

# In[ ]:


sns.boxplot(x="Pclass",y="Age",data=titanic)


# here we can see  Pclass 1 group are between ages 30 to 50,
# Pclass 2 between 25 to 35,
# Pclass 3 between 20 to 30.
# 
# So now we will be imputing the missing age in the dataset with the above inference
# 

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


titanic['Age'] = titanic[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


titanic.drop("Cabin",inplace=True,axis=1)


# In[ ]:


sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Now all the age are imputed.  now there is no bad data.

# In[ ]:


sex = pd.get_dummies(titanic['Sex'],drop_first=True)
embark = pd.get_dummies(titanic['Embarked'],drop_first=True)


# In[ ]:


titanic.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


titanic = pd.concat([titanic,sex,embark],axis=1)


# In[ ]:


titanic.head()


# Converted Sex in machine readable format.  1- Male 0- Female
# 
# 

# In[ ]:


Y_train=titanic["Survived"]
X_train=titanic.drop("Survived",axis=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)


# In[ ]:


titanic_test=pd.read_csv("../input/test.csv")
titanic_test.head()


# In[ ]:


sns.heatmap(titanic_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(titanic_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


titanic_test.drop(["Cabin"],axis=1,inplace=True)


# In[ ]:


sns.heatmap(titanic_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sex_test = pd.get_dummies(titanic_test['Sex'],drop_first=True)
embark_test = pd.get_dummies(titanic_test['Embarked'],drop_first=True)


# In[ ]:


titanic_test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


titanic_test = pd.concat([titanic_test,sex,embark],axis=1)


# In[ ]:


titanic_test.head()


# In[ ]:


titanic_test.describe()


# In[ ]:


predictions = logmodel.predict(titanic_test)


# In[ ]:




