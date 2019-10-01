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


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
from sklearn import metrics


# In[ ]:


train= pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


sns.countplot(train['Survived'])


# In[ ]:


sns.factorplot('Pclass', 'Survived', data=train, hue='Sex')


# In[ ]:


train['Age'].fillna(train['Age'].median(), inplace=True)


# In[ ]:


sns.countplot(train['Survived'], hue=train['Sex'])


# In[ ]:


sns.countplot(train['Embarked'])


# In[ ]:


sns.boxplot(train['Survived'],train['Fare'], hue= train['Embarked'])


# In[ ]:


train[train['Embarked'].isnull()]


# In[ ]:


sns.boxplot(train['Embarked'],train['Fare'], hue= train['Pclass'])


# In[ ]:


train['Embarked'] = train['Embarked'].fillna('C')


# In[ ]:


sns.boxplot(train['Survived'],train['Fare'], hue= train['Embarked'])


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


test[test['Fare'].isnull()]


# In[ ]:


median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()
#'S'
       #print(median_fare)
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df


# In[ ]:


test[(test['Pclass']==3) & (test['Embarked'] == 'S')]['Fare'].unique()


# In[ ]:


Fare_med= test[(test['Pclass']==3) & (test['Embarked'] == 'S')]['Fare'].median()


# In[ ]:


test['Fare']=test['Fare'].fillna('Fare_med')


# In[ ]:


train['cabin']= train.Cabin.str[0]
test['cabin']= test.Cabin.str[0]


# In[ ]:


train.head()


# In[ ]:


train.cabin=train.cabin.fillna('U')
test.cabin=test.cabin.fillna('U')


# In[ ]:


train.head(50)


# In[ ]:


train['Family']= train['Parch']+ train['SibSp']+1
test['Family']= test['Parch']+ test['SibSp']+1


# In[ ]:


train.loc[train["Family"] == 1, "FamilySize"] = 'singleton'
train.loc[(train["Family"] > 1)  &  (train["Family"] < 5) , "FamilySize"] = 'small'
train.loc[train["Family"] >4, "FamilySize"] = 'large'
test.loc[test["Family"] == 1, "FamilySize"] = 'singleton'
test.loc[(test["Family"] > 1)  &  (test["Family"] < 5) , "FamilySize"] = 'small'
test.loc[test["Family"] >4, "FamilySize"] = 'large'


# In[ ]:


sns.countplot(train['FamilySize'],hue=train['Survived'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


sns.heatmap(train.corr(), annot=True)


# In[ ]:


sns.countplot(train['Embarked'])


# In[ ]:


sns.factorplot('Pclass', 'Survived', data=train, hue='Sex')

