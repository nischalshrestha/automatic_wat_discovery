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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm
from pandas.tools.plotting import parallel_coordinates
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
train_data.columns


# In[ ]:


train_data.head()


# In[ ]:


train_data.dtypes


# Clean data by dropping columns which we are not using for visualization

# In[ ]:


train_data.drop(['PassengerId','Ticket','Cabin'], axis=1, inplace = True)


# Check wether data have null values or not

# In[ ]:


train_data.isnull().sum()


# Cleaning missing data
# 
# In statistics, missing data, or missing values, occur when no data value is stored for the variable in an observation. Missing data are a common occurrence and can have a significant effect on the conclusions that can be drawn from the data. The goal of cleaning operations is to prevent problems caused by missing data that can arise when training a model.

# 'Embarked' data is obect type.
# So, we are using mode() to fill the missing data

# In[ ]:


#complete missing age with mean
train_data['Age'].fillna(train_data['Age'].mean(), inplace = True)
#complete embarked with mode
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace = True)


# Check whether all missing data are filled

# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data['survived_dead'] = train_data['Survived'].apply(lambda x : 'Survived' if x == 1 else 'Dead')


# In[ ]:


train_data.describe()


# In[ ]:


sns.clustermap(data = train_data.corr().abs(),annot=True, fmt = ".2f", cmap = 'Blues')


# In[ ]:


sns.countplot('survived_dead', data = train_data)


# In[ ]:


sns.countplot( train_data['Sex'],data = train_data, hue = 'survived_dead', palette='coolwarm')


# In[ ]:


sns.countplot( train_data['Pclass'],data = train_data, hue = 'survived_dead')


# In[ ]:


sns.barplot(x = 'Pclass', y = 'Fare', data = train_data)


# In[ ]:


sns.pointplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = train_data);


# Fare - Passenger Fare
# Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[ ]:


sns.barplot(x  = 'Embarked', y = 'Fare', data = train_data)


# In[ ]:


g = sns.FacetGrid(train_data, hue='Survived')
g.map(sns.kdeplot, "Age",shade=True)


# In[ ]:


sns.catplot(x="Embarked", y="Survived", hue="Sex",
            col="Pclass", kind = 'bar',data=train_data, palette = "rainbow")


# sibsp - Number of Siblings/Spouses Aboard
# 
# 

# In[ ]:


sns.catplot(x='SibSp', y='Survived',hue = 'Sex',data=train_data, kind='bar')


# parch - Number of Parents/Children Aboard

# In[ ]:


sns.catplot(x='Parch', y='Survived',hue = 'Sex',data=train_data, kind='point')


# In[ ]:


g= sns.FacetGrid(data = train_data, row = 'Sex', col = 'Pclass', hue = 'survived_dead')
g.map(sns.kdeplot, 'Age', alpha = .75, shade = True)
plt.legend()

