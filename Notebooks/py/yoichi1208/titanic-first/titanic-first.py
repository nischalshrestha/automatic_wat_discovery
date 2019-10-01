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
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
get_ipython().magic(u'matplotlib inline')
sns.set()



# In[ ]:


# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])


# In[ ]:


data.info()


# In[ ]:


data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()


# In[ ]:


data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# In[ ]:


data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()


# In[ ]:


data_train = data.iloc[:891]
data_test = data.iloc[891:]


# In[ ]:


X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# In[ ]:


Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred


# In[ ]:


df_test[['PassengerId', 'Survived']].to_csv('1st_dec_tree.csv', index=False)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


forest = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)


# In[ ]:


forest.fit(X,y)


# In[ ]:


test


# In[ ]:


Y_pred2 = forest.predict(test)


# In[ ]:


df_test = df_test.drop(['Survived'],axis=1)


# In[ ]:


Y_pred2


# In[ ]:


df_test['Survived'] = Y_pred2


# In[ ]:


df_test


# In[ ]:


df_test[['PassengerId', 'Survived']].to_csv('1st_RF.csv', index=False)


# In[ ]:




