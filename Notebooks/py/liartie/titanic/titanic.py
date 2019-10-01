#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df  = pd.read_csv('../input/train.csv')


# In[ ]:


train_df[train_df['Age'] >= 60]


# In[ ]:


train_df.isnull().sum().sort_values(ascending=False)


# In[ ]:


train_df[train_df['Survived'] == 0]['Survived'].plot.hist()
train_df[train_df['Survived'] == 1]['Survived'].plot.hist()


# In[ ]:


train_df[train_df['Survived'] == 0]['Age'].plot.hist(alpha=0.5)
train_df[train_df['Survived'] == 1]['Age'].plot.hist(alpha=0.5)


# In[ ]:





# In[ ]:


gender = {'male':0, 'female':1}
train_df['Sex'] = train_df['Sex'].map(gender)


# In[ ]:


train_df[train_df['Survived'] == 0]['Sex'].plot.hist(alpha=0.5)
train_df[train_df['Survived'] == 1]['Sex'].plot.hist(alpha=0.5)


# In[ ]:


train_df[train_df['Survived'] == 0]['Pclass'].plot.hist(alpha=0.5, label='Didnt')
train_df[train_df['Survived'] == 1]['Pclass'].plot.hist(alpha=0.5, label="Survived")
plt.legend()


# In[ ]:


train_X = [[1,0],[0,1], [0.25, 0.75], [1,1], [0,0]]
train_Y = [0,1,1,1,0]


# In[ ]:


without_nulls = train_df.dropna()
train_Y = without_nulls['Survived']
train_X = without_nulls[['Age', 'Sex', 'Pclass']]


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df['Sex'] = test_df['Sex'].map(gender)
test_X = test_df[['Age', 'Sex', 'Pclass']]
test_X = test_X.fillna(12)
test_X


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.5)


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_Y)
Y_pred = decision_tree.predict(test_X)
# pd.concat([pd.Series(Y_pred), test_X], axis=1)
# accuracy = decision_tree.score(X_test, y_test)
# accuracy
Y_pred


# In[ ]:


accuracy = decision_tree.score(train_X, train_Y)
accuracy


# In[ ]:


sub = pd.read_csv('../input/gender_submission.csv')
for idx, row in enumerate(sub.iterrows()):
    sub.iloc[idx]['Survived'] = Y_pred[idx]
sub.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




