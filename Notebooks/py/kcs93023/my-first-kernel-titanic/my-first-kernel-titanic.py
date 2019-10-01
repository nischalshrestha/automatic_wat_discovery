#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from os.path import join
input_dir = "../input"

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv(join(input_dir, 'train.csv'))
df_train_label = df_train['Survived']
del df_train['Survived']
df_test = pd.read_csv(join(input_dir, 'test.csv'))
df_test_label = pd.read_csv(join(input_dir, 'gender_submission.csv')).values[:,1]


# Just Watch Train DataSet

# In[ ]:


df_train.head()


# Just Watch Test Dataset

# In[ ]:


df_test.head()


# The 'name' feature does not seem to be related to 'survival'.
# So delete it

# In[ ]:


del df_train["Name"]
del df_test["Name"]


# In[ ]:


df_train


# In[ ]:


df_test.head()


# Check Null Values

# In[ ]:


msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))


# Back up Cabin Columns and delete from dataset

# In[ ]:


df_train_Cabin = df_train["Cabin"]
df_test_Cabin = df_test["Cabin"]


# 'Ticket' feature just means 'Ticket Number', So meaningless feature.

# In[ ]:


del df_train["Cabin"]
del df_test["Cabin"]
del df_train['Ticket']
del df_test['Ticket']


# In[ ]:


df_train.replace(np.NaN, np.nan, inplace=True)
df_test.replace(np.NaN, np.nan, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


df_train_dummy = pd.get_dummies(df_train)
df_test_dummy = pd.get_dummies(df_test)


# In[ ]:


df_train_dummy.head()


# In[ ]:


for c in df_train_dummy.columns : 
    df_train_dummy[c] = df_train_dummy[c].astype(np.float64)
# df_train_dummy.replace(np.NaN, np.nan, inplace=True)
# df_test_dummy.replace(np.NaN, np.nan, inplace=True)


# In[ ]:


from fancyimpute import MICE
train_mice = MICE(n_imputations=200, impute_type='pmm', verbose=False).complete(df_train_dummy.values)
test_mice = MICE(n_imputations=200, impute_type='pmm',  verbose=False).complete(df_test_dummy.values)


# In[ ]:


df_train_mice = pd.DataFrame(train_mice, columns = df_train_dummy.columns)
df_test_mice = pd.DataFrame(test_mice, columns = df_test_dummy.columns)


# In[ ]:


msno.matrix(df=df_train_mice.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))


# In[ ]:


msno.matrix(df=df_test_mice.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))


# Let's Classify to use various Classifier!

# ### 1. Support Vector Machine Classifier

# In[ ]:


from sklearn.svm import SVC
clf1 = SVC(kernel='linear')
clf1.fit(df_train_mice, df_train_label)


# In[ ]:


test_result = clf1.score(df_test_mice,df_test_label)
test_result


# ### 2.Logistic Regression Classifier

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression()
clf2.fit(df_train_mice, df_train_label)


# In[ ]:


test_result = clf2.score(df_test_mice,df_test_label)
test_result


# ### 3. RandomForest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(max_depth = 5)
clf3.fit(df_train_mice, df_train_label)


# In[ ]:


test_result = clf3.score(df_test_mice,df_test_label)
test_result


# ### 4. RidgeClassifier

# In[ ]:


from sklearn.linear_model import RidgeClassifier
clf4 = RidgeClassifier()
clf4.fit(df_train_mice, df_train_label)


# In[ ]:


test_result = clf4.score(df_test_mice,df_test_label)
test_result


# In[ ]:


from sklearn.ensemble import VotingClassifier

eclf = VotingClassifier(estimators=[('svm', clf1),('lr',clf2),('RF', clf3),('Ridge',clf4)])
eclf.fit(df_train_mice, df_train_label)


# In[ ]:


test_result = eclf.score(df_test_mice,df_test_label)
test_result


# In[ ]:


submission = pd.DataFrame()

submission['PassengerId'] = df_test_mice['PassengerId']

submission['Survived'] = test_result

grouped_test = submission[['PassengerId', 'Survived']].groupby('PassengerId').sum().reset_index()
grouped_test.to_csv('submit.csv',index=False)


# In[ ]:




