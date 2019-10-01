#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model

import os
print(os.listdir("../input"))


# In[6]:


df = pd.read_csv('../input/featureengineering-titanic-procedure/fe_rfrage_scaled_data.csv')

print(df.info())


# # 1. logistics regression建模

# 准备数据

# In[8]:


feature_columns = 'Parch|Pclass|SibSp|Family_Survival|Sex_Code|Embarked_.*|Title_.*|Cabin_.*|Age_scaled|Fare_scaled'
df_data_x = df.filter(regex = feature_columns)
df_data_y = df['Survived']

df_train_x = df_data_x.iloc[:891, :]  # 前891个数据是训练集
df_train_y = df_data_y[:891]

df_test_x = df_data_x[891:]
df_test_output = df.iloc[891:, :][['PassengerId','Survived']]

df_data_x


# In[9]:


clf = linear_model.LogisticRegression(C=1.0, penalty='l1',tol=1e-6)
clf.fit(df_train_x,df_train_y)

clf


# In[10]:


df_test_output["Survived"] = clf.predict(df_test_x)
df_test_output['Survived'] = df_test_output['Survived'].astype(int)
df_test_output.to_csv('baseline_titanic_procedure.csv', index = False)


# In[ ]:


df_test_output

