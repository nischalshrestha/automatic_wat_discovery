#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h2o
from h2o.automl import H2OAutoML
import pandas as pd


# In[2]:


h2o.init()


# In[3]:


df = h2o.import_file('../input/train.csv')


# In[4]:


df_test = h2o.import_file('../input/test.csv')


# In[17]:


df_test.head()


# In[6]:


train, test = df.split_frame(ratios=[.7])


# In[7]:


# Identify predictors and response
x = train.columns
y = "Survived"
x.remove(y)


# In[8]:


# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()


# In[9]:


# Run AutoML for 900 seconds
aml = H2OAutoML(max_runtime_secs = 900)
aml.train(x = x, y = y, training_frame = train, leaderboard_frame = test)


# In[10]:


# View the AutoML Leaderboard
aml.leaderboard
aml.leader


# In[11]:


preds = aml.leader.predict(df_test)


# In[12]:


pr = preds['predict']
df_pr = pr.as_data_frame()


# In[16]:


# Запись результата
result = df_test['PassengerId'].as_data_frame()
result.insert(1,'Survived', df_pr)
result.to_csv('Result.csv', index=False)

