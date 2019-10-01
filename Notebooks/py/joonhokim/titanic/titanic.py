#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[ ]:


training_set = pd.read_csv ('../input/train.csv')
test_set = pd.read_csv ('../input/test.csv')
data = training_set.copy ()
predict = test_set.copy ()


# In[ ]:


predict['Survived'] = pd.Series (0, index=predict.index)


# In[ ]:


predict.loc[predict['Sex']=='male','Survived'] = 0
predict.loc[predict['Sex']=='female','Survived'] = 1


# In[ ]:


predict[['PassengerId','Survived']].to_csv('gender.csv', header=True, index=False)


# In[ ]:




