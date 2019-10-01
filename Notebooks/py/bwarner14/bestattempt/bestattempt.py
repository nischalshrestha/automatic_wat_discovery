#!/usr/bin/env python
# coding: utf-8

# # My best attempt at this titanic problem. I will attempt to fit a random forest classifier to determine who survived and who didnt

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


trainNoLbl = train.drop("Survived",axis = 1)
df = pd.concat([trainNoLbl,test])


# In[ ]:


df.shape
df.describe()


# In[ ]:


ttem

