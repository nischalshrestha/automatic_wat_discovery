#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[ ]:


import pandas as pd


# In[ ]:


import seaborn as sns


# # Reading data sets

# In[ ]:


dftrain = pd.read_csv('../input/train.csv')


# In[ ]:


dftrain.head()


# In[ ]:


dftrain.shape


# In[ ]:


dfgendersub = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


dfgendersub.head()


# In[ ]:


dfgendersub.shape


# In[ ]:


dftest = pd.read_csv('../input/test.csv')


# In[ ]:


dftest.head()


# In[ ]:


dftest.shape


# ## Exploring data: nulls

# In[ ]:


dftrain.isnull().sum()


# We don't know the age of 177 passengers. If we are going to use this feature, we need find a good way to estimate the age of those passengers

# In[ ]:


177/891


# This represents 19.86 % of all passengers

# In[ ]:


dftest.isnull().sum()


# In[ ]:




