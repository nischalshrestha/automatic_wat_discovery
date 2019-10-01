#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# The data set for passengers aboard the Titanic:

# In[ ]:


passenger_data_filepath = '../input/titanic/train.csv'
passenger_data = pd.read_csv(passenger_data_filepath)
passenger_data.describe()


# In[ ]:


The information available on passengers is:


# In[ ]:


passenger_data.columns


# We check if there are any entries with missing data:

# In[ ]:


passenger_data.dropna(axis=0).describe()


# In[ ]:


Indeed, only 183 of the initial 891 entries have information on all the available variables.


# 

# 
