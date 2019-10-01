#!/usr/bin/env python
# coding: utf-8

# **EDA**
# 
# EDA stands for Explanatory Data Analysis, EDA/Data cleaning is the infrastructure and the first block in data science, EDA/Data cleaning usually takes approximately 80% of your time when you analyzing data and the modeling process takes "only" 20%. Before we don any modeling we need to make sure our data is clean and credible.

# **Importing libraries and the data**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')

import os
print(os.listdir('../input/'))


# In[ ]:


titanic = pd.read_csv('../input/train.csv')


# Lets read the data description in order to understand what each column means. Also a good practice is a look at our dta to get a clear picture about our data. 

# In[ ]:


titanic.head()


# Now, let check what kind of data we have in dataset.

# In[ ]:


titanic.dtypes


# One of the most important features in pandas is .isnull().sum() which gives us the sum of the all the null values for each column.

# In[ ]:


titanic.isnull().sum()


# Another usefull pandas function is .info() to get both types and the sum of the null values.

# In[ ]:


titanic.info()


# To start with some statistical report we use .describe() 

# In[ ]:


titanic.describe()


# The correlation matrix is very helpful when understanding the relationship between the numerical columns, .corr() will do taht for us.

# In[ ]:


titanic.corr()


# We can also use heat map from seaborn library to see correlation as before, but now in more visible output.

# In[ ]:


plt.figure(figsize=(18,12))
sns.heatmap(titanic.corr(), annot=True)

