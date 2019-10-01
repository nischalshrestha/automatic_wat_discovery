#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import re


# In[ ]:


# Load training data
titanic = pd.read_csv("../input/train.csv")


# In[ ]:


# Review input features
print("Shape of dataframe:", titanic.shape, "\n")
print("Column Headers:", list(titanic.columns.values), "\n")
print(titanic.describe())


# In[ ]:


# Find all the unique feature values
missing_values = []
nonumeric_values = []

for column in titanic:
    print ("*****************************************\n")
    print (column)
    uniq = titanic[column].unique()
    print (uniq)
    
    if (True in pd.isnull(uniq)):
        missing_values.append(column)
           
    for i in range (1, np.prod(uniq.shape)):
        if (re.match('nan', str(uniq[i]))):
            next
        if (re.search('[\D]', str(uniq[i]))):
            if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)'
, str(uniq[i]))):
                nonumeric_values.append(column)
                break
  
print ("Features with missing values: {}" .format(missing_values))
print ("Features with non-numeric values: {}" .format(nonumeric_values))


# # Notes about input features
# 
# ## Features with missing values:
# 
# ## Features with non-numeric values:
# 

# In[ ]:


# Visualize the features
plt.style.use('ggplot')

selected_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
titanic[selected_columns].hist()


# In[ ]:


# Handle Missing Data 


# In[ ]:


# Convert Non-Numeric Columns


# In[ ]:


# Use a simple model


# In[ ]:


# Use Cross Validation


# In[ ]:


# Run Diagnostics


# In[ ]:


# Make Predictions


# In[ ]:




