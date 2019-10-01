#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import os


# 
#  After loading, merging, and preparing dataset, a familiar task is to compute group statistics or possibly pivot table for reporting or visualization purposes. We have flexible and high perfomance groupby fancility in pandas, it anabling you to slice and sum data set in natural way
# 
# ### In this notebook we will learn to 
# *   Split pandas dataframe in to pieces using one or more key
# *  Computing group summary statistics, like count, mean, or standard deviation, or a user-defined function
# 
# 
#  Some content in this notebook you can see in Python for Data Analysis book
# 
# 

# In[ ]:


df = pd.DataFrame({'key1' : ['a','a','b','b','a'],
                    'key2' :['one','two','one','two','one'] ,
                    'data1': np.random.rand(5),
                    'data2':np.random.rand(5)
                  })
df


# If you want to caculate mean, max ,min,  median , size of columm data1 by using groups lable from key1, there are a way to do that
# 

# In[ ]:


grouped_by_key1 = df['data1'].groupby(df['key1'])


# In[ ]:



print(grouped_by_key1.mean())
print(grouped_by_key1.max())
print(grouped_by_key1.median())
print(grouped_by_key1.size())
print(grouped_by_key1.min())


# In the same way, we can cumputes all column using groups both lable key1 and key.  You can pass column name as the group key

# In[ ]:


grouped = df.groupby(['key1','key2'])


# In[ ]:


print(grouped.mean())


# Using unstack() to view unikey pairs of the keys observed
# 

# In[ ]:


grouped.mean().unstack()


# Merge data and transform data

# In[ ]:


k1_means = df.groupby('key1').mean().add_prefix('mean_')
k1_means


# In[ ]:


pd.merge(df , k1_means, left_on='key1', right_index = True)


# Use transform to mean and demean grouped data

# In[ ]:


df.groupby('key2').transform(np.mean)


# In[ ]:


def demean(arr):
    return arr - arr.mean() 
df.groupby('key2').transform(demean)


# Reading Titanic dataset
# 

# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# Use apply function to selects top n rows with the largest values in particular column 

# In[ ]:


def top(data, n, column):
    return data.sort_index(by=column)[:n]

train.groupby('Sex').apply(top, n=5, column='Fare')


# In[ ]:


train.groupby('Embarked').apply(top, n=5, column='Fare')


# Thanks for reading!
