#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <a href="https://en.wikipedia.org/wiki/Pandas_(software)"  target=_blank> Pandas </a> is a great tool for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. It is free software released under the three-clause BSD license.
# Pandas developer is [Wes McKinney](https://en.wikipedia.org/wiki/Wes_McKinney). 
# 
# ###### I just found out that he is an active [Stackoverflow](https://stackoverflow.com) contributer. see [Wes McKinney](https://stackoverflow.com/users/776560/wes-mckinney)

# # The Basics Blocks
# From exprience, those who struggle most with pandas normally rely on their strong loop/map/lambdas writing skills. these skills can be transfred given willingness and time
# 

# In[1]:


import pandas as pd #load the pandas module
import numpy as np  # numpy is a strong linear algebra module on top of which pandas is built


# ## Pandas Series `sr`
# A series is a one dimentsional data structure. it is also the building block of dataframe. i.e. a column in dataframe is series with an  index(key), a row is also a series. A series corresponds to a dictionary where the index values are the keys of the dictionary
# 
# ## Pandas DataFrame `df`
# A dataframe is a two-dimensional data structure, composed of a collection of pandas series. A DataFrame corresponds to a dictionary of dictionaries where the index values are the keys of the parent dictionary

# ## reading a dataset
# `pd.read_***` functions are used to read different types of data. I will cover these in depth later as they have excellent parameters that makes processing data while reading it very easy

# In[2]:


df  = pd.read_csv('../input/train.csv', #provide file name "relative to the notebook location"
                  index_col=0) # use the first column as an index


# In[11]:


get_ipython().magic(u'pinfo2 df.loc')


# In[ ]:


df.head()


# The table displayed above is a DataFrame(`df`) the `head` function display the first 5 elements of the df.  below is a list of basic functions that helps explore the content of a dataframe

# In[ ]:


#find the dataframe shape (rows,columns)
df.shape


# In[ ]:


#find the dataframe index i.e. (rows key), shape = row,
print('The index shape is:', df.index.shape)
df.index


# In[ ]:


# find a series index
sr = df['Pclass']
sr.index


# In[ ]:


#find the dataframe columns i.e. (columns key), shape = columns,
print('The columns shape is:', df.columns.shape)
df.columns


# In[ ]:


#find a series name 
sr = df['Pclass']
sr.name


# In[ ]:


#dataframe columns data types
df.dtypes


# In[ ]:


#series data type
sr = df['Age']
sr.dtype


# In[ ]:


#general df info
df.info()


# 
# 
# 
# To select a column from a dataframe i.e. a series the indexing operator `[]` is used
# #### Notice  
# the index of the series, it's the __index__ of the dataframe `df` and notice the __series name__, its  __column name__ value

# In[ ]:


#the options can be used to control the maximum number of elements displayed in IPython display
pd.options.display.max_rows = 10
view = df['Name']
              
view


# On the variable `view`. if you know SQL you'll be familiar with the term, if not, when you query a dataframe or a series, an instant of that query is create in memory for viewing purposes, unless you assign that memory a name i.e. variable assignemnt, the data will be lost (unless you have access to the globals, but that's another matter)

# To select a row the `loc[]` operator is used. e.g. to select the passenger with id = 550
# #### Notice:
# the __index__ of the series, it's the __column names__ of the dataframe `df` and notice the __series name__, its __index value__ we selected

# In[ ]:


view = df.loc[550]               
view


# To select a subset of rows and column you can also use `loc[]` using an iterator or boolean masking
# ```df.loc[index_selector, column_selector]```

# In[ ]:


view = df.loc[df.index<10,#index selector
       (df.columns.isin(['Name','Age'])) | (df.columns.str.startswith('P')) #column selector
      ] 
view


# The index selector `df.index<10` generates an `numpy.ndarray` of shape equal to df.shape[0], when combining logic conditions the use of paranthesis to seperate each condition is obligatory. all three basic logic operators can be used in combining the conditions i.e.
# <table>
# <tr><td>Operator</td><td>Description</td></tr>
# <tr><td>&</td><td>logic and </td></tr>
# <tr><td>|</td><td>logic or</td></tr>
# <tr><td>~</td><td>logic Not</td></tr>
# <tr><td>-</td><td>logic Not</td></tr>
# </table>

# The column selector `df.columns.str.startswith('P')` utlises one of the most powerful functionality of Pandas (I think its highly underused), i.e. pandas ability to use Python's build-in string funtions on a series. using the str operator

# ### String manipulation
# Suppose you're asked to split the column `Name` to  3 columns`['Title', 'First Name','Family Name']`

# In[ ]:


split_view = df['Name'].str.split(',')
df['Family Name'] = split_view.str[0]
split_view = split_view.str[1].str.split('.')
df['Title'] = split_view.str[0]
df['First Name'] = split_view.str[1].str.strip()
df.head()


# In[ ]:





# Here is another example
# to select the passengers with IDs `[1, 16, 190, 33]` and display only their `Name` and `Age`

# In[ ]:


view = df.loc[[1, 16, 190, 33],#index selectore
      ['Name','Age'] #column selector
      ]
view


# # To be Continued

# In[10]:


get_ipython().magic(u'pinfo2 df.loc')


# In[ ]:




