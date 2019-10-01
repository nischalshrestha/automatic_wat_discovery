#!/usr/bin/env python
# coding: utf-8

# ## In this Kernel, I am exploring different Selection method available for Pandas DataFrame. 
# I am using Titanic data set for running all the selection methdos. This data can be downloaded from Kaggle Titanic competition page. 

# In[ ]:


# lets import pandas package
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


# reading the train file 
data = pd.read_csv('../input/train.csv')


# In[ ]:


# lets have a look at this dataset
data.head()


# ### Selecting a column
# 

# In[ ]:


# select the name column
name = data['Name']
type(name)
name.head()


# In[ ]:


# select the name column but result should be a dataframe
name= data[['Name']]
type(name)
name.head()


# ### Selecting multiple column

# In[ ]:


# select name and sex of the passengers
name = data[['Name', 'Sex']]
name.head()


# ### Making selection based on condition on any column

# In[ ]:


# select all the record of male passengers
male_record = data[(data['Sex']=='male')]
male_record.head()


# In[ ]:


# select all the rows where passengers is male and is 20 years or older
male_record = data[(data['Sex']=='male')&(data['Age']>= 20)]
male_record.head()


# In[ ]:


# select all the rows where age is null
age_null = data[(data['Age'].isnull())]
age_null.head()


# In[ ]:


# select all the rows where cabin is not null
cabin_value = data[(data['Cabin'].notnull())]
cabin_value.head()


# ### Exploring df.iloc method

# Selecting using Pandas iloc command :- df.iloc is used to select rows and columns by the order in which they appear in the data frame. 
#     You can find total number of rows present in any dataframe by using df.shape[0]. 
#     Df.iloc take two arguments. One argument is to specify rows and another is to specify columns. 
#     
# 

# In[ ]:


data.head()


# In[ ]:


# Selecting any row using df.iloc
# select the first row 
data.iloc[0]
# first row can be accessed using 0 


# In[ ]:


# select the last row 
data.iloc[-1]


# In[ ]:


data.iloc[[0]]


# In[ ]:


# select top 10 rows
data.iloc[0:11]


# In[ ]:


# select first second, fourth and tenth rows
data.iloc[[0,1,3,9]]


# In[ ]:


# select first column

data.iloc[:,0]


# In[ ]:


# select starting three columns

data.iloc[:,0:3]


# In[ ]:


# select first, thrid and fifth columns
data.iloc[:,[0,2,4]]


# In[ ]:


# we can access any particluar cell using .iloc
data.iloc[0,0]


# In[ ]:


# Select 4th to 6th column in the first row
data.iloc[0,3:6]


# In[ ]:


# select 4th to 6th column for the first three row
data.iloc[0:3,3:6]


# In[ ]:


# select 2nd, 4th and 7th columns for the first three row
data.iloc[0:3,[1,3,6]]


# In[ ]:


## select 2nd, 4th and 7th columns for the first, three and fifth row
data.iloc[[0,2,4],[1,3,6]]


# In[ ]:


# df.iloc will return a pandas series if you are selecting only one row. It will return a pandas dataframe when multiple rows are selected.
type(data.iloc[0])
type(data.iloc[[0]])


# In[ ]:


type(data.iloc[:,0])
type(data.iloc[:,[0]])


# In[ ]:


data.head()


# In[ ]:


# we cannot use column name here 
# data.iloc[0,'Sex']
# above will throw an error


# ### Exploring df.loc method
# 

# In[ ]:


# select the rows where index = 0
data.loc[0]


# In[ ]:


# select the rows where index=0 and get the output in a dataframe
data.loc[[0]]


# In[ ]:


# select all the rows where index is less than or equal to 10 
data.loc[0:11]


# In[ ]:


# select all the rows where index is equal to 0, 5, 10, 20, 50
data.loc[[0,5,10,20,50]]


# In[ ]:


# select the cell where index = 0 and column = Name
data.loc[0,'Name']


# In[ ]:


# select the cell where index = 0 and columns are Name, Sex, Ticket
data.loc[0,['Name', 'Sex', 'Ticket']]


# In[ ]:


# select the cell where index in in the range from 0 to 11 and columns are Name, Sex, Ticket
data.loc[0:11,['Name', 'Sex', 'Ticket']]


# In[ ]:


# select all the data where index is equal to 0,5,10,20,50 and column names are Name, Sex, Ticket
data.loc[[0,5,10,20,50], ['Name', 'Sex', 'Ticket']]


# In[ ]:


# select all the data where index is in the range of 0 to 11 and all the columns from name to ticket
data.loc[0:11,'Name':'Ticket']

