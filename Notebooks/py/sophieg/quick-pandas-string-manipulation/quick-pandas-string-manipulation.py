#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Table of Contents
# <ol >
# <li> <a href="#re1">Replace a single value in a dataframe</a></li>
# <li> <a href="#re2">Replace items in a column based on a dictionary with old values as keys and new values as values</a></li>
# <li> <a href="#re3">Remove all digits from a columns string entries</a></li>
# <li> <a href="#re4">Remove everything after a character in a columns string entries</a></li>
#  </ol>
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


fresh_df  = pd.read_csv('../input/train.csv', #provide file name "relative to the notebook location"
                  index_col=0) # use the first column as an index
fresh_df.info()


# In[ ]:


fresh_df.head()


# ## 1. <a id="re1">Replace a single value in a dataframe</a>

# In[ ]:


#replace all zeros by -1
#create a copy
df = fresh_df.copy()
df = df.replace(0,-1)
df.head()


# ## 2. <a id="re2"> Replace items in a column based on a dictionary with old values as keys and new values as values</a>
# ### using apply

# In[ ]:


#refresh
df = fresh_df.copy()
def lookup(string):
    """
    input  a string 
    output is the string  if string not in dictionary dic
    d[string] if string in dictionary dic
    """
    if string in dic:
        return dic[string]
    return string
dic = {'Braund, Mr. Owen Harris': 'Mr Harris',
       'Allen, Mr. William Henry':'Mr Allen'}
df.Name = df.Name.apply(lookup)
df.head()


# ### Using replace

# In[ ]:


#fresh
df = fresh_df.copy()
dic = {'Braund, Mr. Owen Harris': 'Mr Harris',
       'Allen, Mr. William Henry':'Mr Allen'}
#syntax {'column name': dictionary of desired replacements}
df.replace({'Name':dic},inplace=True)
df.head()


# ### Using map

# In[ ]:


#fresh
df = fresh_df.copy()
dic = {'Braund, Mr. Owen Harris': 'Mr Harris',
       'Allen, Mr. William Henry':'Mr Allen'}

df['Name'] = df['Name'].map(lambda x: dic[x] if x in dic.keys() else x)
df.head()


# ## 3. <a id='re3'>Remove all digits from a column column's string entries</a>
# remove the all digits from the Ticket column

# ### without regex

# In[ ]:


#fresh
df = fresh_df.copy()
def removedigit(string):
    """
    input is a string 
    output is a string with no digits
    """
    return ''.join(ch for ch in string if not ch.isdigit())
df.Ticket = df.Ticket.apply(removedigit)
df.head()


# ### with regex

# In[ ]:


#refresh df
df = fresh_df.copy()
df.Ticket = df.Ticket.str.replace('[0-9]','')
df.head()


# ## 4. <a id="re4">Remove everything after a character in a columns string entries </a>
# Remove all characters after ',' from the Name Column

# ### Without regex and using apply

# In[ ]:


#refresh df
df = fresh_df.copy()
def removeAfterComma(string):
    """
    input is a string 
    output is a string with everything after comma removed
    """
    return string.split(',')[0].strip()
df.Name = df.Name.apply(removeAfterComma)
df.head()


# ### Using pandas string operations

# In[ ]:


#refresh df
df = fresh_df.copy()
df.Name = df.Name.str.split(',').str[0].str.strip()
df.Name.head()


# ### Using regex

# In[ ]:


#refresh df
df = fresh_df.copy()
df.Name = df.Name.str.replace('\,.*','').str.strip()
df.Name.head()


# In[ ]:




