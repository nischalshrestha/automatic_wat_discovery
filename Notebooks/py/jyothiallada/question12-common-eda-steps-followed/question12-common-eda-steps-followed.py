#!/usr/bin/env python
# coding: utf-8

# ***Common steps observed in EDA***
# 

# *** -Understanding the problem : Looking at each variable closely and understanding the importance of it***

# *** -Univariate Analysis : Focusing on only one i.e. the dependent variable to learn more about it.***

# *** -Bivariate Analysis : Finding the relationship between the dependent and independent variables***

#  ***-Data Cleaning : Handling missing values, outliers.***

#  ***-Data transformation and Presentation***

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# In[ ]:


msno.matrix(df)


# In[ ]:


def get_null_cnt(df):
    """Return pandas Series of null count encounteres in DataFrame, where index will represent df.columns"""
    null_cnt_series = df.isnull().sum()
    null_cnt_series.name = 'Null_Counts'
    return null_cnt_series

def plot_ann_barh(series, xlim=None, title=None, size=(12,6)):
    """Return axes for a barh chart from pandas Series"""
    #required imports
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    #setup default values when necessary
    if xlim == None: xlim=series.max()
    if title == None: 
        if series.name == None: title='Title is required'
        else: title=series.name
    
    #create barchart
    ax = series.plot(kind='barh', title=title, xlim=(0,xlim), figsize=size, grid=False)
    sns.despine(left=True)
    
    #add annotations
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+(xlim*0.01), i.get_y()+.38,                 str(i.get_width()), fontsize=10,
    color='dimgrey')
    
    #the invert will order the data as it is in the provided pandas Series
    plt.gca().invert_yaxis()
    
    return ax


# In[ ]:


pdb_null_cnt = get_null_cnt(df)
# plot series result
ax = plot_ann_barh(pdb_null_cnt, xlim=len(df), title='Count of Null values for each columns in DataFrame')


# **Displaying the statistics**

# In[ ]:


categorical = df.dtypes[df.dtypes == "object"].index
print(categorical)

df[categorical].describe()


# In[ ]:


print( df.describe() )


# **Check dimensions**

# In[ ]:


df.shape


# **Datatypes**

# In[ ]:


df.dtypes


# After looking at the data for the first time, consider the following questions:
# * Do we need all of the variables?
# * Should we transform any variables?
# * Are there any NA values, outliers or other strange values?
# * Should we create new variables?

# **Handling missing values**

# In[ ]:


df.hist(column='Age',    # Column to plot
                   figsize=(9,6),   # Plot size
                   bins=20)   


# In[ ]:


df.hist(column='Fare',    # Column to plot
                   figsize=(9,6),   # Plot size
                   bins=20)   


# In[ ]:


df.Age.fillna(value=df.Age.mean(), inplace=True)
df.Fare.fillna(value=df.Fare.mean(), inplace=True)
df.Embarked.fillna(value=(df.Embarked.value_counts().idxmax()), inplace=True)
df.Survived.fillna(value=-1, inplace=True) 


# In[ ]:


pdb_null_cnt = get_null_cnt(df)
# plot series result
ax = plot_ann_barh(pdb_null_cnt, xlim=len(df), title='Count of Null values for each columns in DataFrame')


# **Transformation of variables**

# In[ ]:


new_survived = pd.Categorical(df["Survived"])
new_survived = new_survived.rename_categories(["Died","Survived"])              

new_survived.describe()


# **Checking for outliers**
# : we have some uncommon values that are so far away from the typical value that the box appears squashed in the plot: this is a clear indication of outliers. 

# In[ ]:


df["Fare"].plot(kind="box",
                           figsize=(9,9))


# **Creating new variables**

# In[ ]:


df["Family"] = df["SibSp"] + df["Parch"]
df["Family"]
#familes with maximum members on board
most_family = np.where(df["Family"] == max(df["Family"]))

df.ix[most_family]


# 

# 
