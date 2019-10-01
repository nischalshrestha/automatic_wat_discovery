#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train.head(5)


# In[ ]:


train.info()


# **surviving rate per sex**

# In[ ]:


{'female' : train[train["Sex"]=="female"]["Survived"].mean(), 'male' : train[train["Sex"]=="male"]["Survived"].mean()}


# **surviving rate per Pcalss**

# In[ ]:


{'1' : train[train["Pclass"]==1]["Survived"].mean(), '2' : train[train["Pclass"]==2]["Survived"].mean() , '3' : train[train["Pclass"]==3]["Survived"].mean()}


# In[ ]:


train.groupby([ "Pclass", "Sex"])["Survived"].count()


# In[ ]:


fille_3 = train[(train["Sex"]=="female") & (train["Pclass"]==3)]


# In[ ]:


fille_3[fille_3["Age"].isnull()]["Survived"].count()


# In[ ]:


train["Name"].head(10)


# In[ ]:


a = "mouad, elaaboudi, Mr"


# In[ ]:


name = train['Name']


# In[ ]:


train["family"] = name.apply(lambda x: x[:x.index(",")])


# In[ ]:


train.info()


# In[ ]:


families = train.groupby("family").mean()


# In[ ]:


families


# In[ ]:




