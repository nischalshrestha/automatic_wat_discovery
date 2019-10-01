#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import library
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# read and get csv file
# global_terrorism = pd.read_csv("../input/gtd/globalterrorismdb_0617dist.csv")
titanic = pd.read_csv("../input/titanic/train.csv")
# preview the data
# global_terrorism.head()
titanic.head()


# In[ ]:


# check data
titanic.info()


# In[ ]:


titanic.describe()

