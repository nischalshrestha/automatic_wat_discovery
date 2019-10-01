#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import standard libraries
import numpy as np
import pandas as pd
import seaborn as sns
import re


# In[ ]:


titanic_data = pd.read_csv('../input/train.csv')


# In[ ]:


titanic_data.describe()


# In[ ]:


titanic_data.head()


# In[ ]:


# Pairplot
sns.pairplot(titanic_data[['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']]);


# In[ ]:


# Heatmap
sns.heatmap((titanic_data.loc[:, ['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare',]]).corr(),
            annot=True);

