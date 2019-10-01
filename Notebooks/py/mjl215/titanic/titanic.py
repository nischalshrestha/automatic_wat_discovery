#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sb
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')

sb.set(style="whitegrid")

import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[ ]:


titanic_train = pd.read_csv("../input/train.csv") #importing file
titanic_test = pd.read_csv("../input/test.csv") #importing file


# In[ ]:


titanic_train.head()


# In[ ]:


titanic_test.head()


# In[ ]:


titanic_train.info()


# In[ ]:


titanic_train.describe()


# In[ ]:





# In[ ]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
titanic_train.hist(bins=50, figsize=(20,15))
plt.show


# In[ ]:


corr_matrix = titanic_train.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# In[ ]:




