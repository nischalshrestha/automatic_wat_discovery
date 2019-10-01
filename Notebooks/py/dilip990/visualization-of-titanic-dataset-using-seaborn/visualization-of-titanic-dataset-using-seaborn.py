#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Seaborn 
# 
# Visualize  Titanic Dataset using Seaborn

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic(u'matplotlib inline')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


titanic = pd.read_csv('../input/train.csv')


# In[ ]:


titanic.head()


# In[ ]:


sns.jointplot(x='Fare',y='Age',data=titanic)


# In[ ]:


sns.distplot(titanic['Fare'],bins=30,kde=False,color='red')


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=titanic,palette='rainbow')


# In[ ]:


sns.swarmplot(x='Pclass',y='Age',data=titanic,palette='Set2')


# In[ ]:


sns.countplot(x='Sex',data=titanic)


# In[ ]:


sns.heatmap(titanic.corr(),cmap='coolwarm')
plt.title('titanic.corr()')


# In[ ]:


g = sns.FacetGrid(data=titanic,col='Sex')
g.map(plt.hist,'Age')


# 
# 
# ### Thank You For Your Time Do Up Vote
