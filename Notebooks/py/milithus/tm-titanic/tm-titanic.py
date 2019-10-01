#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().magic(u'matplotlib inline')
sns.set(style="darkgrid")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
df = pd.read_csv('../input/train.csv')

# Any results you write to the current directory are saved as output.
df_s=df[['PassengerId','Survived','Pclass','Sex','Age']]
df_s=df_s.dropna(axis=0,how='any')


# *  **Tratar Arquivo**
# 
# Nessa etapa será feita uma analise para determinar quais as linha deveram ser apagadas ou preenchidas

# In[ ]:


df.info()


# 'Age' contem 714 valores preenchidos e 177 não preenchidos. Isso representa  19,86% estão sem registor. Será feita uma classificação para compreender quais são as caracteristicas dos passageiros sem idade

# In[ ]:


df_age = df[df["Age"].isnull()]
df_age


# In[ ]:


g =sns.FacetGrid(df_s, row="Sex", col="Survived", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "Age", color="steelblue", bins=bins)


# In[ ]:


g =sns.FacetGrid(df_s, row="Sex", col="Survived", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "Age", color="steelblue", bins=bins)


# In[ ]:





# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

np.random.seed(sum(map(ord, "distributions")))


h =df_s.groupby(['Survived'])['Sex']
x = np.random.normal(size=100)
sns.distplot(h,kde=False)


# In[ ]:


import seaborn as sns
sns.set(style="ticks")

# Load the example dataset for Anscombe's quartet
df = sns.load_dataset("anscombe.csv")

# Show the results of a linear regression within each dataset
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted", size=4,
           scatter_kws={"s": 50, "alpha": 1})

