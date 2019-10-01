#!/usr/bin/env python
# coding: utf-8

# ![](https://www.aquare.la/wp-content/uploads/unimed-bh.png)

# In[ ]:


import matplotlib.pyplot as plt # visualização de dados
import seaborn as sns # visualização de dados
sns.set(color_codes=True)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


train.info(null_counts=True)


# In[ ]:


total = train['Survived'].value_counts()
percent = train['Survived'].value_counts(normalize=True)
data = [[x, y] for x,y in zip(total, percent)]
pd.DataFrame( data, columns=["Total", "Percent"], index=["Died", "Survived"])


# In[ ]:


# histogramas
plt.close()
train.hist(figsize=(15,20))
plt.plot()


# In[ ]:




