#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.



# In[ ]:


import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../input/train.csv', header=0)


df.head(3)


# In[ ]:


type(df)


# In[ ]:


df[ ['Sex', 'Pclass', 'Age'] ]


# In[ ]:


df[df['Age'] > 60]


# In[ ]:


df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]


# In[ ]:


df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]


# In[ ]:


import pylab as P
df['Age'].hist()
P.show()


# In[ ]:




