#!/usr/bin/env python
# coding: utf-8

# ​

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import scipy
from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
filename = "../input/train.csv"
data = pd.read_csv(filename)
data = data.drop(['PassengerId','Name','Ticket'], axis=1)

# Any results you write to the current directory are saved as output.
sns.barplot(x='Sex', y='Survived', data=data)


# In[ ]:


sns.distplot(data['Age'].dropna().astype(int), bins = 20, norm_hist = False, kde = False)
survival = data[(data['Survived'] > 0)]


# In[ ]:


sns.distplot(data['Age'].dropna().astype(int), bins = 20, norm_hist = False, hist = False)
sns.distplot(survival['Age'].dropna().astype(int), bins = 20, norm_hist = False, color = 'g', hist = False)
scipy.stats.ks_2samp(data['Age'].dropna().astype(int), survival['Age'].dropna().astype(int))


# In[ ]:


#sns.countplot(x = 'SibSp', hue = 'Survived', data = data)
#sns.distplot(data['SibSp'].dropna(), kde = False)
#sns.distplot(survival['SibSp'].dropna(), bins = 20, kde = False, color = 'g')
SibSp = []
Survived = []
for sib in range(0, data["SibSp"].max(skipna = True)):
    cnt = 0 
    for i in range(0, data['SibSp'].size):
            if (data.at[i, 'SibSp'] == sib and data.at[i, 'Survived'] == 1 ):
                cnt = cnt + 1
    rate = cnt / data['SibSp'].size
    SibSp.append(sib)
    Survived.append(rate)
sns.jointplot(SibSp, Survived)

    



# In[ ]:





# In[ ]:





# In[ ]:




