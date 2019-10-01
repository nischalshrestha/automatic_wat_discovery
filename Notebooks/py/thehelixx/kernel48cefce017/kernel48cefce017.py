#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__)


# In[ ]:


import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()#For offline use


# In[ ]:


testdata = pd.read_csv('../input/test.csv')

traindata = pd.read_csv('../input/train.csv')
traindata.head()


#  **Converting training DF into format that we will use for analysis and droping data we don't require.**

# In[ ]:


traindf =traindata.drop(['PassengerId','Ticket','Name','Cabin'],axis=1)
traindf.head()


# In[ ]:


traindf.info()


# In[ ]:


traindf = traindf.fillna(traindf['Age'].mean()) 
traindf.head()


# **Lets take look at data and some patterns before going further**

# In[ ]:


sns.countplot('Sex',hue='Survived',data=traindf)


# In[ ]:


sns.countplot('Pclass',hue='Survived',data=traindf)


# In[ ]:


sns.countplot(x='SibSp',hue='Survived',data=traindf)


# In[ ]:


sns.countplot(x='Parch',hue='Survived',data=traindf)


# **converting text into integers**

# In[ ]:


trainfinal=traindf[['Pclass','Sex','SibSp','Parch']]
mapping={'male': 1,'female': 2}
trainfinal.Sex = [mapping[item] for item in trainfinal.Sex]
trainfinal =trainfinal.replace({'SibSp':[5,8]},0)

trainfinal.head()


# **NAIVE BAYES**
# 
#     X is array created using 'trainfinal' dataframe. It contains info. about class ,siblings,perents and sex. **X** is used as a **PREDICTOR** .**Y** array contains **results**. Y is used for train and then testing the accuracy

# In[ ]:


X= np.array(trainfinal[['Pclass','Sex','SibSp','Parch']])

Y = np.array(traindf['Survived'])

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.22, random_state = 0)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

y_pred= clf.predict(x_val)
acc_gaussianFinal = round(accuracy_score(y_pred, y_val) * 100, 2)#finding accuracy for GNB
print(acc_gaussianFinal)


# In[ ]:




