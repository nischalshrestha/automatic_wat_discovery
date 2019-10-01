#!/usr/bin/env python
# coding: utf-8

# <h1>Titanic: Machine Learning from Disaster</h1>

# **Table of Contents**
# 1. Define the Problem<br>
# 2. Gather the Data<br>
# 3. Exploratory data analysis<br>
# 4. Feature engineering<br>
# 5. Modelling<br>
# 6. Testing<br>

# <h1>1. Define the Problem</h1>

# <h1>2. Gather the Data</h1>

# data wrangling, such as data architecture, governance, and extraction

# <h1>3. Exploratory data analysis</h1>

# <h2> 3.2 Import Libraries</h2>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Libraries
#import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import sklearn as sk

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#show plots in Jupyter Notebook browser
get_ipython().magic(u'matplotlib inline')
#sns.set() # setting seaborn default for plots
#mpl.style.use('ggplot')
#sns.set_style('white')
#pylab.rcParams['figure.figsize'] = 12,8

from IPython.display import HTML, display
import tabulate

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

##import os
##print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## 3.1 Data Dictionaries

# In[ ]:


table = [["Variable","Definition","Key"],
         ["survival","Survival","0 = No, 1 = Yes"],
         ["pclass","Ticket class","1 = 1st, 2 = 2nd, 3 = 3rd"],
         ["sex","Sex",""],
         ["Age","Age in years",""],
         ["sibsp","# of siblings / spouses aboard the Titanic",""],
         ["parch","# of parents / children aboard the Titanic",""],
         ["ticket","Ticket number",""],
         ["fare","Passenger fare",""],
         ["cabin","Cabin number",""],
         ["embarked","Port of Embarkation","C = Cherbourg, Q = Queenstown, S = Southampton"]]
display(HTML(tabulate.tabulate(table, tablefmt='html')))


# #### Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[ ]:


print(df_train.shape)
print(df_test.shape)


# In[ ]:


print(df_train.info())
print('-' * 50)
print(df_test.info())


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


print(df_train.isnull().sum())
print('-' * 50)
print(df_test.isnull().sum())


# In[ ]:


corr = df_train.corr()
sns.heatmap(corr,
            annot=True,
            fmt='0.2f',
            cmap="Greens")


# In[ ]:


sns.countplot(x='Survived', data=df_train)


# In[ ]:


sns.countplot(x='Pclass', data=df_train)


# In[ ]:


sns.countplot(x="Sex", data=df_train)


# In[ ]:





# In[ ]:




