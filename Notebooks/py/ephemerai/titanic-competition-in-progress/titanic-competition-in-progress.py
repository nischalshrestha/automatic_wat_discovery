#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load in our libraries
import numpy as np
import pandas as pd

import seaborn as sns
sns.set_style('whitegrid')

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# machine learning
import xgboost as xgb
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)


# First of all, load the datasets.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
gender = pd.read_csv("../input/gender_submission.csv")


# Then, have a look at the datasets.

# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


test.head()


# In[ ]:


test.describe()


# In[ ]:


gender.head()


# In[ ]:


gender.describe()

