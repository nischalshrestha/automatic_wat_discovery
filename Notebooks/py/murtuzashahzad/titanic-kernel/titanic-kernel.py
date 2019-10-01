#!/usr/bin/env python
# coding: utf-8

# **Introduction to Titanic Kernel**
# 
# This Kernel is an introduction to the basics of machine learning classification models.
# I will work on this dataset in 4 steps:
# 1. I will first analyse the dataset using basic Exploratory Data Analysis (EDA) methods. 
# 2. After the initial analysys on the dataset, I will perform feature analysis and see what features are best that could be used in applying Machine Learning models.
# 3. I will apply the various Machine Learning classification models. 
# 4. I will observe the classification results and see if hyperparameter tuning would help in improving the results.

# **Step 1: EDA methods to analyse the dataset**
# 
# We will use Python libraries: Numpy, Pandas and Matplotlib to do EDA.
# 
# Details of these libraries: 
# 
# Numpy: Numpy is a simple Python library that is used for performing operations on data in terms of arrays of objects.
# 
# Pandas: Pandas is a library used to read the data in a form of table having rows and columns. This library is very much useful in extracting the dataset in a tabular format which would be in a better readable form.
# 
# Matplotlib: Matplotlib is Python library for making graphical plots. This library can be used with data in order to visually understand the relations among different attributes/columns in the dataset.

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


#reading the training data
train = pd.read_csv('../input/train.csv')
traindataset = pd.DataFrame(train)
traindataset.head()


# In[ ]:




