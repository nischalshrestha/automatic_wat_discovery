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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


test= pd.read_csv("../input/test.csv")  #Loaded testing data
train=pd.read_csv("../input/test.csv") #Loaded training data


# Q1. Display the age range boarded on the ship.

# In[ ]:


genderclassmodel = pd.read_csv("../input/genderclassmodel.csv")
gendermodel = pd.read_csv("../input/gendermodel.csv")


# In[ ]:


result = pd.concat([train, genderclassmodel], axis=1, join='inner')
result


# In[ ]:


for Survived in result:
    new_train= result[result['Survived'] == 1]


# In[ ]:


new_train


# In[ ]:


new_train.to_csv


# In[ ]:




