#!/usr/bin/env python
# coding: utf-8

# #  Overview
# 
#  The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
#  One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
#  
#  So, this is the analysis of data as well as try to use machine learning algorithms like SVM, Random Forest with different conditions to relate them and to learn from them.

# # Importing modules

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization
get_ipython().magic(u'matplotlib inline')
import seaborn as sns # visualization
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import collections
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Data Extraction

# In[ ]:


# styling 
sns.set(style="ticks", color_codes=True)

################# data extraction ####################
# training data
train_data = pd.read_csv('../input/train.csv')
#testing data
test_data = pd.read_csv('../input/test.csv')

check_data = pd.read_csv('../input/gender_submission.csv')
check_data_values = check_data.values


# In[ ]:


#train_data.head()
test_data.head()


# In[ ]:


#Seperating labels
train_y = train_data.iloc[:,1]
#Dropped unnecessary columns
train_data = train_data.drop(["Survived", "Name", "PassengerId", "Cabin", "Ticket"], axis = 1)
test_data = test_data.drop(["Name", "PassengerId", "Cabin", "Ticket"], axis = 1)
test_data


# In[ ]:


train_data["Sex"].replace(["male","female"], [1,0], inplace=True)
train_data["Embarked"].replace(["S","C","Q"], [1,0,-1], inplace = True)

test_data["Sex"].replace(["male","female"], [1,0], inplace=True)
test_data["Embarked"].replace(["S","C","Q"], [1,0,-1], inplace = True)


# In[ ]:


train_data = train_data.fillna(0)
test_data = test_data.fillna(0)


# In[ ]:


#Normalization
train_data["Age"] = train_data["Age"]/np.mean(train_data["Age"])
train_data["Fare"] = train_data["Fare"]/np.mean(train_data["Fare"])

test_data["Age"] = test_data["Age"]/np.mean(test_data["Age"])
test_data["Fare"] = test_data["Fare"]/np.mean(test_data["Fare"])


# In[ ]:


train_data = train_data.values
test_data = test_data.values


# In[ ]:


print(train_data)
train_y = train_y.values


# In[ ]:


# np.isnan(train_y).any()
np.isnan(train_data).any()


# In[ ]:


train_x, val_x, y, val_y = train_test_split(train_data, train_y, shuffle=True, test_size = 0.1)


# In[ ]:


print(train_x.shape, y.shape)
print(val_x.shape, val_y.shape)
print(train_x)


# In[ ]:


def create_svm(kernel = "linear", c = 1.0, gamma = "auto"):
    clf = svm.SVC(kernel= kernel, C = c, gamma = gamma) 
    clf.fit(train_x, y)
    pred = clf.predict(val_x)
    print("Accuracy Score for SVM = ", accuracy_score(val_y, pred))
    return pred


# In[ ]:


pred1 = create_svm()

pred2 = create_svm(kernel = "rbf")

pred3 = create_svm(kernel = "linear", c = 1.6)

pred = create_svm(kernel = "rbf", c = 1.5, gamma = 1.0)
pred4 = create_svm(kernel = "sigmoid", c = 1.5, gamma = 2)


# In[ ]:


clf = GaussianNB()
clf.fit(train_x, y)
print(clf)
pred4 = clf.predict(val_x)
print(accuracy_score(val_y, pred))


# In[ ]:




