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


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

titanic_data = pd.read_csv("../input/train.csv")

X= titanic_data.iloc[:,[2,4,5,6,7,9,10]]
y= titanic_data.iloc[:,1]

X["Sex"] = X["Sex"].apply(lambda x:1 if x == "male" else 0)
X["Cabin"]=X["Cabin"].fillna(0)
X["Cabin"] = X["Cabin"].apply(lambda x:0 if x ==0  else 1)
X["Age"]=X["Age"].fillna(np.mean(X["Age"]))

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

classifier = XGBClassifier()

classifier = classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print(np.mean(y_pred==y_test))

print(confusion_matrix(y_pred,y_test))




