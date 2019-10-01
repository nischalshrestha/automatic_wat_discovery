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


train=pd.read_csv("../input/train.csv")
print(train.columns)


# In[ ]:


test=pd.read_csv("../input/test.csv")


# In[ ]:


from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
data_train=train.copy()
var=["Pclass","Sex","Age"]
X=data_train[var]
Y=data_train.Survived
na_impute=Imputer()
X=pd.get_dummies(X)
X_impute=pd.DataFrame(na_impute.fit_transform(X))
X_impute.columns=X.columns
X_train,X_test,Y_train,Y_test=train_test_split(X_impute,Y,random_state=0)
clas=GradientBoostingClassifier()
clas.fit(X_train,Y_train)
print(mean_absolute_error(Y_test,clas.predict(X_test)))


# In[ ]:


X2=test[var]
X2=pd.get_dummies(X2)
X2_impute=pd.DataFrame(na_impute.fit_transform(X2))
X2_impute.columns=X2.columns
Y2=clas.predict(X2_impute)


# In[ ]:


Final=pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": Y2})
Final.to_csv("Submission.csv",index=False)

