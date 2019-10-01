#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting

# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/train.csv")
data.head()


# In[ ]:


data.shape


# In[ ]:


X=data.loc[:,"Pclass":]
X.shape
y=data["Survived"]
y.shape


# In[ ]:


# Select only the numeric and categorical columns
X=data.loc[:,["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
X.shape


# In[ ]:


# convert categorical to "dummy" and then fill NaN
X=pd.get_dummies(X)
X = X.fillna(X.mean())
X.head()


# In[ ]:


y=data["Survived"]
y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


# Let's try a  decision tree
from sklearn.tree import DecisionTreeClassifier

DTClf = DecisionTreeClassifier()
DTClf = DTClf.fit(X_train,y_train)
DTClf.score(X_test,y_test)


# In[ ]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test,DTClf.predict(X_test))
metrics.auc(fpr, tpr)


# In[ ]:


plt.figure()
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

