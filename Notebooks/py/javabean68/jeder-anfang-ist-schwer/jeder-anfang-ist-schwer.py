#!/usr/bin/env python
# coding: utf-8

# My Notebook
# 

# In[ ]:


print('Hello World!')


# In[ ]:


import keras


# In[ ]:


import this


# In[ ]:


import pandas
titanic = pandas.read_csv("../input/train.csv")

titanic.head()

titanic.info()

print(titanic["Embarked"].value_counts())

print(titanic["Sex"].value_counts())

print(titanic.describe())


# In[ ]:


# Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version


# In[ ]:


if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

X, y = titanic.iloc[:, 2:].values, titanic.iloc[:, 1].values



X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


len(X_train),len(X_test)


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

