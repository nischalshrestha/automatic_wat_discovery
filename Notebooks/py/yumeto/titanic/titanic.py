#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
import pandas as pd
get_ipython().magic(u'matplotlib inline')
from ipywidgets import interact, FloatSlider

sns.set_palette("muted")


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


_ = train.drop("Cabin", axis=1)
_ = _.dropna()
dummy_sex = pd.get_dummies(_["Sex"], drop_first = True)
dummy_emb = pd.get_dummies(_["Embarked"])
dummy_pclass = pd.get_dummies(_["Pclass"])
_ = pd.merge(_, dummy_sex, left_index=True, right_index=True)
_ = pd.merge(_, dummy_emb, left_index=True, right_index=True)
_ = pd.merge(_, dummy_pclass, left_index=True, right_index=True)
_ = _.corr()
_ = _.abs()
plt.figure(figsize=(20, 10))
sns.heatmap(_, annot=True, square=True, cmap='gray')


# In[ ]:


sns.countplot('Survived',data=train,hue='Sex')


# In[ ]:


train[train['Fare'] ==0]


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.preprocessing import OneHotEncoder
import scipy as sp


# In[ ]:


model = RandomForestClassifier(n_estimators=500)


# In[ ]:


sex = (train.Sex == "male").values.reshape(-1, 1)
noFare = (train.Fare == 0.0).values.reshape(-1, 1)
fare = train.Fare.values.reshape(-1, 1)
age = train.Age.fillna(29.699118).values.reshape(-1, 1)
pclass_onehot = OneHotEncoder()
pclass = pclass_onehot.fit_transform(train.Pclass.values.reshape(-1, 1))


# In[ ]:


model.fit(sp.sparse.hstack([sex, age, pclass, fare]), train.Survived.values)


# In[ ]:


train_predict = model.predict(sp.sparse.hstack([sex, age, pclass, fare]))


# In[ ]:


np.mean(train_predict == train.Survived.values)


# In[ ]:


sex = (test.Sex == "male").values.reshape(-1, 1)
noFare = (test.Fare == 0.0).values.reshape(-1, 1)
fare = test.Fare.fillna(-0).values.reshape(-1, 1)
age = test.Age.fillna(29.699118).values.reshape(-1, 1)
pclass_onehot = OneHotEncoder()
pclass = pclass_onehot.fit_transform(test.Pclass.values.reshape(-1, 1))


# In[ ]:


test["Survived"] = model.predict(sp.sparse.hstack([sex, age, pclass, fare]))


# In[ ]:


test[["PassengerId", "Survived"]].to_csv("./submit.csv", index=False)


# In[ ]:




