#!/usr/bin/env python
# coding: utf-8

# In[253]:


import numpy as np
import pandas as pd


# In[223]:


titanic = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")


# In[224]:


titanic.describe()


# In[225]:


titanic_test.describe()


# In[226]:


titanic.head()


# In[227]:


titanic.drop("PassengerId", axis = 1, inplace = True)
titanic.drop("Name", axis=1, inplace = True)
titanic.drop("Ticket", axis=1, inplace = True)
titanic.drop("Cabin", axis=1, inplace = True)
titanic.drop("Embarked", axis=1, inplace = True)

titanic_test.drop("PassengerId", axis = 1, inplace = True)
titanic_test.drop("Name", axis=1, inplace = True)
titanic_test.drop("Ticket", axis=1, inplace = True)
titanic_test.drop("Cabin", axis=1, inplace = True)
titanic_test.drop("Embarked", axis=1, inplace = True)


# In[228]:


titanic.head()


# In[229]:


column = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]
titanic = titanic.reindex(columns=column)


# In[230]:


titanic.Age.fillna(titanic.Age.mean(), inplace = True)
titanic_test.Age.fillna(titanic_test.Age.mean(), inplace = True)
titanic_test.Fare.fillna(titanic_test.Fare.mean(), inplace = True)
titanic.describe()


# In[231]:


titanic_test.describe()


# In[232]:


titanic_test.head()


# In[233]:


def f(s):
    if s == "male":
        return 0
    else:
        return 1
titanic["Sex"] =titanic.Sex.apply(f)       #apply rule/function f
titanic.head()


# In[234]:


titanic_test["Sex"] =titanic_test.Sex.apply(f)  
titanic_test.head()


# In[235]:


titanic.describe()


# In[236]:


titanic_test.describe()


# In[237]:


from sklearn import preprocessing


# In[238]:


titanic_whole = pd.concat([titanic, titanic_test])


# In[239]:


del titanic_whole['Survived']
titanic_whole.describe()


# In[240]:


titanic_scaled = pd.DataFrame(preprocessing.scale(titanic_whole))
titanic_scaled.describe()


# In[241]:


titanic_train_x = titanic_scaled.iloc[0:891,:]
titanic_test_x = titanic_scaled.iloc[891:1309,:]

titanic_train_y = titanic.iloc[:,6]

titanic_train_x = titanic_train_x.values
titanic_test_x = titanic_test_x.values
titanic_train_y = titanic_train_y.values


# In[242]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[243]:


clf = LogisticRegression()
clf.fit(titanic_train_x, titanic_train_y)

clf.score(titanic_train_x, titanic_train_y)


# In[244]:


output = clf.predict(titanic_test_x)


# In[245]:


df = pd.DataFrame(output)


# In[246]:


titanic_test_df = pd.read_csv("../input/test.csv")
titanic_test_df.head()


# In[247]:


df["PassengerId"] = titanic_test_df["PassengerId"]
df.head()


# In[248]:


df.columns = ["Survived", "PassengerId"]


# In[249]:


df.head()


# In[250]:


result = df.reindex(columns = ["PassengerId", "Survived"])


# In[251]:


result.head()


# In[252]:


result.to_csv("out.csv", header=True, index=False,  )

