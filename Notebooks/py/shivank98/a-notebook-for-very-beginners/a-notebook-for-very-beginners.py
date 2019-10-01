#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv('../input/train.csv', dtype={'Age': np.float64},)
test = pd.read_csv('../input/test.csv', dtype = {'Age':np.float64},)


# In[4]:


data['Age'] = data['Age'].fillna(data['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())


# In[5]:


predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


# In[6]:


data.loc[data['Sex'] == 'male', 'Sex'] = 0
data.loc[data['Sex'] == 'female', 'Sex'] = 1
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test.loc[test['Sex'] == 'male', 'Sex'] = 0
test.loc[test['Sex'] == 'female', 'Sex'] = 1
data["Embarked"] = data["Embarked"].fillna("S")
data.loc[data["Embarked"] == "S", "Embarked"] = 0
data.loc[data["Embarked"] == "C", "Embarked"] = 1
data.loc[data["Embarked"] == "Q", "Embarked"] = 2

test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation


# In[8]:


alg = LogisticRegression(random_state = 1)

scores = cross_validation.cross_val_score(
    alg,
    data[predictors],
    data["Survived"],
    cv=3
)
print(scores.mean())


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation


# In[10]:


alg = RandomForestClassifier(random_state=1, n_estimators = 150, min_samples_split=4, min_samples_leaf=2)


# In[11]:


scores = cross_validation.cross_val_score(alg, data[predictors], data['Survived'], cv = 3)


# In[12]:


print(scores.mean())


# In[13]:


alg.fit(data[predictors], data['Survived'])


# In[14]:


predictions = alg.predict(test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the data set
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })


# In[15]:


submission.to_csv("file_name.csv", index=False)


# In[16]:


scores = cross_validation.cross_val_score(alg, data[predictors], data['Survived'], cv = 3)
print(scores.mean())


# In[ ]:




