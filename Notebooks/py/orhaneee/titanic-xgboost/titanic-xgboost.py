#!/usr/bin/env python
# coding: utf-8

# ## Titanic: Machine Learning from Disaster

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

get_ipython().magic(u'matplotlib inline')


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_submit = pd.read_csv('../input/gender_submission.csv')

df_train.info()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.info()


# In[ ]:


df_train.head(10)


# I mostly do not look data set a lot not to overfit model, but I think that there is absolutely correlation between Pclass and Survived.

# In[ ]:


df_train.groupby('Pclass')['Survived'].mean()


# In[ ]:


sns.countplot(x='Pclass', data=df_train, hue='Survived', palette='deep')
plt.show()


# ### Feature Engineering
# 
# Now, it is time to extract some useful data

# In[ ]:


df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())

df_train.loc[df_train["Sex"] == "male", "Sex"] = 0
df_test.loc[df_test["Sex"] == "male", "Sex"] = 0

df_train.loc[df_train["Sex"] == "female", "Sex"] = 1
df_test.loc[df_test["Sex"] == "female", "Sex"] = 1

df_train.isnull().sum()


# In[ ]:


df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_train = pd.get_dummies(df_train, columns=["Embarked"])
df_test = pd.get_dummies(df_test, columns=["Embarked"])

# Drop unnecessary features
df_train = df_train.drop(["Name", "PassengerId", "Cabin", "Ticket"], axis=1)
df_test = df_test.drop(["Name", "PassengerId", "Cabin", "Ticket"], axis=1)


# In[ ]:


X = df_train.drop(["Survived"], axis=1)
y = df_train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


xg = XGBClassifier(max_depth = 2, n_estimators = 400, learning_rate = 0.1)
xg.fit(X_train, y_train)

results = cross_validate(xg, X_train, y_train)
results['test_score']  

y_pred = xg.predict(X_test)
print(accuracy_score(y_test, y_pred))


# In[ ]:


plot_importance(xg)


# In[ ]:


cv_results = cross_validate(xg, X, y)
predictions = xg.predict(df_test)

df_submit["Survived"] = predictions
df_submit.to_csv("xgboost_submission.csv", index=False)

