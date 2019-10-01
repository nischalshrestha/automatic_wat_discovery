#!/usr/bin/env python
# coding: utf-8

# **LOADING**
# 
# Load libraries and files as pandas dataframe

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# **DISCOVER**
# 
# Describe the trainning dataset and show the first entries

# In[ ]:


print(train.describe(include="all"))
train.head()


# **DISCOVER**
# 
# Show the Age distribution

# In[ ]:


sns.distplot(train["Age"].dropna())
plt.show()


# **DISCOVER**
# 
# Check the correlation between columns

# In[ ]:


sns.heatmap(train.corr(), square=True)
plt.show()


# **PREPARE**
# 
# Prepare data to be use by the classifier:
# 
#  1. Drop some useless columns
#  2. Replace string data as numerical representation ("Sex" and "Embarked")
#  3. Fill empty cells in the "Age" column

# In[ ]:


X = train.drop(["Cabin", "Name", "Ticket", "Survived"], axis=1)
X["Sex"] = X["Sex"].replace(["male", "female"], [0, 1])
X["Embarked"] = X["Embarked"].replace(["S", "C", "Q"], [0, 1, 2])
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
imp.fit(X)
X = imp.transform(X)

y = train["Survived"]


# **PREPARE**
# 
# Make a test and a train part from the train set. It will allow us to test our accuracy

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# **COMPUTE**
# 
# Train a SVC model

# In[ ]:


clf = svm.SVC(kernel='linear').fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)


# **PREDICT**
# 
#  1. Transform the "test.csv" as the "tain.csv" one
#  2. Use the classifier to predict this set

# In[ ]:


X = test.drop(["Cabin", "Name", "Ticket"], axis=1)
X["Sex"] = X["Sex"].replace(["male", "female"], [0, 1])
X["Embarked"] = X["Embarked"].replace(["S", "C", "Q"], [0, 1, 2])
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
imp.fit(X)
X = imp.transform(X)
res_survived = clf.predict(X)


# **OUTPUT**
# 
# Manage the data and output as the expected format

# In[ ]:


res = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": res_survived
})
#res.to_csv("prediction.csv", index=False)

