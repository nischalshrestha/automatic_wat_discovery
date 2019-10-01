#!/usr/bin/env python
# coding: utf-8

# I made this kernel for myself to organize my knowledge, so I don't guarantee the quality of kernel. If you want to see great kernels, I recommend other kernels：）  For example, [this kernel](https://www.kaggle.com/samsonqian/titanic-guide-with-sklearn-and-eda/notebook) is very helpful.

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


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# fill missing values

# In[ ]:


train["Name"] = train["Name"].str.extract(r"([a-zA-Z]+)\.")
test["Name"] = test["Name"].str.extract(r"([a-zA-Z]+)\.")


# In[ ]:


plt.figure(figsize=(12, 6))
sns.countplot(train["Name"], hue=train["Survived"])


# In[ ]:


plt.figure(figsize=(12,6))
sns.stripplot(train["Name"], train["Age"])


# In[ ]:


def arrangename(name):
    if name == "Mr" or name == "Mrs" or name == "Miss" or name == "Master":
        return name
    else:
        return "Others"


# In[ ]:


train["Name"] = train["Name"].apply(arrangename)
test["Name"] = test["Name"].apply(arrangename)


# In[ ]:


train.loc[(train["Name"]=="Mr")&(train["Age"].isnull()), "Age"] = train.loc[train["Name"]=="Mr","Age"].mean()
train.loc[(train["Name"]=="Mrs")&(train["Age"].isnull()), "Age"] = train.loc[train["Name"]=="Mrs","Age"].mean()
train.loc[(train["Name"]=="Miss")&(train["Age"].isnull()), "Age"] = train.loc[train["Name"]=="Miss","Age"].mean()
train.loc[(train["Name"]=="Master")&(train["Age"].isnull()), "Age"] = train.loc[train["Name"]=="Master","Age"].mean()
train.loc[(train["Name"]=="Others")&(train["Age"].isnull()), "Age"] = train["Age"].mean()

test.loc[(test["Name"]=="Mr")&(test["Age"].isnull()), "Age"] = train.loc[train["Name"]=="Mr","Age"].mean()
test.loc[(test["Name"]=="Mrs")&(test["Age"].isnull()), "Age"] = train.loc[train["Name"]=="Mrs","Age"].mean()
test.loc[(test["Name"]=="Miss")&(test["Age"].isnull()), "Age"] = train.loc[train["Name"]=="Miss","Age"].mean()
test.loc[(test["Name"]=="Master")&(test["Age"].isnull()), "Age"] = train.loc[train["Name"]=="Master","Age"].mean()
test.loc[(test["Name"]=="Others")&(test["Age"].isnull()), "Age"] = train["Age"].mean()


# In[ ]:


test["Fare"].fillna(train["Fare"].mean(), inplace=True)


# In[ ]:


train["Cabin"].unique()


# In[ ]:


train["Cabin"] = train["Cabin"].str.extract(r"([A-Z])")
test["Cabin"] = test["Cabin"].str.extract(r"([A-Z])")


# In[ ]:


sns.countplot(train["Cabin"], hue=train["Survived"])


# In[ ]:


def cabintoint(cabin):
    if cabin == "B" or cabin == "C" or cabin == "D" or cabin == "E":
        return 1
    else:
        return 0


# In[ ]:


train["Cabin"] = train["Cabin"].apply(cabintoint)
test["Cabin"] = test["Cabin"].apply(cabintoint)


# In[ ]:


train["Embarked"].value_counts()


# In[ ]:


train["Embarked"].fillna("S", inplace=True)


# Investigate and arrange all attributes in terms of "Survived"

# In[ ]:


train.drop("PassengerId", axis=1, inplace=True)


# In[ ]:


sns.countplot(train["Pclass"], hue=train["Survived"])


# In[ ]:


sns.countplot(train["Name"], hue=train["Survived"])


# In[ ]:


tmp = pd.get_dummies(train["Name"])
train.drop("Name", axis=1, inplace=True)
train = pd.concat((train, tmp), axis=1)
train.drop(["Others"], axis=1, inplace=True)

tmp = pd.get_dummies(test["Name"])
test.drop("Name", axis=1, inplace=True)
test = pd.concat((test, tmp), axis=1)
test.drop(["Others"], axis=1, inplace=True)


# In[ ]:


sns.countplot(train["Sex"], hue=train["Survived"])


# In[ ]:


train["Sex"].replace({"male": 0, "female": 1}, inplace=True)
test["Sex"].replace({"male": 0, "female": 1}, inplace=True)


# In[ ]:


sns.countplot(train["Age"].apply(lambda x: int(x//5*5)), hue=train["Survived"])


# In[ ]:


sns.countplot(train["SibSp"], hue=train["Survived"])


# In[ ]:


sns.countplot(train["Parch"], hue=train["Survived"])


# In[ ]:


train["FamSize"] = train["SibSp"] + train["Parch"] + 1
train["IsAlone"] = train["FamSize"].apply(lambda x: 0 if x==1 else 1)
train.drop(["SibSp", "Parch"], axis=1, inplace=True)

test["FamSize"] = test["SibSp"] + test["Parch"] + 1
test["IsAlone"] = test["FamSize"].apply(lambda x: 0 if x==1 else 1)
test.drop(["SibSp", "Parch"], axis=1, inplace=True)


# In[ ]:


train.drop("Ticket", axis=1, inplace=True)
test.drop("Ticket", axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(10, 6))
sns.countplot(train["Fare"].apply(lambda x: int(x//5*5)), hue=train["Survived"])


# In[ ]:


sns.countplot(train["Cabin"], hue=train["Survived"])


# In[ ]:


sns.countplot(train["Embarked"], hue=train["Survived"])


# In[ ]:


train.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)
test.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)


# arrange data to put them in models

# In[ ]:


X_train = train.drop("Survived", axis=1)
columns = X_train.columns
y_train = train["Survived"]
Id = test["PassengerId"]
test = test.drop("PassengerId", axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler

std = StandardScaler()
X_train = std.fit_transform(X_train)
test = std.transform(test)


# train models

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


# In[ ]:


def modeling(params, estimator):
    '''
    receive hyper paramaters and a model,
    execute GridSearchCV using 10 folds and print best hyper paramaters
    print accuracy about the model using the hyper paramater by testing validation data
    return accuracy and the model
    '''
    
    grid = GridSearchCV(estimator, params, scoring="accuracy", n_jobs=-1, cv=10)
    grid.fit(X_train, y_train)
    
    print("paramater:", grid.best_params_)
    print("accuracy:", grid.best_score_)
    
    return grid.best_score_, grid.best_estimator_


# In[ ]:


params = {"n_estimators": [10, 20, 25, 30],
         "max_depth": [3, 5, 7, 9, None],
         "max_features": ["auto", "sqrt", "log2", None]}

rfc_accuracy, rfc_clf = modeling(params, RandomForestClassifier())
importance = pd.DataFrame({"feature": columns, "importance": rfc_clf.feature_importances_})
importance.sort_values(by="importance", ascending=False)


# In[ ]:


params = {"C": [0.5, 1.0, 1.5],
         "gamma": [0.01, 0.05, 0.1],
         "probability": [True]}

svc_accuracy, svc_clf = modeling(params, SVC())


# In[ ]:


params =  {"C": [0.1, 1, 10],
          "max_iter": [50, 100, 200]}

lr_accuracy, lr_clf = modeling(params, LogisticRegression())


# In[ ]:


params = {"n_neighbors": [2, 3, 4, 5, 10, 15],
         "leaf_size": [20, 30, 50]}

knc_accuracy, knc_clf = modeling(params, KNeighborsClassifier())


# In[ ]:


params = {}

gnb_accuracy, gnb_clf = modeling(params, GaussianNB())


# In[ ]:


params = {"C": [0.005, 0.01, 0.5, 1.0]}
    
lsvc_accuracy, lsvc_clf = modeling(params, LinearSVC())


# In[ ]:


accuracy = pd.DataFrame({"model": ["RandomForestClassifier", "SVC", "LogisticRegression", "KNeighborsClassifier", "GaussianNB", "LinearSVC"],
                        "accuracy": [rfc_accuracy, svc_accuracy, lr_accuracy, knc_accuracy, gnb_accuracy, lsvc_accuracy]})
accuracy.sort_values(by="accuracy", ascending=False)


# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

vt_clf = VotingClassifier(estimators=[("svc", svc_clf), ("rfc", rfc_clf), ("lsvc", lsvc_clf), ("lr", lr_clf), ("knc", knc_clf)])
print("accuracy:", np.mean(cross_val_score(vt_clf, X_train, y_train, cv=5, n_jobs=-1)))


# In[ ]:


rfc_clf.fit(X_train, y_train)
submission_predictions = rfc_clf.predict(test)
submission = pd.DataFrame({"PassengerId": Id, "Survived": submission_predictions})
submission.to_csv("submission.csv", index=False)


# In[ ]:




