#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn import tree

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv("../input/train.csv")
train.shape


# Data Dictionary   
# Variable	Definition	Key   
# survival	Survival	0 = No, 1 = Yes   
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd   
# sex	Sex	   
# Age	Age in years   	
# sibsp	# of siblings / spouses aboard the Titanic   	
# parch	# of parents / children aboard the Titanic	   
# ticket	Ticket number	   
# fare	Passenger fare	   
# cabin	Cabin number	   
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton   
# Variable Notes   
# pclass: A proxy for socio-economic status (SES)   
# 1st = Upper   
# 2nd = Middle   
# 3rd = Lower   
#    
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5   
# sibsp: The dataset defines family relations in this way...   
# Sibling = brother, sister, stepbrother, stepsister   
# Spouse = husband, wife (mistresses and fiancés were ignored)   
# parch: The dataset defines family relations in this way...   
# Parent = mother, father   
# Child = daughter, son, stepdaughter, stepson   
# Some children travelled only with a nanny, therefore parch=0 for them.   

# In[ ]:


train.head()


# In[ ]:


train["Sex"] = pd.get_dummies(train["Sex"])['female']


# In[ ]:


train.head()


# In[ ]:


train["Embarked"] = train["Embarked"].map({'C':0,'Q':1,'S':2})


# In[ ]:


train["Embarked"] 


# In[ ]:


train.Embarked.unique()


# In[ ]:


train.Embarked.fillna(4).unique()


# In[ ]:


train["Embarked"] = train.Embarked.fillna(4)


# In[ ]:


train["Embarked"] = train.Embarked.astype('int')
train.head()


# In[ ]:


train.Age.unique()


# In[ ]:


train.Age.isnull().sum()


# In[ ]:


round(train.Age.mean(),3)


# In[ ]:


train['Age'] = train.Age.fillna(round(train.Age.mean(),2))


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.Cabin.value_counts()


# In[ ]:


train.dropna().get(['Survived','Cabin'])


# In[ ]:


train.get(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])


# In[ ]:


#リスト 6-2-(6)
#  ロジスティック回帰モデル -----------------
def logistic(x, w):
    y = 1 / (1 + np.exp(-((np.hstack((np.ones((x.shape[0],1)),x))).dot(w.T))))
    return y


# In[ ]:


#リスト 6-2-(9)
# 交差エントロピー誤差 ------------
def cee_logistic(w, x, t):
    X_n = x.shape[0]
    y = logistic(x, w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t.loc[n] * np.log(y[n]) +
                     (1 - t.loc[n]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee


# In[ ]:


# 交差エントロピー誤差の微分 ------------
def dcee_logistic(w, x, t):
    X_n=x.shape[0]
    y = logistic(x, w)
    dcee = np.zeros(8)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t.loc[n])
        for m, key in enumerate(X.keys()):
            dcee[m+1] = dcee[m+1] + (y[n] - t.loc[n]) * x[key][n]
    dcee = dcee / X_n
    return np.array(dcee)


# In[ ]:


def fit_logistic(w_init, x, t):
    res = minimize(cee_logistic, w_init, args=(x, t),
                   jac=dcee_logistic, method="CG")
    return res.x


# In[ ]:


W_init = [1, 1, 1, 1, 1, 1, 1, 1]
X = train.get(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
W = fit_logistic(W_init, X, train['Survived'])
print(W)
print('end')


# In[ ]:


W


# In[ ]:


test = pd.read_csv("../input/test.csv")
test.shape


# In[ ]:


test["Sex"] = pd.get_dummies(test["Sex"])['female']
test["Embarked"] = test["Embarked"].map({'C':0,'Q':1,'S':2})
test["Embarked"] = test.Embarked.fillna(4)
test["Embarked"] = test.Embarked.astype('int')
test['Age'] = test.Age.fillna(round(test.Age.mean(),2))
test_x = test.get(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
test.head()


# In[ ]:


test_Y = 1 / (1 + np.exp(-((np.hstack((np.ones((418,1)),test_x))).dot(W.T))))
test_Y


# In[ ]:


test_Y01 = np.where(test_Y > 0.5, 1, 0)
test_Y01


# In[ ]:


PassengerId = np.array(test["PassengerId"]).astype(int)
 
my_solution = pd.DataFrame(test_Y01, PassengerId, columns = ["Survived"])
 
my_solution.to_csv("my_test_Y.csv", index_label = ["PassengerId"])


# In[ ]:


print(os.listdir("."))


# # ライブラリで解く

# In[ ]:


X = train.get(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
y = train['Survived']


clf = LogisticRegression()
skf = StratifiedKFold(shuffle=True)
scoring = {
    'acc': 'accuracy',
    'auc': 'roc_auc',
}
scores = cross_validate(clf, X, y, cv=skf, scoring=scoring)

print('Accuracy (mean):', scores['test_acc'].mean())
print('AUC (mean):', scores['test_auc'].mean())


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

clf = LogisticRegression()
clf.fit(X, y)

print(clf.intercept_)
print(clf.coef_)


# In[ ]:


t_y = np.dot(clf.coef_,test_x.T) + clf.intercept_
t_y.shape


# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[ ]:


test_Y01 = np.where(sigmoid(t_y) > 0.5, 1, 0)
test_Y01.shape


# In[ ]:


PassengerId = np.array(test["PassengerId"]).astype(int)
 
my_solution = pd.DataFrame(test_Y01.T, PassengerId, columns = ["Survived"])
 
my_solution.to_csv("my_test_Y.csv", index_label = ["PassengerId"])


# In[ ]:




