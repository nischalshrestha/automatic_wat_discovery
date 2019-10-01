#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


# In[ ]:


train = "../input/train.csv"
test = "../input/test.csv"
data = pd.read_csv(train)
datacv = pd.read_csv(test)
datacv.columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']


# In[ ]:


for i in range(0,len(data["Name"])):
    name = data.ix[i,"Name"]
    if "Mr." in str(name):
        data.set_value(i, 'Name', "Mr")
    elif "Miss." in str(name):
        data.set_value(i, 'Name', "Miss")
    elif "Mrs." in str(name):
        data.set_value(i, 'Name', "Mrs")
    elif "Master." in str(name):
        data.set_value(i, 'Name', "Master")
    elif "Dr." in str(name):
        data.set_value(i, 'Name', "Dr")
    elif "Rev." in str(name):
        data.set_value(i, 'Name', "Rev")
    elif "Mme." in str(name):
        data.set_value(i, 'Name', "Mme")
    elif "Ms." in str(name):
        data.set_value(i, 'Name', "Ms")
    elif "Major." in str(name):
        data.set_value(i, 'Name', "Major")
    elif "Col." in str(name):
        data.set_value(i, 'Name', "Col")
    elif "Mlle." in str(name):
        data.set_value(i, 'Name', "Mlle")
    elif "Countess." in str(name):
        data.set_value(i, 'Name', "Countess")
    elif "Jonkheer." in str(name):
        data.set_value(i, 'Name', "Jonkheer")
    elif "Capt." in str(name):
        data.set_value(i, 'Name', "Capt")
    elif "Sir." in str(name):
        data.set_value(i, 'Name', "Sir")
    elif "Don." in str(name):
        data.set_value(i, 'Name', "Don")
    elif "Dona." in str(name):
        data.set_value(i, 'Name', "Dona")
    elif "Lady." in str(name):
        data.set_value(i, 'Name', "Lady")


# In[ ]:


for i in range(0,len(datacv["Name"])):
    name = datacv.ix[i,"Name"]
    if "Mr." in str(name):
        datacv.set_value(i, 'Name', "Mr")
    elif "Miss." in str(name):
        datacv.set_value(i, 'Name', "Miss")
    elif "Mrs." in str(name):
        datacv.set_value(i, 'Name', "Mrs")
    elif "Master." in str(name):
        datacv.set_value(i, 'Name', "Master")
    elif "Dr." in str(name):
        datacv.set_value(i, 'Name', "Dr")
    elif "Rev." in str(name):
        datacv.set_value(i, 'Name', "Rev")
    elif "Mme." in str(name):
        datacv.set_value(i, 'Name', "Mme")
    elif "Ms." in str(name):
        datacv.set_value(i, 'Name', "Ms")
    elif "Major." in str(name):
        datacv.set_value(i, 'Name', "Major")
    elif "Col." in str(name):
        datacv.set_value(i, 'Name', "Col")
    elif "Mlle." in str(name):
        datacv.set_value(i, 'Name', "Mlle")
    elif "Countess." in str(name):
        datacv.set_value(i, 'Name', "Countess")
    elif "Jonkheer." in str(name):
        datacv.set_value(i, 'Name', "Jonkheer")
    elif "Capt." in str(name):
        datacv.set_value(i, 'Name', "Capt")
    elif "Sir." in str(name):
        datacv.set_value(i, 'Name', "Sir")
    elif "Don." in str(name):
        datacv.set_value(i, 'Name', "Don")
    elif "Dona." in str(name):
        datacv.set_value(i, 'Name', "Dona")
    elif "Lady." in str(name):
        datacv.set_value(i, 'Name', "Lady")


# In[ ]:


data['Name'] = data['Name'].map({'Mr' : 1, 'Miss' : 2, 'Mrs' : 3, 'Master' : 4, 'Dr' : 5, 'Rev' : 6, 'Mme' : 7,
                                 'Ms' : 8, 'Major': 9, 'Col' : 10, 'Mlle' : 11, 'Countess' : 12, 'Jonkheer' : 13,
                                'Capt': 14, 'Sir' : 15, 'Don' : 16, 'Dona' : 17, 'Lady' : 18})


# In[ ]:


datacv['Name'] = datacv['Name'].map({'Mr' : 1, 'Miss' : 2, 'Mrs' : 3, 'Master' : 4, 'Dr' : 5, 'Rev' : 6, 'Mme' : 7,
                                 'Ms' : 8, 'Major': 9, 'Col' : 10, 'Mlle' : 11, 'Countess' : 12, 'Jonkheer' : 13,
                                'Capt': 14, 'Sir' : 15, 'Don' : 16, 'Dona' : 17, 'Lady' : 18})


# In[ ]:


data['Sex'] = data['Sex'].map({'male': 1, 'female': 2})
datacv['Sex'] = datacv['Sex'].map({'male': 1, 'female': 2})


# In[ ]:


data["Embarked"] = data["Embarked"].fillna('S')


# In[ ]:


data['Embarked'] = data['Embarked'].map({'C': 3, 'Q': 2, 'S': 1})
datacv['Embarked'] = datacv['Embarked'].map({'C': 3, 'Q': 2, 'S': 1})


# There is only 2 value is NaN. For the sake of non-NaN values we will assign them as 1.

# In[ ]:


data["Age"] = data["Age"].fillna(data["Age"].mean())
datacv["Age"] = datacv["Age"].fillna(datacv["Age"].mean())


# In[ ]:


datacv["Fare"] = datacv["Fare"].fillna(datacv["Fare"].mean())


# In[ ]:


data = data.drop("Cabin", axis=1)
datacv = datacv.drop("Cabin", axis=1)


# In[ ]:


data = data.drop("Ticket", axis=1)
datacv = datacv.drop("Ticket", axis=1)
data = data.drop("PassengerId", axis=1)
datacv = datacv.drop("PassengerId", axis=1)


# In[ ]:


Y = data["Survived"]
X_train = data.iloc[:,1:9]


# In[ ]:


X_test = datacv.iloc[:,0:8]


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y)
final_res = logreg.predict(X_test)


# In[ ]:


logreg = LogisticRegression()
scores_accuracy = cross_val_score(logreg, X_train, Y, cv=10, scoring='accuracy')
scores_log_loss = cross_val_score(logreg, X_train, Y, cv=10, scoring='neg_log_loss')
scores_auc = cross_val_score(logreg, X_train, Y, cv=10, scoring='roc_auc')
print('K-fold cross-validation results:')
print(logreg.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())
print(logreg.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())
print(logreg.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())

