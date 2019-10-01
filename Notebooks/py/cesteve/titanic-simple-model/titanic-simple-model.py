#!/usr/bin/env python
# coding: utf-8

# Hi there,  <br>
# This notebook is still a work in progress,  would very much appreciate any comment or advice, thank you in advance :)

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
target = train["Survived"]
combine = train.drop(["Survived"], axis=1).append(test).drop(["PassengerId", "Ticket"], axis=1)
combine.head()


# In[ ]:


def extractDeck(x):
    if str(x) != "nan":
        return str(x)[0]
    else :
        return

combine["hasParents"]=combine["Parch"].apply(lambda x : (x>0)*1)
combine["hasSibs"]=combine["SibSp"].apply(lambda x : (x>0)*1)
combine["title"] = combine['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
combine["Deck"] = combine['Cabin'].apply(extractDeck)
combine.drop(["Parch", "SibSp", "Cabin", "Name"], axis=1)


# In[ ]:


combine["Pclass"]=combine["Pclass"].astype("str")
treated = pd.get_dummies(combine)


# In[ ]:


for c in treated.columns:
    if treated[c].count() < 2919:
        treated[c]=treated[c].fillna(treated[c].mean())
preprocessing.scale(treated)


# In[ ]:


nb = train.shape[0]
regr = BaggingClassifier(n_estimators=50)
regr.fit(treated[:nb], target)
preds = regr.predict(treated[nb:])
preds


# In[ ]:


regr.score(treated[:nb], target)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": preds
})

submission.to_csv('submission.csv', index=False)

