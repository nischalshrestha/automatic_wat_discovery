#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import nltk
import csv
import json
import seaborn as sns
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier

print(os.listdir("../input"))
get_ipython().magic(u'matplotlib inline')


# In[ ]:


df = pd.read_csv("../input/train.csv")
print(df.shape)
df.head(10)
test = pd.read_csv("../input/test.csv")
print(test.sample(3))
pass_id = test.PassengerId


# In[ ]:


df.drop(['Cabin', 'Name', 'Fare'], axis=1, inplace=True)


# In[ ]:


pred = df.Survived.tolist()
df.drop(['Ticket', 'Survived', 'PassengerId'], axis=1, inplace=True)


# In[ ]:


n = LabelEncoder()
df['Sex'] = n.fit_transform(df['Sex'].astype('str'))
test['Sex'] = n.fit_transform(test['Sex'].astype('str'))
df['Embarked'] = n.fit_transform(df['Embarked'].astype('str'))
test['Embarked'] = n.fit_transform(test['Embarked'].astype('str'))


# In[ ]:


#DEALING WITH NAN's
m = df.Age.mean()
df.Age.fillna(m, inplace=True)
df.dtypes


# In[ ]:


clf = RandomForestClassifier(n_estimators=200, max_depth=3,
                             random_state=0)
clf.fit(df.values, pred)


# In[ ]:


m = test.Age.mean()
test.Age.fillna(m, inplace=True)
test.sample(10)


# In[ ]:


test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'], axis=1, inplace=True)


# In[ ]:


val = clf.predict(test.values)


# In[ ]:


c1 = pd.Series(pass_id)
c2 = pd.Series(list(val))


# In[ ]:


fd = pd.DataFrame({"PassengerId" :c1,"Survived" :c2})


# In[ ]:


fd.to_csv("Submissions1.csv", index=False)


# In[ ]:


## USING NAIVE BAYES!!
gnb = GaussianNB()
y_pred = gnb.fit(df.values, pred).predict(test.values)
c3 = pd.Series(y_pred)


# In[ ]:


fd2 = pd.DataFrame({"PassengerId":c1, "Survived":c3})


# In[ ]:


fd2.to_csv("NBSub.csv", index=False)


# In[ ]:


## CALCULATING MY OWN MODEL ACCURACY, BY SPLITTING THE TRAIN DATA INTO TRAIN AND TEST!!
n_tr = df[:600]
n_pred = pred[:600]
n_test = df[600:]
n_test_pred = pred[600:]


# In[ ]:


#Training our RF and NB model again!!
gnb1 = GaussianNB()
clf1 = RandomForestClassifier(n_estimators=200, max_depth=3,
                             random_state=0)
gnb1.fit(n_tr, n_pred)
clf1.fit(n_tr, n_pred)


# In[ ]:


gnb1.score(n_test, n_test_pred)


# In[ ]:


clf1.score(n_test, n_test_pred)


# **IMPLEMENTING XGBOOST AND OTHER BOOSTIING ALGORITHMS TO SEE HOW THE SCORE VARIES!!!**

# In[ ]:


my_model1 = XGBClassifier(n_estimators=1000)
my_model1.fit(n_tr, n_pred, verbose=True)
predictions = my_model1.predict(n_test)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, n_test_pred)))


# **XGBoost Prediction for the dataset!!!**

# In[ ]:


my_model = XGBClassifier(n_estimators=150, learning_rate=0.25)
my_model.fit(df.values, pred, verbose=True)


# In[ ]:


predictions = my_model.predict(test.values)
col = pd.Series(predictions)
final_df = pd.DataFrame({"PassengerId":c1, "Survived":col})
final_df.to_csv("XGBSub.csv", index=False)


# In[ ]:


final_df.sample(19)


# In[ ]:


gbc = GradientBoostingClassifier(n_estimators=150, learning_rate=1,
    max_depth=3, random_state=0).fit(df.values, pred)
x = gbc.predict(test.values)
c4 = pd.Series(list(x))
final_df = pd.DataFrame({"PassengerId":c1, "Survived":c4})
final_df.to_csv("GBMSub.csv", index=False)


# In[ ]:





# In[ ]:




