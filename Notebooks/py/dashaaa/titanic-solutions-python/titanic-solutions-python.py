#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import statistics as st
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn import preprocessing
import seaborn as sns
import matplotlib as plt
import pylab
import re


# In[ ]:


gender=pd.read_csv('../input/datacamptraining/my_solution.csv')
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train.isnull().sum()


# In[ ]:


gender.isnull().values.any()


# In[ ]:


get_ipython().magic(u'matplotlib inline')
pl=train.pivot_table('PassengerId','Pclass', 'Survived', 'count').plot(kind='bar')


# In[ ]:


pl1=train.pivot_table('PassengerId','Sex', 'Survived', 'count').plot(kind='bar')


# In[ ]:


pl2=train.pivot_table('PassengerId','SibSp', aggfunc='count').plot(kind='bar')


# In[ ]:


pl3=train.pivot_table('PassengerId','Embarked', aggfunc='count').plot(kind='bar')


# In[ ]:


train['Embarked'].fillna('S', inplace=True)


# In[ ]:


a=train['Fare']==0
a.sum()


# In[ ]:


d=train[['Pclass','Fare']]
d_test=test[['Pclass','Fare']]
fc=d.groupby(['Pclass']).mean()
fc_test=d_test.groupby(['Pclass']).mean()
fc


# In[ ]:


fc_test


# In[ ]:


fc_test


# In[ ]:


a = (train['Pclass'] == 1) & (train['Fare'] == 0)
df=train[a]
df['Fare']=86.147806
train[a]=df
b = (train['Pclass'] == 2) & (train['Fare'] == 0)
df1=train[b]
df1['Fare']=21.357921
train[b]=df1
c = (train['Pclass'] == 3) & (train['Fare'] == 0)
df2=train[c]
df2['Fare']=13.787867
train[c]=df2


# In[ ]:


a_test = (test['Pclass'] == 1) & (test['Fare'] == 0)
df_test=test[a_test]
df_test['Fare']=96.075485
test[a_test]=df_test
b_test = (test['Pclass'] == 2) & (test['Fare'] == 0)
df1_test=test[b_test]
df1_test['Fare']=22.202104
test[b_test]=df1_test
c_test = (test['Pclass'] == 3) & (test['Fare'] == 0)
df2_test=test[c_test]
df2_test['Fare']=12.459678
test[c_test]=df2_test


# In[ ]:


index = test['Fare'].index[test['Fare'].apply(np.isnan)]
test['Fare'].values[152] = fc_test['Fare'].values[2]


# In[ ]:


test.isnull().sum()


# In[ ]:


train['title']='Mrs'
test['title']='Mrs'


# In[ ]:


for i in range(len(train['Name'])):
    train['title'].values[i-1]= re.search('([A-Za-z]+)\.', train['Name'].values[i-1]).group()
for i in range(len(test['Name'])):
    test['title'].values[i-1]= re.search('([A-Za-z]+)\.', test['Name'].values[i-1]).group()


# In[ ]:


age=train[['title','Age']]
age_test=test[['title','Age']]
age1=age.groupby(['title']).mean()
age1_test=age_test.groupby(['title']).mean()


# In[ ]:


train['Age'].fillna(train.groupby('title')['Age'].transform('mean'), inplace=True)
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test.groupby('title')['Age'].transform('mean'), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)


# In[ ]:


train['Family']=train['SibSp']+train['Parch']
train['family_is']='0'
var = (train['Family'] > 0)
df_var=train[var]

for i in range(len(df_var['Family'])):
        df_var['family_is'].values[i-1] = '1'

train[var]=df_var


# In[ ]:


test['Family']=test['SibSp']+test['Parch']
test['family_is']='0'
var1 = (test['Family'] > 0)
df_var1=test[var1]

for i in range(len(df_var1['Family'])):
        df_var1['family_is'].values[i-1] = '1'

test[var1]=df_var1


# In[ ]:


test['cabin_is']='1'
var2 = (test['Cabin'].isnull())
df_var2=test[var2]

for i in range(len(df_var2['Cabin'])):
        df_var2['cabin_is'].values[i-1] = '0'

test[var2]=df_var2


# In[ ]:


train['cabin_is']='1'
var3 = (train['Cabin'].isnull())
df_var3=train[var3]

for i in range(len(df_var3['Cabin'])):
        df_var3['cabin_is'].values[i-1] = '0'

train[var3]=df_var3


# In[ ]:


corr = train.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


genders = {"male": 1, "female": 0}
train["SexF"] = train["Sex"].apply(lambda s: genders.get(s))
test["SexF"] = test["Sex"].apply(lambda s: genders.get(s))


# In[ ]:


train_l=pd.DataFrame(train['Survived'])
test_l=pd.DataFrame(gender['Survived'])
train_f=train[['Pclass','SexF','Age', 'Fare', 'family_is','cabin_is']]
test_f=test[['Pclass','SexF','Age', 'Fare', 'family_is','cabin_is']]


# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
results = cross_val_score(rfc, train_f, train_l['Survived'], cv=skf)
results.mean()*100


# In[ ]:


parameters = {'min_samples_leaf': [1, 3, 5, 7], 'max_depth': [5,10,15,20], "min_samples_split": [6, 8, 10]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, 
                             n_jobs=-1, oob_score=True)
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
rf=gcv.fit(train_f, train_l['Survived'])
pred_test=rf.predict(test_f)
ac=accuracy_score(test_l,pred_test)*100
ac

