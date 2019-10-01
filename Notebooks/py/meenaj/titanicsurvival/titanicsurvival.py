#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
from sklearn import metrics


# In[ ]:


train= pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


sns.countplot('Embarked', hue='Survived', data= train)


# In[ ]:


train['Age'].hist(bins=50)


# In[ ]:


sns.boxplot(train['Survived'],train['Fare'], hue= train['Embarked'])


# In[ ]:


train['Embarked'] = train['Embarked'].fillna('C')


# In[ ]:


test[test['Fare'].isnull()]


# In[ ]:


Fare_med= test[(test['Pclass']==3) & (test['Embarked'] == 'S')]['Fare'].median()


# In[ ]:


test['Fare']=test['Fare'].fillna(Fare_med)


# In[ ]:


train['cabin']=train.Cabin.str[0]
test['cabin']=test.Cabin.str[0]


# In[ ]:


train['cabin'].fillna('U', inplace=True)
test['cabin'].fillna('U', inplace=True)


# In[ ]:


sns.countplot(train['cabin'], hue=train['Survived'])


# In[ ]:


train['Family']= train['Parch']+ train['SibSp']+1
test['Family']= test['Parch']+ test['SibSp']+1


# In[ ]:


train.loc[train["Family"] == 1, "FamilySize"] = 'singleton'
train.loc[(train["Family"] > 1)  &  (train["Family"] < 5) , "FamilySize"] = 'small'
train.loc[train["Family"] >4, "FamilySize"] = 'large'
test.loc[test["Family"] == 1, "FamilySize"] = 'singleton'
test.loc[(test["Family"] > 1)  &  (test["Family"] < 5) , "FamilySize"] = 'small'
test.loc[test["Family"] >4, "FamilySize"] = 'large'


# In[ ]:


sns.countplot(train['FamilySize'],hue=train['Survived'])


# In[ ]:


test.info()


# In[ ]:


sns.countplot(test['FamilySize'],hue=test['cabin'])


# In[ ]:


sns.countplot(train['FamilySize'], hue=train['cabin'])


# In[ ]:


train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
train['Name_Title'].value_counts()


# In[ ]:


test['Name_Title'] = test['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
test['Name_Title'].value_counts()


# In[ ]:


train.head()


# In[ ]:


sns.countplot(train['Name_Title'], hue=train['Survived'])


# In[ ]:


train['ticketlength'] = train['Ticket'].apply(lambda x: len(x))
test['ticketlength'] = test['Ticket'].apply(lambda x: len(x))


# In[ ]:


sns.countplot(train['ticketlength'], hue=train['Survived'])


# In[ ]:


train['ticketlength'].value_counts()


# In[ ]:


test['ticketlength'].value_counts()


# In[ ]:


train['TicketfirstL'] = train['Ticket'].apply(lambda x: str(x)[0])
test['TicketfirstL'] = test['Ticket'].apply(lambda x: str(x)[0])


# In[ ]:


sns.countplot(train['TicketfirstL'], hue=train['Survived'])


# In[ ]:


train= train.drop(['Name','Ticket','Cabin'],axis=1)
test= test.drop(['Name','Ticket','Cabin'],axis=1)


# In[ ]:


test.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cat=['Sex','Embarked','cabin','FamilySize','Name_Title','TicketfirstL']
for col in cat:
    train[col]=le.fit_transform(train[col])
    test[col]=le.fit_transform(test[col])


# In[ ]:


test.head()


# In[ ]:


X=train.drop(['Survived'],axis=1)
A_train= train.loc[train.Age.notnull()]      
A_test= train.loc[train.Age.isnull()]       #Age = Nan( to be predicted)
X_Age=A_train.drop(['Age'], axis=1)
y_Age=A_train['Age']
y_test_Age= A_test['Age']
X_test_Age= A_test.drop(['Age'],axis=1)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor()
rf.fit(X_Age,y_Age)
y_pred_Age=rf.predict(X_test_Age)


# In[ ]:


train.loc[(train.Age.isnull()), 'Age' ] = y_pred_Age


# In[ ]:


plt.hist(train['Age'], bins=50)


# In[ ]:


AA_train= test.loc[test.Age.notnull()]      
AA_test= test.loc[test.Age.isnull()]       #Age = Nan( to be predicted)
X1=AA_train.drop(['Age'], axis=1)
y1=AA_train['Age']
y2= AA_test['Age']
X2= AA_test.drop(['Age'],axis=1)


# In[ ]:


rf.fit(X1,y1)
y2_pred=rf.predict(X2)


# In[ ]:


test.loc[(test.Age.isnull()), 'Age' ] = y2_pred


# In[ ]:


test.info()


# In[ ]:


train.corr()['Survived']


# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(train.corr(), annot=True)


# In[ ]:


U=train.drop(['PassengerId','Survived','Family','SibSp','Parch'],axis=1)
V=train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split
U_tr, U_ts,V_tr,V_ts = train_test_split(U, V, test_size=0.20, random_state=0)
print(U_tr.shape, V_tr.shape, U_ts.shape, V_ts.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


clf = RandomForestClassifier(random_state=0, n_jobs=-1)
param_grid = { "criterion" : ["gini", "entropy"]
             , "min_samples_leaf" : [1, 5, 10]
             , "min_samples_split" : [2, 6, 10, 12, 16]
             , "n_estimators": [10, 50, 100, 400, 600]}
gs = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
gs = gs.fit(U_tr, V_tr)


# In[ ]:


print(gs.best_score_)
print(gs.best_params_)


# In[ ]:


model=RandomForestClassifier(criterion='entropy',min_samples_leaf=1,min_samples_split=12,n_estimators=50,random_state=0,n_jobs=-1)
model.fit(U_tr,V_tr)
pred= model.predict(U_ts)
model.score(U_ts,V_ts)


# In[ ]:


ID=test['PassengerId']
test=test.drop(['PassengerId','SibSp','Parch','Family'], axis=1)


# In[ ]:


model.fit(U,V)
titanic_s=model.predict(test)


# In[ ]:


titanic=pd.DataFrame({'PassengerId': ID, 'Survived':titanic_s})
titanic.head()


# In[ ]:


titanic.to_csv('titanic.csv',index=False)


# In[ ]:




