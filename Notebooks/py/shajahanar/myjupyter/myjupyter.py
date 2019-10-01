#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# **Importing required libraries**

# In[169]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# **Reading test and train files to Dataframes**

# In[170]:


test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')


# In[171]:


ttt = pd.read_csv('../input/test.csv')
train.head()


# In[172]:


test.tail()


# In[173]:


train = train.drop(['Cabin','Ticket','Fare','Name','PassengerId'],axis=1)
train


# In[174]:


test = test.drop(['Cabin','Ticket','Fare','Name','PassengerId'],axis=1)
test


# In[175]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[176]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[177]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[178]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[179]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[180]:


train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[181]:


sns.countplot(x='SibSp',data=train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[182]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',hue='Sex',data=test,palette='winter')


# In[183]:


def impute_age_test(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    
    if pd.isnull(Age):

        if Pclass == 1:
            if Sex == 'male':
                return 42.5
            else:
                return 41

        elif Pclass == 2:
            if Sex == 'male':
                return 28
            else:
                return 24

        else:
            if Sex == 'male':
                return 24
            else:
                return 22.5

    else:
        return Age


# In[184]:


def impute_age_train(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    
    if pd.isnull(Age):

        if Pclass == 1:
            if Sex == 'male':
                return 40
            else:
                return 35

        elif Pclass == 2:
            if Sex == 'male':
                return 30
            else:
                return 28

        else:
            if Sex == 'male':
                return 25
            else:
                return 22

    else:
        return Age


# **Filling missing Age values using the information obtained from above boxplots and using defined impute function**

# In[189]:


train['Age'] = train[['Age','Pclass','Sex']].apply(impute_age_train,axis=1)
test['Age'] = test[['Age','Pclass','Sex']].apply(impute_age_test,axis=1)


# In[ ]:





# In[191]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# 

# In[192]:


train.head()


# In[193]:


test.head()


# In[ ]:





# In[195]:


sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
test = pd.concat([test,sex,embark],axis=1)
test.drop(['Sex','Embarked'],axis=1,inplace=True)
test.info()


# In[ ]:





# In[197]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Embarked'],axis=1,inplace=True)
train.info()


# In[198]:


X_train = train.drop('Survived',axis=1)
y_train = train['Survived']
test


# In[206]:


# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# Y_pred = logreg.predict(test)
# acc_log = round(logreg.score(X_train, y_train) * 100, 2)
# acc_log


# In[208]:


# from sklearn.svm import SVC
# svc = SVC()
# svc.fit(X_train, y_train)
# Y_pred = svc.predict(test)
# acc_svc = round(svc.score(X_train, y_train) * 100, 2)
# acc_svc


# In[209]:


# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(X_train, y_train)
# Y_pred = knn.predict(test)
# acc_knn = round(knn.score(X_train, y_train) * 100, 2)
# acc_knn


# In[215]:


# from sklearn.tree import DecisionTreeClassifier
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, y_train)
# Y_pred = decision_tree.predict(test)
# acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
# acc_decision_tree


# In[220]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(test)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest


# In[221]:


survival_prediction = Y_pred


# In[222]:


submission = pd.DataFrame({
        "PassengerId": ttt["PassengerId"],
        "Survived": survival_prediction
    })


# In[223]:


submission.to_csv('submission.csv', index=False)


# In[224]:


submission


# In[ ]:




