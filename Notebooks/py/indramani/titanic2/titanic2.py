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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


combine = [train_df,test_df]


# In[ ]:


train_df.isna().sum()


# In[ ]:


train_df[["Survived",'Sex']].groupby('Sex').mean()


# In[ ]:


print(train_df.columns.values)


# In[ ]:


train_df.Embarked.unique()


# In[ ]:


print(train_df)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


test_df.Cabin.isna().sum()


# In[ ]:


train_df[['SibSp','Survived']].groupby('SibSp').mean()


# In[ ]:


train_df[['Parch','Survived']].groupby('Parch').mean()


# In[ ]:


g = sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist,'Age',bins=20)


# In[ ]:


pc =sns.FacetGrid(train_df,col = 'Survived',row= 'Pclass')
pc.map(plt.hist, 'Age', bins= 20)


# In[ ]:


train_df = train_df.drop(['Cabin','Ticket'],axis = 1)
test_df = test_df.drop(['Cabin','Ticket'],axis=1)


# In[ ]:


train_df['Title']= train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    


# In[ ]:


test_df['Title']= test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


pd.crosstab(train_df['Title'], train_df['Sex'])


# In[ ]:


train_df['Title']= train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')


# In[ ]:


test_df['Title']= test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')


# In[ ]:


train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train_df['Title'] = train_df['Title'].map(title_mapping)


# In[ ]:


train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


test_df['Title'] = test_df['Title'].map(title_mapping)


# In[ ]:


train_df = train_df.drop('Name',axis =1)
test_df= test_df.drop('Name',axis =1)


# In[ ]:


train_df.head()


# In[ ]:


sex_mapping ={"male":0,'female':1}


# In[ ]:


train_df['Sex']= train_df['Sex'].map(sex_mapping).astype(int)
test_df['Sex']= test_df['Sex'].map(sex_mapping).astype(int)
train_df.head()


# In[ ]:


pd.cut(train_df['Age'], 5).unique()


# In[ ]:


train_df['Age']= train_df['Age'].fillna(train_df["Age"].median())


# In[ ]:


test_df['Age']= test_df['Age'].fillna(test_df["Age"].median())


# In[ ]:


train_df.loc[train_df["Age"]<=16,'Age']=0


# In[ ]:


train_df.loc[(train_df["Age"]>16)&(train_df['Age']<=32),'Age']=1
train_df.loc[(train_df["Age"]>32)&(train_df['Age']<48),'Age']=2
train_df.loc[(train_df["Age"]>48)&(train_df['Age']<=64),'Age']=3
train_df.loc[train_df['Age']>64,'Age']=4


# In[ ]:


test_df.loc[test_df["Age"]<=16,'Age']=0
test_df.loc[(test_df["Age"]>16)&(test_df['Age']<=32),'Age']=1
test_df.loc[(test_df["Age"]>32)&(test_df['Age']<48),'Age']=2
test_df.loc[(test_df["Age"]>48)&(test_df['Age']<=64),'Age']=3
test_df.loc[train_df['Age']>64,'Age']=4


# In[ ]:


train_df['familySize'] =train_df.Parch+train_df.SibSp +1


# In[ ]:


test_df['familySize']= test_df.Parch+test_df.SibSp +1


# In[ ]:


train_df['IsAlone'] =0
train_df.loc[train_df['familySize']==1,'IsAlone'] = 1


# In[ ]:


train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:


train_df.head()


# In[ ]:


test_df['IsAlone'] =0
test_df.loc[test_df['familySize']==1,'IsAlone'] = 1


# In[ ]:


train_df =train_df.drop(['SibSp','Parch','familySize'],axis=1)
test_df = test_df.drop(['SibSp','Parch','familySize'],axis=1)


# In[ ]:


combine =[train_df,test_df]


# In[ ]:


train_df.isna().sum()


# In[ ]:


train_df.Embarked =train_df.Embarked.fillna(train_df.Embarked.dropna().mode()[0])


# In[ ]:


test_df.Embarked =test_df.Embarked.fillna(test_df.Embarked.dropna().mode()[0])


# In[ ]:


train_df.Embarked = train_df.Embarked.map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


# In[ ]:


test_df.Embarked = test_df.Embarked.map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[ ]:


pd.cut(train_df['Fare'],5).unique()


# In[ ]:


train_df.loc[train_df['Fare']<=102,"Fare"]=0
train_df.loc[(train_df['Fare']>102)&(train_df['Fare']<=204),"Fare"]=1
train_df.loc[(train_df['Fare']>204)&(train_df['Fare']<=307),"Fare"]=2
train_df.loc[(train_df['Fare']>307)&(train_df['Fare']<=409),"Fare"]=3
train_df.loc[train_df['Fare']>409,"Fare"]=4


# In[ ]:


test_df.loc[test_df['Fare']<=102,"Fare"]=0
test_df.loc[(test_df['Fare']>102)&(test_df['Fare']<=204),"Fare"]=1
test_df.loc[(test_df['Fare']>204)&(test_df['Fare']<=307),"Fare"]=2
test_df.loc[(test_df['Fare']>307)&(test_df['Fare']<=409),"Fare"]=3
test_df.loc[test_df['Fare']>409,"Fare"]=4


# In[ ]:


train_df.head()


# In[ ]:


corrmat = train_df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[ ]:


train_df=pd.get_dummies(train_df)
test_df= pd.get_dummies(test_df)


# In[ ]:


X_train = train_df.drop(["Survived",'PassengerId'], axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop(["PassengerId",], axis=1).copy()


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# In[ ]:


X_test.isna().sum()


# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:





# In[ ]:


logistic = LogisticRegression()
logistic.fit(X_train, Y_train)
Y_pred = logistic.predict(X_test)
acc_log = round(logistic.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


Y_pred = random_forest.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




