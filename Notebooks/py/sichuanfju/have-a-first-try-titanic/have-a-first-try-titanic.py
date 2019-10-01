#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import train_test_split


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.tail()


# In[ ]:


test.tail()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.shape


# In[ ]:


test.shape


# Data Exploration

# In[ ]:


#Age
fig=plt.figure(figsize=(10,5))

age1=train[train['Survived']==1]['Age'].value_counts()
age2=train[train['Survived']==0]['Age'].value_counts()

sns.set_style('white')
sns.distplot(age1,hist=False,label='Survived')
sns.distplot(age2,hist=False,label='Died')


# In[ ]:


#Sex
sns.countplot(x='Sex',hue='Survived',data=train)


# In[ ]:


#Pclass
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train)


# In[ ]:


#SibSp & Parch
fig=plt.figure(figsize=(10,5))

train["Family"]=train['SibSp']+train['Parch']
sns.barplot(x='Family',y='Survived',data=train)


# In[ ]:


#Embarked
sns.countplot(x='Embarked',hue='Survived',data=train)


# Data Cleaning

# In[ ]:


train.drop(['Family'],inplace=True,axis=1)
train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


# Embarked  missing
train['Embarked']=train["Embarked"].fillna("S")


# In[ ]:


#Fare  missing
test['Fare'].fillna(test["Fare"].mean(),inplace=True)


# In[ ]:


# drop unnecessary columns
train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1,inplace=True)
test.drop(['Name','Ticket','Cabin'], axis=1,inplace=True)


# In[ ]:


train.head()


# Feature Engineering

# In[ ]:


#creature feature : Family
train["Family"]=train['SibSp']+train['Parch']
train.drop(['SibSp','Parch'],axis=1,inplace=1)

test["Family"]=test['SibSp']+test['Parch']
test.drop(['SibSp','Parch'],axis=1,inplace=1)


# In[ ]:


#Encoding sex, female: 0 and male: 1
train['Sex'].loc[train['Sex']=='female']=0
train['Sex'].loc[train['Sex']=='male']=1

test['Sex'].loc[test['Sex']=='female']=0
test['Sex'].loc[test['Sex']=='male']=1


# In[ ]:


train=pd.get_dummies(train,columns=["Pclass","Embarked"])
test=pd.get_dummies(test,columns=["Pclass","Embarked"])


# In[ ]:


#Age
mean1 = train["Age"].mean()
std1= train["Age"].std()
count1 = train["Age"].isnull().sum()

mean2= test["Age"].mean()
std2= test["Age"].std()
count2 = test["Age"].isnull().sum()

rand1=np.random.randint(mean1-std1,mean1+std1,size=count1)
rand2=np.random.randint(mean2-std2,mean2+std2,size=count2)

train['Age'][np.isnan(train['Age'])]=rand1
test['Age'][np.isnan(test['Age'])]=rand2


# In[ ]:


#Standardization
sd_train=train[['Age','Fare','Family']]
sd_test=test[['Age','Fare','Family']]

sds = StandardScaler()
sds.fit(sd_train)

sds_xtrain1 = sds.transform(sd_train)
sds_xtest1  = sds.transform(sd_test)
train[['Age','Fare','Family']]=sds_xtrain1
test[['Age','Fare','Family']]=sds_xtest1


# In[ ]:


train.head()


# In[ ]:


xdata=train.drop("Survived",axis=1)
ydata=train["Survived"]
xtrain,xtest,ytrain,ytest = train_test_split(xdata,ydata,test_size = 0.2)


# LogisticRegression

# In[ ]:


lr= LogisticRegression()
param = {"C":[0.001,0.01,0.1,1],'max_iter':[10,50,100,200]}
gs = GridSearchCV(lr,param,cv = 5)
gs.fit(xtrain,ytrain)


# In[ ]:


lr=gs.best_estimator_


# In[ ]:


lr.fit(xtrain,ytrain)


# In[ ]:


lr.score(xtest,ytest)


# Random Forest

# In[ ]:


rfc=RandomForestClassifier()
param1={"n_estimators":list(range(5,50))}
gs1=GridSearchCV(rfc,param1,cv=5)
gs1.fit(xtrain,ytrain)


# In[ ]:


rfc=gs1.best_estimator_


# In[ ]:


rfc.fit(xtrain,ytrain)


# In[ ]:


rfc.score(xtest,ytest)


# SVM

# In[ ]:


clf=svm.SVC(probability=True)
param2={"kernel":("linear","rbf"),"C":[0.001,0.01,0.1,1]}
gs2=GridSearchCV(clf,param2,cv=5)
gs2.fit(xtrain,ytrain)


# In[ ]:


clf=gs2.best_estimator_


# In[ ]:


clf.fit(xtrain,ytrain)


# In[ ]:


clf.score(xtest,ytest)


# Make submission

# In[ ]:


ID = test['PassengerId']
xtest=test.drop(['PassengerId'],axis=1)


# In[ ]:


ytest=clf.predict(xtest)


# In[ ]:


submission=pd.DataFrame({'PassengerId':ID,'Survived':ytest})


# In[ ]:


submission.to_csv("titanic.csv",index=False)


# In[ ]:




