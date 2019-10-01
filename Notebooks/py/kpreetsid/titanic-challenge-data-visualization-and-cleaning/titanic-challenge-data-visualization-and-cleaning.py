#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# # Data Visualization

# In[ ]:


#Calculating the number of male/female passenger Survived.
sns.countplot(x='Survived',data=train_df,hue='Sex')


# In[ ]:


#Plotting the percentage of passengers survived according to the Class they were in. 
sns.factorplot(x='Pclass',data=train_df,kind='count',hue='Survived')


# In[ ]:


#Further breaking the above graph to male/female level
sns.factorplot(x='Survived',data=train_df,hue='Sex',kind='count',col='Pclass')


# In[ ]:


#Age distribution of the passengers
sns.distplot(train_df['Age'].dropna(),bins=30,kde=False)


# In[ ]:


#Survivers according to their gender and Pclass
sns.factorplot(x='Pclass',y='Survived',data=train_df,hue='Sex')


# # Data Cleaning

# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


#Dropping Cabin column from both datasets
train_df.drop(['Cabin'],inplace=True,axis=1)
test_df.drop(['Cabin'],inplace=True,axis=1)


# In[ ]:


train_df['Embarked']=train_df['Embarked'].fillna('S')


# In[ ]:


test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].mean())


# In[ ]:


train_df.head()


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Age',data=train_df)


# In[ ]:


def age_mean(x):
    Age,Pclass=x
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 28
        else:
            return 24
    else:
        return Age


# In[ ]:


train_df['Age']=train_df[['Age','Pclass']].apply(age_mean,axis=1)


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Age',data=test_df)


# In[ ]:


def age_mean_test(x):
    Age,Pclass=x
    if pd.isnull(Age):
        if Pclass==1:
            return 43
        elif Pclass==2:
            return 26
        else:
            return 25
    else:
        return Age


# In[ ]:


test_df['Age']=test_df[['Age','Pclass']].apply(age_mean_test,axis=1)


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(train_df.isnull())


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(test_df.isnull())


# In[ ]:


def m_f(x):
    Sex=x
    if Sex=='male':
        return 1
    else:
        return 0


# In[ ]:


train_df['Sex']=train_df['Sex'].apply(m_f)
test_df['Sex']=test_df['Sex'].apply(m_f)


# In[ ]:


train_df.head()


# In[ ]:


def name(x):
    Name=x
    if Name=='Mr.':
        return 'Mr'
    elif Name=='Miss.':
        return 'Miss'
    elif Name=='Mrs.':
        return 'Mrs'
    else:
        return 'other'


# In[ ]:


train_df['Name']=train_df['Name'].map(lambda x: x.split(' ')[1])


# In[ ]:


train_df['Name']=train_df['Name'].apply(name)


# In[ ]:


test_df['Name']=test_df['Name'].map(lambda x: x.split(' ')[1])


# In[ ]:


test_df['Name']=test_df['Name'].apply(name)


# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


nametrain=pd.get_dummies(train_df['Name'],drop_first=True)
nametest=pd.get_dummies(test_df['Name'],drop_first=True)


# In[ ]:


embarkedtrain=pd.get_dummies(train_df['Embarked'],drop_first=True)
embarkedtest=pd.get_dummies(test_df['Embarked'],drop_first=True)


# In[ ]:


pclasstrain=pd.get_dummies(train_df['Pclass'],drop_first=True)
pclasstest=pd.get_dummies(test_df['Pclass'],drop_first=True)


# In[ ]:


tr_df=pd.concat([train_df,nametrain,embarkedtrain,pclasstrain],axis=1)
te_df=pd.concat([test_df,nametest,embarkedtest,pclasstest],axis=1)


# In[ ]:


te_df.drop(['Name','Embarked','Pclass','Ticket'],axis=1,inplace=True)
tr_df.drop(['Name','Embarked','Pclass','Ticket'],axis=1,inplace=True)


# In[ ]:


tr_df.head()


# In[ ]:


#Applying Mean Normalization to both datasets


# In[ ]:


tr_df['Age']=(tr_df['Age']-tr_df['Age'].mean())/(tr_df['Age'].max()-tr_df['Age'].min())
tr_df['Fare']=(tr_df['Fare']-tr_df['Fare'].mean())/(tr_df['Fare'].max()-tr_df['Fare'].min())
tr_df.head()


# In[ ]:


te_df['Age']=(te_df['Age']-te_df['Age'].mean())/(te_df['Age'].max()-te_df['Age'].min())
te_df['Fare']=(te_df['Fare']-te_df['Fare'].mean())/(te_df['Fare'].max()-te_df['Fare'].min())
tr_df.head()


# In[ ]:


x_train=tr_df[['Sex', 'Age', 'SibSp', 'Parch','Fare', 'Mr', 'Mrs', 'other', 'Q', 'S',2,3]]
y_train=tr_df['Survived']


# In[ ]:


x_test=te_df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Mr','Mrs', 'other', 'Q', 'S',2,3]]


# In[ ]:


#Data looks clean and nice, it's time for the model training.


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# #LogisticRegression

# In[ ]:


lr=LogisticRegression()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


lr.score(x_train,y_train)


# #SVM

# In[ ]:


svc=SVC()


# In[ ]:


svc.fit(x_train,y_train)


# In[ ]:


svc.score(x_train,y_train)


# #Random Forest

# In[ ]:


rnf=RandomForestClassifier()


# In[ ]:


rnf.fit(x_train,y_train)


# In[ ]:


rnf.score(x_train,y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


a

