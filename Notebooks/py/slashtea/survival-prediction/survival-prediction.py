#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# ### In this notebook, we're going to explore first of the training set, make some EDA see how those features are what they are, then select our features and finally train a classifier which predicts whether a person survived or not.  

# In[ ]:


train = pd.read_csv('../input/train.csv', header='infer', index_col='PassengerId')
test = pd.read_csv('../input/test.csv', header='infer', index_col='PassengerId')
train.head()


# Let's check the **shape** of dataset and see if there are any **zeros** or **NaNs**. As well as some **basic summary** about the dataframe.

# In[ ]:


print('****** DFs shape: ******', train.shape)
print(train.info())
print(train.describe())


# ##### Initial Hypothesis.
#     * Women are likely to survive more than men.
#     * First class people, are likely to survive more.
#     * People travelling without any relatives are more likely to survive
#     
#  Let's check those hyposesis with some metrics and visualizatons!

# In[ ]:


train.loc[:, train.isnull().any()].head()


# We can see that **Age** and **Cabin** columns have quite some NaN, this will need to be fixed.
# 
# #### For now, let's explore the data through some plots.

# In[ ]:


print(train['Embarked'].value_counts())

# We will fill the missing values in Embarked with 'S' which is the most common.
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked']  = test['Embarked'].fillna('S')


# In[ ]:


# We'll fill the missing age values by the mean for training and testing data.
train['Age'].fillna(train['Age'].median(), inplace = True)
test['Age'] = test['Age'].fillna(test['Age'].mean())


# In[ ]:


train.loc[:, train.isnull().any()].head()


# In[ ]:


survived = train['Survived'] == 1
male = train['Sex'] == 'male'
female = train['Sex'] == 'female'

print('Average age of male who survived', train[survived & male].Age.mean())
print('Average age of female who survived', train[survived & female].Age.mean())


# #### You might have noticed that in the text Column there are some tittles for people.
# 
# Let's get a count for those.
# 

# In[ ]:


train['Title'] = train['Name'].str.extract(' ([A-Za-z]+).', expand=False)
pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


g = sns.factorplot(x="Embarked", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")


# In[ ]:


print(train[['Pclass', 'Survived']].groupby('Pclass').mean())
g = sns.factorplot(x="Pclass", y="Survived", data=train,
                   size=6, kind="bar", palette="muted")
g.set_ylabels("survival probability")


# In[ ]:


print(train[['Sex', 'Survived']].groupby('Sex').mean())

g = sns.factorplot(x="Sex", y="Survived", data=train,
                   size=6, kind="bar", palette="muted")
g.set_ylabels("survival probability")


# In[ ]:


print(train[['Embarked', 'Survived']].groupby('Embarked').mean())
g = sns.factorplot(x="Embarked", y="Survived", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")


# In[ ]:


sns.violinplot(x="Pclass", y="Survived", hue="Sex", data=train, split=True,
               inner="quart", palette="Set3")
sns.despine(left=True)


# In[ ]:


sns.violinplot(x="Embarked", y="Survived", hue="Sex", data=train, split=True,
               inner="quart", palette="Set2")
sns.despine(left=True)


# ###### From the plots we can validate the following assumptions:
# 
#     * Women survived more than men, gentleman titanic people.
#     * People who embarked from Cherbourg survived more.
#     * First class people survived more than other classes, their survival percentage is 63%!
#     * Survival was lowest in third class passengers.

# ##### It is safe to assume the following:
#     - Cabine feature seems to be irrelevant to whether a person survived or not, plus there are a lot of missing values.
#     - Ticket doesn't really affect survival.
#     - Fare either
#     - And name

# In[ ]:


train.drop(['Name', 'Fare', 'Ticket', 'Cabin'], axis=1).head()


# In[ ]:


train.columns


# In[ ]:


test.drop(['Name', 'Fare', 'Ticket', 'Cabin'], axis=1).head()


# In[ ]:


# Here some mapping to encode categorical variables.

sex_mapping  = {'male': 1, 'female': 0}
embark_encode = {'C': 1, 'S': 2, 'Q': 3}


# In[ ]:


test.head()
test['Sex'] = test['Sex'].map(sex_mapping)
test['Embarked'] = test['Embarked'].map(embark_encode)


# In[ ]:


train['Sex'] = train['Sex'].map(sex_mapping)
train['Embarked'] = train['Embarked'].map(embark_encode)


# In[ ]:


train.head()


# In[ ]:


X_train = train.loc[:, ['Pclass', 'Sex', 'Age', 'Embarked']]
Y_train = train.loc[:, ['Survived']]

X_test = test.loc[:, ['Pclass', 'Sex', 'Age', 'Embarked']]

print(X_test.head())
print(X_test.info())


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)
# acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_logreg)
print(round(lr.score(X_train, Y_train) * 100, 2))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

print(round(clf.score(X_train, Y_train) * 100, 2))


# In[ ]:


X_test.reset_index(inplace=True)
X_test.columns
submission = pd.concat([pd.Series(X_test["PassengerId"]), pd.Series(y_pred)], axis=1)
submission.head()


# In[ ]:


submission.to_csv('to_submit.csv', index=False)


# In[ ]:




