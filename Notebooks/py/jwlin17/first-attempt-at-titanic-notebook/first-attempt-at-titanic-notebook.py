#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train = train.append(test, sort=False)


# In[ ]:


train.info()


# **TODO LIST**
# 
# **1. None values in age **
# 
# **2. None values in cabin **
# 
# **3. None values in fare **
# 
# **4. None values in embarked **
# 
# 

# **Embarked Training Set**
# Replace with S because it is most common port for that fare price

# In[ ]:


train['Embarked'].fillna('S', inplace=True)


# In[ ]:


embarked = pd.get_dummies(train['Embarked'])
embarked_test = pd.get_dummies(test['Embarked'])


# In[ ]:


embarked.head()


# In[ ]:


train = pd.concat([train, embarked], axis=1)
test = pd.concat([test, embarked_test], axis=1)


# In[ ]:


train.head()


# In[ ]:


train.head()


# **Finished with Embarked Location Dummy Variables**

# ** Starting with Title from name**

# In[ ]:


import re

titles = [re.sub('(.*, )|(\\..*)', '', x) for x in train['Name']]
train['Title'] = titles


# In[ ]:


from collections import Counter

Counter(titles)


# In[ ]:


train.groupby(['Title']).mean()


# ** Based on survival rates of these titles **

# In[ ]:


title_survivors = ['the Countess', 'Ms', 'Mme', 'Mlle', 'Lady', 'Mrs', 'Miss', 'Dona']
title_not_survivors = ['Capt', 'Rev', 'Jonkheer']
title_low_survivors = ['Mr', 'Don']
title_average_surv = ['Col', 'Dr', 'Major', 'Master', 'Sir']


# In[ ]:


train['Survivor Title'] = [int(x in title_survivors) for x in train['Title'] ]
train['Dead Title'] = [int(x in title_not_survivors) for x in train['Title'] ]
train['Low Survivor Title'] = [int(x in title_low_survivors) for x in train['Title'] ]
train['Avg Survivor Title'] = [int(x in title_average_surv) for x in train['Title'] ]


# In[ ]:


train.head()


# ** Change male and female to values **

# In[ ]:


train = train.replace('female', 1)
train = train.replace('male', 0)


# In[ ]:


train.head()


# In[ ]:


train.info()


# ** STILL TO DO **
# 
# ** 1. Missing Value in Fare **
# 
# ** 2. Missing Values in Cabin **
# 
# ** 3. Missing Values in Age **

# **Starting with Fare Price**

# In[ ]:


train[train['Fare'].isnull()]


# In[ ]:


grouped = train.groupby(['Embarked']).mean()
grouped


# In[ ]:


train['Fare'].fillna(grouped['Fare']['S'], inplace=True)


# In[ ]:


train.info()


# ** Finished Dealing with missing fare value **
# 
# ** 1. Missing Cabin **
# 
# ** 2. Missing Age ** 

# ** Taking a look at what to do with Cabin data **

# In[ ]:


train['Cabin Area'] = [str(x)[0] for x in train['Cabin']]


# In[ ]:


train.groupby('Cabin Area').mean()


# In[ ]:


survivor_cabins = ['B', 'D', 'E', 'C', 'F']


# In[ ]:


train['Good Cabin'] = [int(x in survivor_cabins) for x in train['Cabin Area']]


# In[ ]:


train.head()


# ** Looking at age now with random forest prediction **

# In[ ]:


train.info()


# ** Make new data set to predict ages that is clean **

# In[ ]:


age_train = train


# In[ ]:


age_train.head()


# In[ ]:


age_train = age_train.drop(['Cabin', 'Embarked', 'Name', 'Cabin Area', 'Title','PassengerId', 'Ticket'], axis=1)


# In[ ]:


age_train.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


age_train = age_train.drop('Survived', axis=1)


# In[ ]:


test_age_train = age_train[age_train.isnull().any(axis=1)]
test_age_train = test_age_train.drop('Age', axis=1)


# In[ ]:


age_train = age_train.dropna(axis=0)


# In[ ]:


age_train['Fare'] = [float(x) for x in age_train['Fare']]


# In[ ]:


age_train.head()


# In[ ]:


from sklearn.model_selection import train_test_split

predictors = ['Fare', 'Parch', "Pclass", "Sex", 'SibSp','C', 'Q', 'S', 'Survivor Title', 
              'Dead Title', 'Low Survivor Title', 'Avg Survivor Title', 'Good Cabin']
X_age_train, X_age_test, Y_age_train, Y_age_test = train_test_split(age_train[predictors], age_train["Age"])


# In[ ]:


X_age_train = age_train.drop('Age', axis=1)
Y_age_train = age_train['Age']


# In[ ]:


Y_age_train = [int(x) for x in Y_age_train]


# In[ ]:


age_train.head()


# In[ ]:


predictors = ['Fare', 'Parch', "Pclass", "Sex", 'SibSp','C', 'Q', 'S', 'Survivor Title', 'Dead Title', 'Low Survivor Title', 'Avg Survivor Title', 'Good Cabin']
clf = RandomForestClassifier(n_estimators=100,
                             criterion='gini',
                             max_depth=5,
                             min_samples_split=10,
                             min_samples_leaf=5,
                             random_state=0)
clf.fit(age_train[predictors], Y_age_train)
prediction = clf.predict(test_age_train)


# In[ ]:


prediction


# In[ ]:


test_age_train['Age'] = prediction


# In[ ]:


test_age_train.head()


# In[ ]:


age_train = age_train.append(test_age_train, sort=False)


# In[ ]:


age_train.head()


# In[ ]:


import math


# In[ ]:


age_train['Survived'] = [x for x in train['Survived']]


# In[ ]:


age_train = age_train.replace(1.0, 1)
age_train = age_train.replace(0.0, 0)


# In[ ]:


age_train.head()


# ** SWEET Now we have no missing values other than survived values **
# 
# Now to seperate the tables again and use random forest to calculate for survived

# In[ ]:


train_set = age_train.dropna(axis=0)


# In[ ]:


test_set = age_train[age_train.isnull().any(axis=1)]


# In[ ]:


X_train_set = train_set.drop('Survived', axis=1)
Y_train_set = train_set['Survived']


# In[ ]:


X_test_set = test_set.drop('Survived', axis=1)


# In[ ]:


predictors = ['Age', 'Fare', 'Parch', "Pclass", "Sex", 'SibSp','C', 'Q', 'S', 'Survivor Title', 'Dead Title', 'Low Survivor Title', 'Avg Survivor Title', 'Good Cabin']
clf = RandomForestClassifier(n_estimators=4000,
                             criterion='gini',
                             max_depth=40,
                             min_samples_split=10,
                             min_samples_leaf=20,
                             oob_score=True,
                             random_state=0)
clf.fit(X_train_set[predictors], Y_train_set)
print("%.4f" % clf.oob_score_)
prediction = clf.predict(X_test_set)


# In[ ]:


prediction.size


# In[ ]:


submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})
submission.head()


# In[ ]:


submission.to_csv("submission3.csv", index=False)


# As the resulting trained model doesn't quite hit the level of accuracy I would like in my ML decision tree. I will continue trying to improve my feature engineering. Looking forward to future uploads!

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




