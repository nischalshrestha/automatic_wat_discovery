#!/usr/bin/env python
# coding: utf-8

# This kernel gives a score of 0.80861 if run on python 2.7
# if run on kaggle, it gives a score of 0.79904

# In[ ]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import re
fig=plt.figure()
import string
import random


# In[ ]:


# reading in data
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


# checking missing values
train.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


# cabin can be important but has too many missing values so we make a new class nan and chop off the first letter of each cabin.
train.Cabin = train.Cabin.fillna('N')
train.Cabin = train.Cabin.apply(lambda x: x[0])
test.Cabin = test.Cabin.fillna('N')
test.Cabin = test.Cabin.apply(lambda x: x[0])
train


# In[ ]:


train['Cabin'].value_counts()


# In[ ]:


# majority embarked = Southampton
train.Embarked.fillna(value='S', inplace=True)


# In[ ]:


# create new column title containing title of every passenger
frames = [train,test]
for df in frames:
    df["Title"] = df.Name.str.replace('(.*, )|(\\..*)', '')


# In[ ]:


# Let's define another feature. FamilySize = Parch + SibSp + 1
for df in frames:
    df["FamilySize"] = df.Parch + df.SibSp + 1


# In[ ]:


train["Title"].value_counts()


# In[ ]:


train[np.isnan(train["Age"])].Title.unique()


# In[ ]:


# These can be discussed, of course.
titledict = {"Dr"   : "Mr",
             "Col"  : "Officer",
             "Mlle" : "Miss",
             "Major": "Officer",
             "Lady" : "Royal",
             "Dona" : "Royal",
             "Don"  : "Royal",
             "Mme"  : "Mrs",
             "the Countess": "Royal",
             "Jonkheer": "Royal",
             "Capt" : "Officer",
             "Sir"  : "Mr"
             }
#There is probably a pandas way to do this but i'll do this the python way
for df in frames:
    for key,val in titledict.items():
        train.loc[train["Title"]==key, "Title"] = val


# In[ ]:


train["Title"].value_counts()


# In[ ]:


#sns.barplot(x="Title", y="Survived", data=train);


# In[ ]:


#sns.barplot(x="FamilySize", y="Survived", hue="Sex", data=train);


# In[ ]:


train.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


# capture digits and ignore the 1st digit part of text
train['tkno'] = train['Ticket'].str.extract('(\d\d+)', expand=True)
test['tkno'] = test['Ticket'].str.extract('(\d\d+)', expand=True)



# In[ ]:


# replacing missing age values by title median
for t in train[np.isnan(train["Age"])].Title.unique():
    for df in frames:
        df.loc[(df["Title"]==t) & np.isnan(df["Age"]), "Age" ] = train[train["Title"]==t].Age.median()


# In[ ]:


# no more missing values
train=train.dropna()
train.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


#sns.barplot(x="Cabin", y="Age", data=train);


# In[ ]:


del train['Name']


# In[ ]:


del test['Name']


# In[ ]:


del train['Parch']


# In[ ]:


del test['Parch']


# In[ ]:


del train['SibSp']


# In[ ]:


del test['SibSp']


# In[ ]:


del train['Ticket']


# In[ ]:


del test['Ticket']


# In[ ]:


test.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


test['Age'].fillna(test['Age'].mean(), inplace=True)


# In[ ]:


test['Fare'].fillna(test['Fare'].mean(), inplace=True)


# In[ ]:


test=test.fillna('0')


# In[ ]:


test.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


train['Age*Class']=train['Age']*train['Pclass']
test['Age*Class']=test['Age']*test['Pclass']


# In[ ]:


Ticket_count = dict(train['tkno'].value_counts())


# In[ ]:


# new feature
def Tix_ct(y):
    return Ticket_count[y]
train["TicketGrp"] = train.tkno.apply(Tix_ct)
def Tix_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

train["TicketGrp"] = train.loc[:,"TicketGrp"].apply(Tix_label)  


# In[ ]:


Ticket_count = dict(test['tkno'].value_counts())


# In[ ]:


# another new feature
def Tix_ct(y):
    return Ticket_count[y]
test["TicketGrp"] = test.tkno.apply(Tix_ct)
def Tix_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

test["TicketGrp"] = test.loc[:,"TicketGrp"].apply(Tix_label)  


# In[ ]:


sns.barplot(x="Embarked", y="Survived",hue='Sex', data=train);


# In[ ]:


# encoding categorical to numeric
from sklearn import preprocessing
def encode_features(train, test):
    features = [ 'Sex','Cabin','Embarked','Title']
    df_combined = pd.concat([train[features], test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        train[feature] = le.transform(train[feature])
        test[feature] = le.transform(test[feature])
    return train, test
    
data_train, data_test = encode_features(train, test)
data_train.head()


# In[ ]:


# another new feature
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0

train["Fam"] = train.loc[:,"FamilySize"].apply(Fam_label)
test["Fam"] = test.loc[:,"FamilySize"].apply(Fam_label)


# In[ ]:


del train['tkno']
del test['tkno']
#del train['Embarked']
#del test['Embarked']
#del train['TicketGrp']
#del test['TicketGrp']
#del train['Fam']
#del test['Fam']
del train['Age*Class']
del test['Age*Class']
#del train['Cabin']
#del test['Cabin']
#del train['Title']
#del test['Title']
#del train['Pclass']
#del test['Pclass']
del train['FamilySize']
del test['FamilySize']
#del train['Fare']
#del test['Fare']


# In[ ]:


#train['fam'].value_counts()


# In[ ]:


test.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


from sklearn.cross_validation import train_test_split

X_all = train.drop(['Survived', 'PassengerId'], axis=1)
y_all = train['Survived']

num_test = 0.0
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


# In[ ]:


# neural net
from sklearn.neural_network import MLPClassifier
c = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15), random_state=1,activation='logistic',max_iter=3000)
c.fit(X_train, y_train)


# In[ ]:


c.score(X_train,y_train)


# In[ ]:


X_train.axes


# In[ ]:


ids = test['PassengerId']
predictions = c.predict(test.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions13.csv', index = False)

