#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# Hi I am new in data science I have created this notebook to test and improve myself, I Hope it benefits your business

# **Import Lib's**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# **Import Data sets and read **

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


train.head(8)


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


survived, N = len(train[train['Survived'] == 1]), len(train)
print("Survived:", survived)
print("Total Passenger:", N)
print("Survive Percent:", survived/N)


# In[ ]:


train.isnull().sum()


# **There are 891 Data in Our Data Set**
# 
# 
# Strategy : 
# * The Age Feature have 19% missing data but i think age data too important to analise for survive feature
# * The Cabin Feature Have %77 missing data i think this not too important for survive feature, i will delete that data on dataset 
# * The Embarked Feature have %0.22 missing data I do not think it is a problem

# **1)Sex Feature**

# In[ ]:


import seaborn as sns
sns.barplot(x="Sex", y="Survived", data=train)

print('Men who survived %', 100*np.mean(train['Survived'][train['Sex'] == 'male']))
print('Women who survived %', 100*np.mean(train['Survived'][train['Sex'] == 'female']))


# * as you can see females have more chance to live than males
# 

# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=train)

print('Passengers who survived in first class % ', 100*np.mean(train['Survived'][train['Pclass'] == 1]))
print('Passengers who survived in second class % ', 100*np.mean(train['Survived'][train['Pclass'] == 2]))
print('Passengers who survived in third class %', 100*np.mean(train['Survived'][train['Pclass'] == 3]))


# * think that those who have economically higher incomes have a higher survival rate

# **3)Siblings/Spouces Feature**

# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=train)

print('Passengers who survived siblings = 0  %',100*np.mean(train['Survived'][train['SibSp'] == 0]))
print('Passengers who survived siblings = 1  %',100*np.mean(train['Survived'][train['SibSp'] == 1]))
print('Passengers who survived siblings = 2  %',100*np.mean(train['Survived'][train['SibSp'] == 2]))
print('Passengers who survived siblings = 3  %',100*np.mean(train['Survived'][train['SibSp'] == 3]))
print('Passengers who survived siblings = 4  %',100*np.mean(train['Survived'][train['SibSp'] == 4]))


# * As you can see, Siblings Feature effect negative how many siblings you have so far, your chances of survival are so low

# **4)Parch Feature**

# In[ ]:


sns.barplot(x="Parch", y="Survived", data=train)

print('Passengers who survived Parents/Children = 0  %',100*np.mean(train['Survived'][train['Parch'] == 0]))
print('Passengers who survived Parents/Children = 1  %',100*np.mean(train['Survived'][train['Parch'] == 1]))
print('Passengers who survived Parents/Children = 2  %',100*np.mean(train['Survived'][train['Parch'] == 2]))
print('Passengers who survived Parents/Children = 3  %',100*np.mean(train['Survived'][train['Parch'] == 3]))
print('Passengers who survived Parents/Children = 5  %',100*np.mean(train['Survived'][train['Parch'] == 5]))


# * as you can see, Parch Feature effect  Positive how many Children/Parents you have so far, your chances of survival are so high

# **5)Age Feature**

# In[ ]:


#sort the ages into logical categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)

print('Passengers who survived Age = 0-5  %',100*np.mean(train['Survived'][train['Age'] < 5]))
print('Passengers who survived Age = 5-12  %',100*np.mean(train['Survived'][train['Age'] < 12]))
print('Passengers who survived Age = 12-18  %',100*np.mean(train['Survived'][train['Age'] < 18]))
print('Passengers who survived Age = 18-35  %',100*np.mean(train['Survived'][train['Age'] < 35]))
print('Passengers who survived Age = 35-60  %',100*np.mean(train['Survived'][train['Age']< 60 ]))
print('Passengers who survived Age = 60+  %',100*np.mean(train['Survived'][train['Age'] > 60]))


# * babies survive more than other AgeGroup

# **6)Embarked Feature**

# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=train)

print('Passengers who survived go Southampton %',100*np.mean(train['Survived'][train['Embarked'] == 'S']))
print('Passengers who survived go Cherbourg %',100*np.mean(train['Survived'][train['Embarked'] == 'C']))
print('Passengers who survived go Queenstown %',100*np.mean(train['Survived'][train['Embarked'] == 'Q']))


# * Those who go to Cherbourg are more alive

# **Lets See Test Data**

# In[ ]:


test.describe(include="all")


# * We Have 418 Passenger 
# * 1 value from the Fare feature is missing.
# * Around 20% of the Age feature is missing, we will need to fill that in.

# In[ ]:


#we will drop some features

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


# In[ ]:


train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

#i drop this features because is not useful


# **Embarked Data **

# In[ ]:


print("Number of people embarking in Southampton (S):")
southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)


# * We Need to fill missing values on Southampton

# In[ ]:


train = train.fillna({"Embarked": "S"})


# * Now we need to fill missing age features, we have too much missing features we need to predict missing ages

# In[ ]:


#create a combined group of both datasets
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


#replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()


# In[ ]:


# fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

#I tried to get this code to work with using .map(), but couldn't.
#I've put down a less elegant, temporary solution for now.
#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})
#test = test.fillna({"Age": test["Title"].map(age_title_mapping)})

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
        
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
        


# In[ ]:


#drop the name feature since it contains no more useful information.
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# In[ ]:


train.isnull().sum()


# * as you can see we fill that missing values

# In[ ]:


#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# * we digitize sex to 0-1 male=1 female=0

# In[ ]:


embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()


# * now we need the fill 1 Fare Feature

# In[ ]:


#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[ ]:


train.head()


# In[ ]:


age_mapping = {"Baby": 1, "Child": 2, "Teenager": 3, "Student": 4, "Young Adult": 5, "Adult": 6, "Senior": 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train.head()


# In[ ]:


train.head()


# In[ ]:


test.head()


# **Lets Get Best Model**

# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# I will be testing the following models with my training data (got the list from here):
# 
# * Gaussian Naive Bayes
# * Logistic Regression
# * Decision Tree Classifier
# * Random Forest Classifier

# In[ ]:


train.head()


# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Decision Tree'],
    'Score': [ acc_logreg, 
              acc_randomforest, acc_gaussian, acc_decisiontree,
             ]})
models.sort_values(by='Score', ascending=False)


# **Create Submission File**

# In[ ]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = randomforest.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# **Sources :**
# 
# * [Titanic Survival Predictions (Beginner)](https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner)
# * [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# * [A tutorial for Complete Beginners](https://www.kaggle.com/drgilermo/a-tutorial-for-complete-beginners)

# **Thank you for reading, if you have a questions dont be hesitate comment**

# In[ ]:




