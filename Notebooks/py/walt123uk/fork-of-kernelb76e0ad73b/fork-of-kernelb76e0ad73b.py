#!/usr/bin/env python
# coding: utf-8

# **Analysing Titanic Data (Beginner)**
# 

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

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#import train and test CSV files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#take a look at the training data
train.describe(include="all")


# In[ ]:


train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]
train.shape, test.shape


# In[ ]:


#get a list of the features within the dataset
print(train.columns)


# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train.head(10)
train.shape, test.shape


# In[ ]:


freq_port = train.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:



for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head(10)


# In[ ]:


#see a sample of the dataset to get an idea of the variables
train.sample(50)


# In[ ]:


print(train.dtypes)


# In[ ]:


#check for any other unusable values
print(pd.isnull(train).sum())


# In[ ]:


#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=train)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", train["Survived"][train["Sex"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", train["Survived"][train["Sex"] == 0].value_counts(normalize = True)[1]*100)


# In[ ]:


#draw a bar plot of survival by sex
sns.barplot(x="Pclass", y="Survived", data=train)

#print percentages of Pclass 1,2,3 that survive
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# In[ ]:


r = train.corr()
sns.heatmap(r, annot=True, square=True, fmt="0.1f", cmap='coolwarm')


# In[ ]:


#now we need to fill in the missing values in the Embarked feature
southampton = train[train["Embarked"] == 0].shape[0]
print("Number of people embarking in Southampton (S):",southampton )
print("Percentage of people embarking in Southampton (S) who survived:", train["Survived"][train["Embarked"] == 0].value_counts(normalize = True)[1]*100)
print("Percentage of people embarking in Southampton (S) who did not survive:", train["Survived"][train["Embarked"] == 0].value_counts(normalize = True)[0]*100)

cherbourg = train[train["Embarked"] == 1].shape[0]
print("Number of people embarking in Cherbourg (C):",cherbourg)
print("Percentage of people embarking in Cherbourg (C) who survived:", train["Survived"][train["Embarked"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of people embarking in Cherbourg (C) who did not survive:", train["Survived"][train["Embarked"] == 1].value_counts(normalize = True)[0]*100)

queenstown = train[train["Embarked"] == 2].shape[0]
print("Number of people embarking in Queenstown (Q):",queenstown)
print("Percentage of people embarking in Queenstown (Q) who survived:", train["Survived"][train["Embarked"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of people embarking in Queenstown (Q) who did not survive:", train["Survived"][train["Embarked"] == 2].value_counts(normalize = True)[0]*100)


# In[ ]:



fare_0 = train[train["Fare"] != None].shape[0]
print("Fares total:",fare_0 )

fare_32 = train[train["Fare"] >= 32].shape[0]
print("Fares >= 32:",fare_32 )
print("Percentage of people fare >32 who survived:", train["Survived"][train["Fare"] >= 32].value_counts(normalize = True)[1]*100,"%")

fare_15_32 = train[train["Fare"].between(15,31.99)].shape[0]
print("15 >= Fares < 32:",fare_15_32 )
print("Percentage of people 15 >= fare <32 who survived:", train["Survived"][train["Fare"].between(15,31.99)].value_counts(normalize = True)[1] * 100,"%") 

fare_8_15 = train[train["Fare"].between(8,14.99)].shape[0]
print("8 >= Fares < 15:",fare_8_15 )
print("Percentage of people 8 >= fare < 15 who survived:", train["Survived"][train["Fare"].between(8,14.99)].value_counts(normalize = True)[1]*100,"%")

fare_8 = train[train["Fare"].between(0.01,7.99)].shape[0]
print("Fares < 8:",fare_8 )
print("Percentage of people fare < 8 who survived:", train["Survived"][train["Fare"].between(0.01,7.99)].value_counts(normalize = True)[1]*100,"%")


# In[ ]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
print(test['Fare'].dropna().median())
test.head(100)


# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 8, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 15), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 32), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 32, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

combine = [train, test]
    
train.head(40)


# In[ ]:


freq_age = train.Age.dropna().mean()
freq_age
for dataset in combine:
    dataset['Age'] = dataset['Age'].fillna(freq_age)

combine = [train, test]
    
train.head(40)


# In[ ]:



age_0 = train[train["Age"] != None].shape[0]
print("Age total:",age_0 )


age_0 = train[train["Age"] < 10].shape[0]
print("Age < 10:",age_0 )
print("Percentage of people who survived:", train["Survived"][train["Age"] < 10].value_counts(normalize = True)[1]*100,"%")

age_1 = train[train["Age"].between(10,19.9999)].shape[0]
print("Age teens:",age_1 )
print("Percentage of people who survived:", train["Survived"][train["Age"].between(10,19.9999)].value_counts(normalize = True)[1]*100,"%")

age_2 = train[train["Age"].between(20,29.9999)].shape[0]
print("Age 20s:",age_2 )
print("Percentage of people who survived:", train["Survived"][train["Age"].between(20,29.9999)].value_counts(normalize = True)[1]*100,"%")

age_3 = train[train["Age"].between(30,39.9999)].shape[0]
print("Age 30s:",age_3 )
print("Percentage of people who survived:", train["Survived"][train["Age"].between(30,39.9999)].value_counts(normalize = True)[1]*100,"%")

age_4 = train[train["Age"].between(40,49.9999)].shape[0]
print("Age 40s:",age_4 )
print("Percentage of people who survived:", train["Survived"][train["Age"].between(40,49.9999)].value_counts(normalize = True)[1]*100,"%")

age_5 = train[train["Age"].between(50,59.9999)].shape[0]
print("Age 50s:",age_5 )
print("Percentage of people who survived:", train["Survived"][train["Age"].between(50,59.9999)].value_counts(normalize = True)[1]*100,"%")

age_6 = train[train["Age"]>= 60].shape[0]
print("Age 60s:",age_6 )
print("Percentage of people who survived:", train["Survived"][train["Age"]>=60].value_counts(normalize = True)[1]*100,"%")


# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Age'] < 10, 'Age'] = 0
    dataset.loc[(dataset['Age'] >= 10) & (dataset['Age'] < 20), 'Age'] = 1
    dataset.loc[(dataset['Age'] >= 20) & (dataset['Age'] < 30), 'Age'] = 2
    dataset.loc[(dataset['Age'] >= 30) & (dataset['Age'] < 40), 'Age'] = 3
    dataset.loc[(dataset['Age'] >= 40) & (dataset['Age'] < 50), 'Age'] = 4
    dataset.loc[(dataset['Age'] >= 50) & (dataset['Age'] < 60), 'Age'] = 5
    dataset.loc[ dataset['Age'] >= 60, 'Age'] = 6
    dataset['Age'] = dataset['Age'].astype(int)

combine = [train, test]
    
train.head(40)


# In[ ]:


#take a look at the training data
train.describe(include="all")


# In[ ]:


train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]
train.describe(include="all")
test.describe(include="all")


# In[ ]:


r = train.corr()
sns.heatmap(r, annot=True, square=True, fmt="0.1f", cmap='coolwarm')


# In[ ]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
X_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


# Support Vector Machines

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


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


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


pwd


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('../output/submission.csv', index=False)

