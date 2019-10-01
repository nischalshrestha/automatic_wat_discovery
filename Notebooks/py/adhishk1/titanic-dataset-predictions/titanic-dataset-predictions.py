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


# **Introduction**
# 
# This is my first notebook of machine learning. This kernel is inspired from [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)

# In[ ]:


#Importing required Libraries
#data analysis
import pandas as pd
import numpy as np
import random as rnd

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

#machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#acquiring the training and testing datasets 
train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
complete_set = [train_set, test_set]


# **Analyzing the Data**

# Analyzing the training set

# In[ ]:


#preview training data
train_set.head()


# In[ ]:


train_set.columns


# In[ ]:


train_set.info()


# Taining data has missing values in Age,Cabin and Embarked Columns

# In[ ]:


train_set.describe()


# Analyzing Test Set

# In[ ]:


#preveiw test data
test_set.head()


# In[ ]:


test_set.info()


# Test set has missing values in Age and Cabin columns

# In[ ]:


test_set.describe()


# In[ ]:


#Describing the correlation between the columns
# Set figure size with matplotlib
plt.figure(figsize=(10,6))
sns.heatmap(train_set.corr(),annot=True)


# From the heatmap we can conclude the correlation 

# Analyzing the realtion between every column with "Survived" using plots

# In[ ]:


sns.countplot(x="Pclass",hue="Sex" ,data=train_set)


# In[ ]:


#Analyzing for Pclass
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_set);


# From this barplot we can see that survival rate is dependent on Classs (as most of the females survived from Class 1).
# To confirm this relation we will analyze by pivoting the PClass column

# In[ ]:


#Pivioting features
train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# Here we conclude that most number of people survived from classs 1 and least from class 3

# Analyzing the relation with Age column

# In[ ]:


sns.boxplot(x="Survived", y="Age", hue="Sex", data=train_set);


# We will divide age into bins for getting more valuable insights from the data

# In[ ]:


X = sns.FacetGrid(train_set, col='Survived')
X.map(plt.hist, 'Age', bins=20)


# So we conclude from the above garph that we have to divide age into bins (chnage from continous to discreete type) 

# Analyzing the realtion of Sex column

# In[ ]:


sns.countplot(x="Sex" ,data=train_set)


# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train_set);


# As we can see from the above barplot female survival rate is much greater than male survival rate.
# We will group Sex and Survived columns to get a general idea of survival for each gender.

# In[ ]:


#Pivoting for Sex
train_set[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()


# Analyzing the realtion of SibSp and Parch column

# In[ ]:


#Analyzing for SibSp
sns.barplot(x="SibSp", y="Survived", hue="Sex", data=train_set);


# In[ ]:


#Analyzing for SibSp
train_set[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()


# #Analyzing the realtion with Parch Column

# In[ ]:


#Analyzing for SibSp
sns.barplot(x="Parch", y="Survived", data=train_set);


# In[ ]:


#Analyzing for Parch
train_set[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()


# Analyzing the realtion with Embarked Column

# In[ ]:


sns.countplot(x="Embarked",data=train_set);


# Most passengers embarked from port 'S' i.e Southampton

# In[ ]:


sns.barplot(x="Embarked", y="Survived", data=train_set);


# In[ ]:


sns.barplot(x="Embarked", y="Survived",hue="Sex", data=train_set);


# In[ ]:


#Analyzing the realtion between every column with "Survived"
#Analyzing for Embarked
train_set[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()


# Analyzing the realtion with Embarked Column

# In[ ]:


#Analyzing the realtion between every column with "Survived"
#Analyzing for Fare
sns.swarmplot(x="Survived", y="Fare",hue='Pclass', data=train_set);


# Here we can see the fare differece of each class 

# In[ ]:


y = sns.FacetGrid(train_set, col='Survived')
y.map(plt.hist, 'Fare', bins=10)


# We ignore Name, Ticket and Cabin Columns for now, we will deal with them later

# **Cleaning the Data**

# First we identify the columns which contains the missing values

# In[ ]:


#Total Nuber of missing Values in train _set
train_set.isnull().sum()


# Dealing with missing values in Age

# In[ ]:


#complete missing age with median
train_set['Age'].fillna(train_set['Age'].median(), inplace = True)


# In[ ]:


train_set.Age.isnull().sum()


# All missing values replaced with median in  Age column of the train_set DataFrame

# In[ ]:


#complete embarked with mode
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace = True)


# In[ ]:


train_set.Embarked.isnull().sum()


# Dealing with missing values in Test Set

# In[ ]:


test_set.isnull().sum()


# In[ ]:


#complete missing age with median
test_set['Age'].fillna(test_set['Age'].median(), inplace = True)
#complete embarked with mode
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace = True)
#complete missing fare with median
test_set['Fare'].fillna(test_set['Fare'].median(), inplace = True)


# In[ ]:


#Total Nuber of missing Values in test_set
test_set.isnull().sum()


# **Creating New Features **
# 
# We will create some new features from the exisiting ones in order to represent our data well, and provide a relevant data to our models.
# 1. Create FamilySize feature by adding SibSp and Parch
# 2. Create IsAlone Feature from FamilySize, i.e if FamilySize is 1 then the person is traveling alone
# 3. Create bins for Age and Fare Columns
# 4. Create Title feature by extracting title from Name column

# In[ ]:


#Creating new features in both train_set and test_set
for dataset in complete_set:    
    #Two features combined in one
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #1 means passanger is travelling Alone
     # now update value of IsAlone wherever FamilySize is greater than 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 
   

    #Regular Expression to split title from name
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


    
#Cleanup Title Column from less frequent Titles
for dataset in complete_set:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#preview data again
train_set.info()
test_set.info()

#for the warning below
#https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas


# **Convert Formats**
# We will convert categorical data to dummy variables for mathematical analysis. There are multiple ways to encode categorical variables; we will use the sklearn and pandas functions.

# In[ ]:


#Backup data before converting it to numerical value
train_backup=train_set
test_backup=test_set
complete_set_backup=complete_set
#Convert objects to category using Label Encoder for train_set and test_set
from sklearn.preprocessing import LabelEncoder
#code categorical data
label = LabelEncoder()
for dataset in complete_set:    
    dataset['Sex'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked'] = label.fit_transform(dataset['Embarked'])
    dataset['Title'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin'] = label.fit_transform(dataset['FareBin'])


# In[ ]:


train_set.head()


# In[ ]:


test_set.head()


# In[ ]:


print(train_set.shape)
print(test_set.shape)


# Input train_set for a model

# In[ ]:


train_model=train_set[['Survived','Pclass','Sex','Embarked','IsAlone','Title','FareBin','AgeBin']]
train_model.head()


# Input test_set for a model 

# In[ ]:


test_model=test_set[['Pclass','Sex','Embarked','IsAlone','Title','FareBin','AgeBin']]
test_model.head()


# **Model and Prediction**
# Now we are ready to train a model and predict the required solution. We will use the following regression algorithms:
# 
# * Logistic Regression
# * KNN or k-Nearest Neighbors
# * Support Vector Machines
# * Naive Bayes classifier
# * Decision Tree
# * Random Forrest
# * Perceptron 
# *  Linear SVC
# * Stochastic Gradient Descent
# * Decision Tree
# * Random Forest

# In[ ]:


X_train = train_model[['Pclass','Sex','Embarked','IsAlone','Title','FareBin','AgeBin']]
Y_train = train_model["Survived"]
X_test  = test_model
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
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


#K nearest neighbours
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
from sklearn.linear_model import Perceptron
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
from sklearn import linear_model
sgd = linear_model.SGDClassifier(max_iter=1000)
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


#Comparing the scores
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


submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:


#Submit output as CSV File
submission.to_csv('submission.csv', index=False)

