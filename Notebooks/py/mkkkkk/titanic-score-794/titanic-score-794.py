#!/usr/bin/env python
# coding: utf-8

# Titanic: Machine Learning from Disaster-Beginner
# 

# Titanic:This is my 2nd submission and kernel where i could improve my score from .765 to.794
#  Here I worked on the Agegroup survival and familysize versus survival
# Contents:
# Importing Required Libraries
# Importing and Analysing the Data
# Data Visualization
# Cleaning Data
# Choosing the Best Model
# Creating Submission File
# Any and all feedback is welcome!
# 
# 

# Importing Required Libraries
# 

# In[ ]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Importing the Libraries
import pandas as pd
import numpy as np
import random as rnd

# Importing Library for visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')




# Importing data
# The Python Pandas packages helps us work with our datasets. We start by acquiring the training and testing datasets into Pandas DataFrames. .
# 
# 

# In[ ]:


#Importing the train dataset and reading
train = pd.read_csv('../input/train.csv', )

train.head()


# In[ ]:


#Importing and reading the test dataset

test = pd.read_csv('../input/test.csv')

test.head()


# In[ ]:


# Getting a summary of the dataframe using 'describe()'
train.describe()


# In[ ]:


test.head()


# Inspecting the dataframe's columns, shapes, variable types etc.

# In[ ]:


# Check the number of rows and columns in the dataframe

train.shape


# In[ ]:


test.shape


# In[ ]:


# Check the column-wise info of the dataframe

train.info()


# In[ ]:


test.info()


# In[ ]:


# Get a summary of the dataframe using 'describe()'

train.describe()


# In[ ]:


test.describe()


# Visualisation of columns which would help us in prediction the Survivalfor eg,Age,sex,Pclass,Fare

# 

# Survived
# So we can see that 62% of the people in the training set died. .
# 
# 

# In[ ]:


train['Survived'].value_counts(normalize=True)


# Pclass
# 
# Class played a critical role in survival, as the survival rate decreased drastically for the lowest class. This variable is both useful and clean
# 
# 

# In[ ]:


train['Survived'].groupby(train['Pclass']).mean


# In[ ]:




sns.barplot(x='Pclass', y='Survived', data=train)


# Sex
# 

# In[ ]:


train['Sex'].value_counts(normalize=True)

train['Survived'].groupby(train['Sex']).mean()


# Sex is the one of feature which we will use to analyse the survival.We can see from the train dataset that 74 percent female survived.Hence,female had more survival chances

# We can see that Female has more survival chance compared to male

# In[ ]:





# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train)
plt.show()


# SibSp
# 

# In[ ]:


train['Survived'].groupby(train['SibSp']).mean()


# In[ ]:


sns.barplot(x='SibSp', y='Survived', data=train)


# Parch
# 

# In[ ]:


train['Survived'].groupby(train['Parch']).mean()


# In[ ]:


sns.barplot(x='Parch', y='Survived', data=train)


# Ticket :this feature I am going to drop because it doesnt show any correlation to survival

# Cabin:this column i am going to drop as it has more null values maybe in next version and submission i will try to deal with it

#  Name Column i am going to drop as I do not see much correlation to survival

# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=train)


# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train)


# High Fare has more chances of survival

#  2: Cleaning the Data
# 
# -   Inspect Null values
# Find out the number of Null values in all the columns and rows.
# Also, find the percentage of Null values in each column. Round-off 
# the percentages upto two decimal places.

# In[ ]:


# Get the column-wise Null count using 'is.null()' alongwith the 'sum()' function
train.isnull().sum()


# Now we can see the Age of train data set has 177 and emabarked 2 null values.We need to impute the data for those null rows as these the required fields to predict the survival.
# Cabin for this submission i will drop the column

# In[ ]:


train = train.drop(['Ticket','Name','Cabin'],axis=1)


# In[ ]:


round(100*(train.isnull().sum()/len(train.index)), 2)


# In[ ]:


test.isnull().sum()


# Now we can see the Age of train data set has 86 and cabin 327  null values.i will drop the cabin as it has high null values and will deal with itin next submission
# And will impute the mean values in the Age column

# In[ ]:


# Get the percentages by dividing the sum obtained previously by the total length, multiplying it by 100 and rounding it off to
# two decimal places

round(100*(train.isnull().sum()/len(train.index)), 2)


# Cleaning the data by imputing the mean values in the missing age  

# In[ ]:


test = test.drop(columns=['Cabin','Ticket','Name'],axis=1)
# dropping the cabin column as i am not going use this for predicting the survived


# In[ ]:


#calculating the mean of the age of train data set and test
Age_mean= train['Age'].mean()
train['Age'] = train['Age'].fillna(Age_mean)
#


# In[ ]:


#most occuring embarked and filling the missing row of embarked
sns.barplot(x='Embarked', y='Survived', data=train)


# In[ ]:


#As we can see Embarked C is high so can fill missing values as C
train['Embarked'] = train['Embarked'].fillna('S')


# In[ ]:


# checking again the null values now the train dataset is clean and can do the same to the test dataset
round(100*(train.isnull().sum()/len(train.index)), 2)


# Cleaning Test Data set and imputing the values in age and embarked column as it has null values and we need it for prediction

# In[ ]:


round(100*(test.isnull().sum()/len(test.index)), 2)


# In[ ]:





# In[ ]:


#Imputing mean values in the fare column
Fare_mean = test['Fare'].mean()
test['Fare'] = test['Fare'].fillna(Fare_mean)


# In[ ]:


round(100*(test.isnull().sum()/len(test.index)), 2)


# let us plot and see how columns are related to each and to anlayse the survival rate

# #In Sklearn we cannot pass the string so we have map integer values
# to the Gender(Male  as   0 and female as 1)

# In[ ]:


gender={'male':0,'female':1}

train['Sex']=train['Sex'].apply(lambda x:gender[x])


# In[ ]:


Embarked_map={'S':1,'C':2,'Q':3}


# In[ ]:


train['Embarked']=train['Embarked'].apply(lambda x:Embarked_map[x])


# In[ ]:


test['Embarked']=test['Embarked'].apply(lambda x:Embarked_map[x])


# In[ ]:


test['Sex']=test['Sex'].apply(lambda x:gender[x])


# In[ ]:


test['Sex'].head()


# In[ ]:


test['FamilySize'] =  test['SibSp'] + test['Parch']

   


# In[ ]:


#Imputing mean values in the age column
Age_mean = test['Age'].mean()
test['Age'] = test['Age'].fillna(Age_mean)


# Let us create Age bands and determine correlations with Survived.
# 
# 

# In[ ]:


train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


combine= [train,test]
for dataset in combine:   
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
test.head()


# In[ ]:


test.head()


# Fare :As we know the People with high fare had more chances of survival

# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:



for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
combine = [train, test]
    
train.head(10)


# Let us deal with familysize

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
pd.crosstab(train['FamilySize'], train['Survived']).plot(kind='bar', stacked=True, title="Survived by family size")


# It seems that for families from 1 to 4 people, family size increases survival rates. But for families of 5 and up, survival rates is much lower.
# 
# 

# In[ ]:


train.head()


# In[ ]:


train=train[['PassengerId','Age','Survived','Pclass','Sex','Fare','FamilySize','Embarked']]


# Splitting the Training Data
# We will use part of our training data (22% in this case) to test the accuracy of our different models.
# 
# 

#  Modelling
# Choosing the best model
# Splitting the Training Data
# We will use part of our training data (20% in this case) to test the accuracy of our different models.
# 

# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)


# Testing Different Models
# I will be testing the following models with my training data:
# 
# Gaussian Naive Bayes
# Logistic Regression
# Support Vector Machines
# Decision Tree Classifier
# Random Forest Classifier
# KNN or k-Nearest Neighbors
# Stochastic Gradient Descent
# Gradient Boosting Classifier
# For each model, we set the model, fit it with 80% of our training data, predict for other 20% of the training data and check the accuracy.
# Referred from
# https://www.kaggle.com/sisakk/titanic-survival-prediction-model-building
# 

# Testing Different Models
# I will be testing the following models with my training data:
# 
# Gaussian Naive Bayes
# Logistic Regression
# Support Vector Machines
# Decision Tree Classifier
# Random Forest Classifier
# KNN or k-Nearest Neighbors
# Stochastic Gradient Descent
# Gradient Boosting Classifier
# For each model, we set the model, fit it with 80% of our training data, predict for other 20% of the training data and check the accuracy.
# 
# 

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


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[ ]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[ ]:





# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
from sklearn.linear_model import SGDClassifier




# In[ ]:


from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


#Checking test data
test.head()


# As we need only numberical data to test the model and we have done the required cleaning by filling the missing values and converting the gender to 0 and 1

# In[ ]:


#saving passenger id for submission
test_ids = test['PassengerId']


# In[ ]:


#only taking the columns required for prediction
test=test[['Age','Pclass','Sex','Fare','FamilySize','Embarked']]
test_predictions =randomforest.predict(test)


# 8. Create Submission File
# 

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived":test_predictions })

submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




