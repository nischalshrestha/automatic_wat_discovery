#!/usr/bin/env python
# coding: utf-8

# <h2>Table of Contents</h2>
# 1. [ Import libraries ](#1.-Import-necessary-libraries)
# 2. [ Read and explore data ](#2.-Read-and-explore-Data)
# 3. [ Missing Values ](#3.-Missing-Values)
# 4. [ Data Visualization ](#4.-Data-Visualization)
# 5. [ Manipulation ](#5.-Data-Manipulation)
# 6. [ Modelling ](#6.-Modelling)
# 7. [ Clean test data to input to model ](#7.-Clean-Test-Data-to-input-to-model)
# 8. [ Submission ](#8- Create-Submission-File)
# 9. [ Conclusion ](#9.-Conclusion)

# <h2>1. Import necessary libraries</h2>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from math import ceil # import ceil function from math module
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# model building
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h2>2. Read and explore Data</h2>

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.describe(include='all')


# * less than 50% passengers were survived.
# * more than 50% passengers travelled in 3rd class ( pclass = 3 )
# * less than 50% passengers have atleast 1 sibling or spouse.
# * less than 25% passengers have atleast 1 parent or child.
# * mean age of passengers is 29
# * more male passengers than female (m - 577)

# In[ ]:


train.info()


# <p>We can find missing data in couple of columns.</p>
# <p>Age, Cabin, Embarked have null or missing entries</p>

# In[ ]:


train.sample(5)


# Definitions and quick thoughts for reference:
# 
# * **PassengerId**. Unique identification of the passenger. *( primary key, not needed for model building )*
# * **Survived**. Survival (0 = No, 1 = Yes). Binary variable - our target variable y.
# * <u>**Pclass**. Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd). Ready to go.</u>
# * **Name**. Name of the passenger. **Format** -> lastname, (Mr/Ms/Miss etc) first name (other name for female). Mr/Ms/Miss etc can give additional info like their occupation <i> ( will remove for now ) </i>
# * **Sex**. Sex of passenger. Categorical variable that should be encoded.
# * <u>**Age**. Age of passengers in years. Ready to go.</u>
# * <u>**SibSp**. # of siblings / spouses aboard the Titanic.</u> <i> ( combine with Parch to create a new feature family size )</i>
# * <u>**Parch**. # of parents / children aboard the Titanic.  </u>
# * **Ticket**. Ticket number. some complicated structure ( will remove for now )
# * <u>**Fare**. Passenger fare.  Ready to go.</u>
# * **Cabin**. Cabin number. It needs to be parsed.  ( will remove for now )
# * **Embarked**. Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton). Categorical feature that should be encoded.

# <h2>3. Missing Values</h2>

#     Numerical : Age, Fare -> (continous)  |  Pclass, Sibsp, Parch, pclass (discrete)
#     Categorical : Sex, Survived, Embarked
#     Alphanumerics : Ticket, Cabin

# In[ ]:


# get total and percent of missing values for each feature
total_missing = train.isnull().sum().sort_values(ascending=False)
percent_missing = (train.isnull().sum() * 100/train.isnull().count()).sort_values(ascending=False)
pd.concat([total_missing, percent_missing], axis=1, keys=['total', 'percent'])


# *<h4> Missing Value Observations </h4>*
# * Cabin has 77% missing values and it's alphanumeric. let us delete this feature.
# * Age has 19.8% missing and it's important feature in predicting the survival. We have to fill this feature.
# * Embarked has just 2 values missing. we can remove those 2 rows

# <h2> 4. Data Visualization </h2>

# In[ ]:


print(train.groupby('Sex')['Survived'].mean())
# train.plot(x='Sex', y='Survived')
# train.groupby('Sex')['Survived'].mean().plot(x='Sex')

sns.barplot(x='Sex', y='Survived', data=train).set_title('Percent Survived by Gender')


# In[ ]:


# percent survived by pclass
sns.barplot(x='Pclass', y='Survived', data=train)

#print percentage of people by Pclass that survived
print("Pclass 1:", round(train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100, 2), "% survived")
print("Pclass 2:", round(train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100, 2), '% survived')
print("Pclass 3:", round(train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100, 2), '% survived')


# In[ ]:


# passengers survived by Pclass and sex
sns.barplot(x='Pclass', y='Survived', data=train, hue='Sex')


# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train, hue='Embarked')


# In[ ]:


# import math
train_nonNullAge = train[train['Age'].notnull()]
train_nonNullAge['Age'] = train_nonNullAge['Age'].apply(lambda x: ceil(x))

plt.title('Survival rate across ages ( 0 to 1 )')
train_nonNullAge.groupby('Age')['Survived'].mean().plot(kind='line')


# People less than 20 years had high Survival rate

# <h2>5. Data Manipulation</h2>

# In[ ]:


# Combine Sibsp, Parch into Family feature and drop the 2 features
train['FamilySize'] = train['SibSp'] + train['Parch']
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[ ]:


# drop Cabin, ticket and name of passenger
train.drop(['Cabin','Name', 'Ticket'], axis=1, inplace=True)


# In[ ]:


# Remove rows with null embarked values
train.drop(train[train['Embarked'].isnull()].index, inplace=True)


# In[ ]:


# Fill age values in null fields using randint(mean-std, mean+std, count(null values))
train['Age'].fillna(value=np.random.randint(train['Age'].mean() - train['Age'].std(), train['Age'].mean() + train['Age'].std()), inplace=True)
train.isnull().any().any()


# `train.isnull().any().any()` False indicates that none of the columns in DataFrame contains a NULL entry. <br>
# So DataFrame is missing value free.

# In[ ]:


# show max 11 columns
pd.set_option('display.max_columns', 11)

# convert categoricals sex and embarked to numericals for modelling purpose

# lets map sex male to 0, female to 1
sex_map = {'female': 0, 'male': 1}

# Emabrked mapping
embarked_map = {'S': 0, 'C': 1, 'Q': 2}

train['Sex'] = train['Sex'].map(sex_map)
train['Embarked'] = train['Embarked'].map(embarked_map)

train.head()


# All missing values are treated. <br>
# Now that whole data is numerical and no data is missing, we can build a model using it.

# <h2>6. Modelling</h2>
# <h3>Choosing the best model</h3>
# * Splitting the Training Data
# * We will use part of our training data (20% in this case) to test the accuracy of our different models.

# In[ ]:


# from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)


# **Testing Different Models**<br>
# I will be testing the following models with my training data:
# 
# * Gaussian Naive Bayes
# * Logistic Regression
# * Support Vector Machines
# * Decision Tree Classifier
# * Random Forest Classifier
# * KNN or k-Nearest Neighbors
# * Stochastic Gradient Descent
# * Gradient Boosting Classifier
# 
# For each model, we set the model, fit it with 80% of our training data, predict for other 20% of the training data and check the accuracy.

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


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[ ]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[ ]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[ ]:


# Comparing accuracies of each models
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# Random Forest algorithm gives the maximum accuracy. So let's use it to test our model

# <h2>7. Clean Test Data to input to model</h2>

# In[ ]:


# Clean test data and Submit Predictions
test.describe(include='all')


# In[ ]:


# missing values for test data
total_missing_test = test.isnull().sum().sort_values(ascending=False)
percent_missing_test = (test.isnull().sum() * 100/test.isnull().count()).sort_values(ascending=False)
pd.concat([total_missing_test, percent_missing_test], axis=1, keys=['total', 'percent'])


# Remove Cabin feature, fill Age with values the way we filled for train data and mean for fare

# In[ ]:


# save passenger IDs to use in submission file
test_ids = test['PassengerId']

# drop cabin, Name, Ticket, combine Sibsp, Parch into FamiliSize and drop those 2 features
test['FamilySize'] = test['SibSp'] + test['Parch']
test.drop(['PassengerId', 'Cabin', 'SibSp', 'Parch', 'Name', 'Ticket'], axis=1, inplace=True)

test.head()


# In[ ]:


# fill with mean values for fare
test['Age'].fillna(value=np.random.randint(test['Age'].mean() - test['Age'].std(), test['Age'].mean() + test['Age'].std()), inplace=True)
test.isnull().sum()


# In[ ]:


# null record of Fare in test
test_fare_null = test[test['Fare'].isnull()].index
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

# False implying that no feature of any record is null.
test.isnull().any().any()


# In[ ]:


# converting test categoricals sex and embarked to numericals just like we did for train

# sex_map = {'female': 0, 'male': 1}

# Emabrked mapping
# embarked_map = {'S': 0, 'C': 1, 'Q': 2}

test['Sex'] = test['Sex'].map(sex_map)
test['Embarked'] = test['Embarked'].map(embarked_map)

# all data is converted to format the way we did for train data.
# test data is ready to be fed into random forrest model since it gave maximum accuracy.
test.head()


# <h2>8. Create Submission File</h2>

# In[ ]:


# randomforest

#set the output as a dataframe and convert to csv file named submission.csv
submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": randomforest.predict(test)
    })

submission.to_csv('submission.csv', index=False)


# <h2>9. Conclusion</h2>
# * Random Forest algorithm gave the best accuracy for training dataset and the same is used to predict the survival of test dataset. The accuracy of the model is 73%.
# * Accuracy of the model can be improved taking names into consideration.
# * Age and Sex are major features effecting the survival of passengers. ( female of Pclass 1 has the highest survival rate while Male of Pclass3 has the lowest survival rate according the data we observed in data visualization section )
# * Around 20% of Age values are missing and it's treatment greatly affects the model prediction. The Missing values filled using aggregate mean values can be improved using the parameters like name, title of passenger present in name and their sex.
