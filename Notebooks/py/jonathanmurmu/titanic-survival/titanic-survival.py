#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Read the data - train, test, combine(both test and test to run certain operations.**

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_combine = [df_train, df_test]


# **Find features available in the dataset.**

# In[ ]:


df_train.columns.values


# In[ ]:


df_test.columns.values


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# **Find features with null, empty or blank values.**
# 
# Age, Cabin and Embarked has missing values.

# In[ ]:


df_train.corr()


# **Find Numerical values**
# 
# Sibsp, Fare, Parch, age

# **Find categorical values**
# 
# survived, pclass, cabin, embarked, sex

# **Find mixed data types features(mix for numeric and alphanumeric).**
# 
# Ticket,  Cabin

# In[ ]:


df_train.describe(include='all')


# **Drop unnecessary columns.**
# 
# Ticket and Cabin

# In[ ]:


df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)
combine = [df_train, df_test]


# In[ ]:


df_train.shape, df_test.shape


# **Create new features**
# 
# New categorical feature called Title can be derived from the Name.
# Title value can be extracted by fetching the word that has .(dot) in the end.

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_train['Title'], df_train['Sex'])


# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


df_train.head()


# Title has string values. Convert them to numbers.

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset.Title.map(title_mapping)
    dataset['Title'] = dataset.Title.fillna(0)

df_train.head()


# **Name, passengerId can be droped now**

# In[ ]:


df_train = df_train.drop(['PassengerId', 'Name'], axis=1)
df_test = df_test.drop(['Name'], axis=1)
combine = [df_train, df_test]
df_train.shape, df_test.shape


# In[ ]:


df_train.describe(include='all')


# **Convert string categorical string features to numeric**

# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} )


# In[ ]:


df_train.head()


# **Complete Missing values.**
# 
# Lets complete the missing Age values with median.

# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
    print('-'*20)
    print(guess_df)
    print('-'*20)
    print(guess_ages)
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# **Complete Missing values.**
# 
# Lets complete the missing Embark values with median.

# In[ ]:


# Find out the rows with missing Embark values.
df_train[df_train['Embarked'].isnull()]


# The Pclass is 1 and fare is 80 for the above missing Embarked row.
# So lets guess what will be the Embarked value for such entry.

# In[ ]:


sns.boxplot(x='Embarked', y='Fare', hue='Pclass', data=df_train)


# In[ ]:


df_train['Embarked'] = df_train['Embarked'].fillna('C')


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# **Fill missing values in Test dataset for Fare column**
# 
# Fill it with the median

# In[ ]:


df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)


# In[ ]:


df_test.info()


# **Now both train and test data does not contain missing values.**

# **Convert categorical feature to numeric**

# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df_train.head()


# **Analyse feature correlation by pivoting features with target features.**

# In[ ]:


df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# **Analyse by visualization**

# In[ ]:


g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# **Model and Predict**

# In[ ]:


X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]
X_test  = df_test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# **Logistic Regression**

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# **Support Vector Machines**

# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# **k-Nearest Neighbors **

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# **Gaussian Naive Bayes**

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# **Perceptron**

# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# **Linear SVC**

# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# **Stochastic Gradient Descent**

# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# **Decision Tree**

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# **Random Forest**

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# **Model Evaluation**

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


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:


submission.to_csv('first_submission.csv', index=False)


# In[ ]:




