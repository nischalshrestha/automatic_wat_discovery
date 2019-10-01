#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load modules
#data wrangling
import numpy as np
import pandas as pd
import re as re

#visualization
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
full_data = pd.concat([train, test])


# ### Preview Data

# In[ ]:


#preview data
print("train")
print (train.info())
print("\ntest")
print (test.info())
print("full_data")
print (full_data.info())
train.head(10)


# In[ ]:


#Data summary
train.describe(include = "all").transpose()


# #### Observations  in training dataset
# |Column|Data Type|Number of Missing Values|Other obervations|
# |--|--|--|--|
# |PassengerId    |integer|0|-|
# |Survived       |integer|0|Survive rate is 38%|
# |Pclass         |integer|0||
# |Name           |character|0||
# |Sex            |character|0|Over half of the passengers were male|
# |Age            |float|177|Average age is 29,75% of passengers are under 38. There were infant(6 month old) and elder people(80 years old)|
# |SibSp|integer|0|Over half of passengers travel without siblings or spouses|
# |Parch          |integer|0|Over 75% of passengers travel without parent or child|
# |Ticket         |character|0|-|
# |Fare           |float|0|Though average ticket price is 32, there is great variations on ticket price. The most expensive ones are 512|
# |Cabin          |character|687||
# |Embarked       |character|2||
# 

# In[ ]:


# Pclass
print('Pclass')
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
#Extract title from names
print('\nName(Title)')
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
#Sex
print("\nSex")
print (train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
#Age
print("\nAge")
train['CategoricalAge'] = pd.cut(train['Age'], 5)
print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
#Siblings and Spouses
print("\nSibSp")
print (train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
#Parent and child
print("\nParch")
print (train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
#Fare
print("\nFare")
train['CategoricalFare'] = pd.cut(train['Fare'], 5)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
print("\nEmbarked")
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


# ### Replace missing values and create new features(on full data)
# 
# #### Replace missing values
# 
# 1. replace missing values from Embarked with S(mode)
# 2. replace missing age with randomly generated value refering to its title
# 
# #### Add new features
# 
# 1. Create "title: from "name"
# 2. Create "travel_size" from "Parch" and "SibSp"

# In[ ]:


#1.replace missing values from Embarked with S(mode)
full_data['Embarked'] = full_data['Embarked'].fillna('S')

#2.Create "title: from "name", and collapse som of the titles
full_data['Title'] = full_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
full_data['Title'] = full_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
full_data['Title'] = full_data['Title'].replace('Mlle', 'Miss')
full_data['Title'] = full_data['Title'].replace('Ms', 'Miss')
full_data['Title'] = full_data['Title'].replace('Mme', 'Mrs')
print (full_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
print("\n")

#3.Create "TravelSize" from "Parch" and "SibSp"
full_data['TravelSize'] = full_data['SibSp'] + full_data['Parch'] + 1
#Make TravelGroup from TravelSize
full_data.loc[full_data['TravelSize'] == 1, 'TravelGroup'] = "alone"
full_data.loc[(full_data['TravelSize'] > 1) & (full_data['TravelSize'] < 5), 'TravelGroup'] = "small"
full_data.loc[full_data['TravelSize'] >= 5, 'TravelGroup'] = "big"
print (full_data[['TravelGroup', 'Survived']].groupby(['TravelGroup'], as_index=False).mean())

#4.
print("\ntitles and average age")
print (full_data[['Title', 'Age']].groupby(['Title'], as_index=False).mean())

titles = ["Master","Miss","Mr","Mrs","Rare"]
for title in titles:
    sub_dataset = full_data.loc[full_data["Title"] == title,'Age']
    age_avg = sub_dataset.mean()
    age_std    = sub_dataset.std()
    age_null_count = sub_dataset.isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    sub_dataset[np.isnan(sub_dataset)] = age_null_random_list
    full_data.loc[full_data["Title"] == title,'Age'] = sub_dataset.astype(int) 
#Notice that one observation's "fare" column is null, we find it and use mean fare of its pclass
print("There is only on row of fare missing, which is:")

print(full_data[full_data["Fare"].isnull()])
temp_mean = full_data[full_data["Pclass"] == 3]['Fare'].mean()
print("Since it is class 3, so set the fare to mean fare for class 3:(which is " + str(round(temp_mean)) +")")
full_data.loc[full_data["Fare"].isnull(),"Fare"] = full_data[full_data["Pclass"] == 3]['Fare'].mean()


# In[ ]:


print (full_data.info())
print (full_data.describe())


# We can see that, except for Cabin, there are no other columns that contain NAs.
# 
# ### Data Mapping

# In[ ]:


#Data mapping
# Mapping Sex
full_data['Sex'] = full_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
 
# Mapping titles
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
full_data['Title'] = full_data['Title'].map(title_mapping)
full_data['Title'] = full_data['Title'].fillna(0)
    
# Mapping Embarked
full_data['Embarked'] = full_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
# Mapping Fare
full_data.loc[ full_data['Fare'] <= 7.91, 'Fare'] = 0
full_data.loc[(full_data['Fare'] > 7.91) & (full_data['Fare'] <= 14.454), 'Fare'] = 1
full_data.loc[(full_data['Fare'] > 14.454) & (full_data['Fare'] <= 31), 'Fare']   = 2
full_data.loc[ full_data['Fare'] > 31, 'Fare'] = 3
full_data['Fare'] = full_data['Fare'].astype(int)
    
# Mapping Age
full_data.loc[ full_data['Age'] <= 16, 'Age'] 					       = 0
full_data.loc[(full_data['Age'] > 16) & (full_data['Age'] <= 32), 'Age'] = 1
full_data.loc[(full_data['Age'] > 32) & (full_data['Age'] <= 48), 'Age'] = 2
full_data.loc[(full_data['Age'] > 48) & (full_data['Age'] <= 64), 'Age'] = 3
full_data.loc[ full_data['Age'] > 64, 'Age']  = 4

#Mapping TravelGroup
full_data['TravelGroup'] = full_data['TravelGroup'].map( {'alone': 0, 'small': 1, 'big': 2} ).astype(int)


# In[ ]:


drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'TravelSize']
droped_dataset = full_data.drop(drop_features,axis = 1)
cols = list(droped_dataset.columns.values)
cols = cols[5:] + cols[0:5]
droped_dataset = droped_dataset[cols]
train = droped_dataset.iloc[0:891,:]
test = droped_dataset.iloc[891:,:]
print(train.head())


# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test  = test.drop("Survived",axis=1).copy()


# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

#knn
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# ### Result
# 
# We can see that: random forest has the best accuracy

# In[ ]:


Y_pred = random_forest.predict(X_test).astype(int)
test = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:


submission.to_csv('submission.csv', index=False)


# 
