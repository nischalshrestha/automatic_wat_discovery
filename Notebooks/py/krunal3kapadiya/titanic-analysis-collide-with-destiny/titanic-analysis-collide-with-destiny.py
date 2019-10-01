#!/usr/bin/env python
# coding: utf-8

# # Titanic Analysis: Collide with destiny

# ![](https://i.imgur.com/rE1OxtK.png)

# ## Table of Contents
# - [Overview](#overview)
# - [Wrangling Data](#wranglingData)
# - [Developing Model](#developingModel)
# - [Validating Model](#validatingModel)
# - [Conclusion](#conclusion)
# > <B>NOTE</B>: This is my first Kaggle comptition kernel. Any feedback or suggestions will be warmly appreciated. 

# ----
# <a id='overview'></a>
# ## Overview
# 
# The data has been split into two groups:
# 
# - training set (train.csv)
# - test set (test.csv)
# 
# **The training set** should be used to build your machine learning models. For the training set, we provide the outcome (also known as the `ground truth`) for each passenger. Your model will be based on `features` like passengers’ gender and class. You can also use feature engineering to create new [features](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/).
# 
# **The test set** should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
# 
# > ### Variable Notes
# **pclass**: A proxy for socio-economic status (SES)<br>
# 1st = Upper<br>
# 2nd = Middle<br>
# 3rd = Lower<br>
# **age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5<br>
# **sibsp**: The dataset defines family relations in this way... <br>
# Sibling = brother, sister, stepbrother, stepsister<br>
# Spouse = husband, wife (mistresses and fiancés were ignored)<br>
# **parch**: The dataset defines family relations in this way...<br>
# Parent = mother, father<br>
# Child = daughter, son, stepdaughter, stepson<br>
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[ ]:


# importning libraries
import pandas as pd
import numpy as np

#data visualization library 
import seaborn as sns 
import matplotlib.pyplot as plt

# models libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# In[ ]:


# reading csv file
test_df = pd.read_csv('../input/test.csv')
train_df = pd.read_csv('../input/train.csv')


# ----
# <a id='wranglingData'></a>
# ## Wrangling Data

# In[ ]:


train_df.head()


# In[ ]:


train_df.tail()


# From above we can see that, Name, Sex, Ticket, Cabin, Embarked colunms have Object (String) values. 

# In[ ]:


train_df.info()


# We have total, 2 columns float type, 5 columns have integer values and 5 columns have object values.

# In[ ]:


train_df.describe()


# From above we can see that Age column have 177 missing values, also gender do not have any numeric values. <br> Adding numerica values to the gender.

# In[ ]:


genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# We have added numerica values to the `Sex` column, now, adding null values to the `Age` column.

# In[ ]:


data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()


# In[ ]:


train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


# In[ ]:


data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# Same as `Sex` columns we can add numeric values to the `Embarked` columns, This column have 3 values, S, C and Q.

# In[ ]:


common_value = 'S'
ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# In[ ]:


data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_df['not_alone'].value_counts()


# In[ ]:


import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
# we can now drop the cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# In[ ]:


data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)


# In[ ]:


data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']


# In[ ]:


for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
# Let's take a last look at the training set, before we start training the models.
train_df.head()


# In[ ]:


train_df = train_df.drop(['PassengerId'], axis=1)


# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
test_df.head()


# In[ ]:


X_train.head()


# In[ ]:


Y_train.head()


# In[ ]:


X_test.head()


# Upto now we have done wrangling with dateset, Both `X_train` and `X_test` contains similar rows.

# ----
# <a id='developingModel'></a>
# ## Developing Model

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)


# In[ ]:


logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, Y_train)
Y_prediction = logisticRegression.predict(X_test)
logisticRegression.score(X_train, Y_train)


# In[ ]:


xgBoost = XGBClassifier()
xgBoost.fit(X_train, Y_train)
Y_prediction = xgBoost.predict(X_test)
xgBoost.score(X_train, Y_train)


# ----
# <a id='validatingModel'></a>
# ## Validating Model

# I have evaluated 3 models and it's score. You can see from above score that random forest classifier is the best model  out of 3 for the dataset.

# In[ ]:


acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)


# In[ ]:


from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# In[ ]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)


# ----
# <a href='conclusion'></a>
# ## Conclusion
# Here I have implemented three ML algorithms and found the best model *random forest regression*.<br>
# 
# #### Inspired from the [End to End Project](https://www.kaggle.com/niklasdonges/end-to-end-project-with-python) with Python and it's Medium article [Predicting the survival of titanic passengers](https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8).
# 
# What will be in next version
# - Improvement accuracy of model
# - Explanatory analysis
# 
# To know more about me go to my website [https://krunal3kapadiya.app/](https://krunal3kapadiya.app/ )  <br>
# If you like this kernel, don't forgot to **upvote** it.
