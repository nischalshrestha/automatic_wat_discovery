#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
get_ipython().magic(u'matplotlib inline')


# In[4]:


data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# # Exploratory Data Analysis

# In[5]:


data_train.head()


# In[6]:


plt.style.use('fivethirtyeight')
sns.barplot(x='Sex', y='Survived', data=data_train)


# In[7]:


sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=data_train)


# In[8]:


data_train['Age'].hist(bins=70)


# In[9]:


data_train['Age'].quantile([0, 0.25, .75, .9])


# In[10]:


def age_categorize(age):
    """Categorize Age by three Category (Young, Adult, Senior) using Age Quantile"""
    if age < 20.125:
        return 'Young'
    elif age >= 20.125 and age <= 38.000:
        return 'Adult'
    else:
        return 'Senior'        


# In[11]:


data_train['Age'] = data_train['Age'].fillna(data_train['Age'].mean())


# In[12]:


data_train['Age_Category'] = data_train['Age'].apply(age_categorize)


# In[13]:


data_train.info()


# In[14]:


sns.barplot(x='Age_Category', y='Survived', data=data_train)


# In[15]:


sns.factorplot(x='Age_Category', kind='count',hue='Survived' ,data=data_train)


# In[16]:


data_train.head()


# In[17]:


sns.distplot(data_train.Fare)


# In[18]:


data_train.Fare.quantile([.25, .5, .75])


# In[19]:


def fare_categorize(age):
    """Categorize Fare by three Category (Low, Middle, High) using Age Quantile"""
    if age < 7.9104:
        return 'Low'
    elif age >= 14.4542 and age <= 31.0000:
        return 'Middle'
    else:
        return 'High' 


# In[20]:


data_train['Fare_Category'] = data_train.Fare.apply(fare_categorize)


# In[21]:


data_train.head()


# In[22]:


sns.barplot(x='Fare_Category', y='Survived', data=data_train)


# In[23]:


sns.barplot(x='Embarked', y='Survived', data=data_train)


# In[24]:


data_train['Is_Alone'] = data_train['SibSp'] + data_train['Parch'] == 0


# In[25]:


sns.barplot(x='Is_Alone', y='Survived', data=data_train)


# # Preprocessing Data

# In[26]:


data_train.head()


# # Make Predictions

# In[27]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[28]:


train_df.head(6)


# In[29]:


def preproccesing_data(df):
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Age_Category'] = (df['Age'].apply(age_categorize)).map({'Young':0, 'Adult':1, 'Senior':2})
    df['Fare_Category'] = (df.Fare.apply(fare_categorize)).map({'Low':0, 'Middle':1, 'High':2})
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})
    df['Sex'] = df['Sex'].map({'male':0, 'female':1}).astype(int)
    df_final = df.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1)
    return df_final


# In[30]:


train_final = preproccesing_data(train_df)
test_final = preproccesing_data(test_df)


# In[31]:


test_final


# # Prediction

# In[32]:


X_train = train_final.drop("Survived", axis=1).fillna(0.0)
Y_train = train_final["Survived"]
X_test  = test_final.copy()
X_train.shape, Y_train.shape, X_test.shape


# In[33]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[35]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[36]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[37]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[38]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[39]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[40]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[41]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[42]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[44]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[47]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




