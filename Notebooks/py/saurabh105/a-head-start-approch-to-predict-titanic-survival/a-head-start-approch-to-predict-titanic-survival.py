#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

print(os.listdir("../input"))


# **Read Data Files**

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# **Analyse the features**

# In[ ]:


train_df.columns


# Identify categorical and numerical features

# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# <p>Survived, Sex, Embarked are the categorical features and Pclass is ordinal feature.</p>
# <p>Age and Fare comes in continous features.</p>
# <p>SibSp and Parch comes under discrete features.</p>
# <p>It will help us while visualization.</p>

# **Check datatypes of columns**

# In[ ]:


train_df.dtypes


# Check which column contains null values

# In[ ]:


train_df.info()


# In train dataset, columns ['Age',  'Cabin',  'Embarked'] contains null values. 

# In[ ]:


test_df.info()


# In test dataset, columns ['Age',  'Cabin',  'Fare'] contains null values. 

# In[ ]:


train_df.describe()


# We can drop Ticket, PassengerId, Cabin and Name as it seems very less important related to survival, and Cabin contains very less amount of data. Filling Cabin values by any other way may make it bias towards a particular result.

# In[ ]:


pid = test_df.PassengerId


# <h4>Now lets see how categorical featurees are related to Survival(target)</h4>

# For Sex vs Survival

# In[ ]:


train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()


# For Pclass vs Survival

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# For SibSp vs Survival

# In[ ]:


train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# For Parch vs Survival

# In[ ]:


train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# **Analyse continous features**
# <p>For Age</p>

# In[ ]:


train_df.Age.describe()


# In[ ]:


sns.FacetGrid(train_df, col='Survived').map(plt.hist, "Age")


# We also have missing values in age feature, we will fill it later.

# In[ ]:


sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6).map(plt.hist, 'Age', alpha=.5, bins=20)


# Most of the passengers from Pclass 1 survived. Pclass 3 contains most passengers but they didn't survived.

# In[ ]:


sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6).map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)


# Higher paying passengers had better survival rate.

# Drop unrelated features.

# In[ ]:


train_df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
test_df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)


# **Create new features**

# In[ ]:


train_df.drop(['Name', 'PassengerId'], axis=1, inplace=True)
test_df.drop(['Name', 'PassengerId'], axis=1, inplace=True)


# In[ ]:


train_df.shape


# In[ ]:


train_df.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# Convert categorical features into ordinal

# In[ ]:


enc = LabelEncoder()
enc.fit(train_df.Sex)
train_df.Sex = enc.transform(train_df.Sex)


# In[ ]:


enc = LabelEncoder()
enc.fit(test_df.Sex)
test_df.Sex = enc.transform(test_df.Sex)


# In[ ]:


enc = LabelEncoder()
train_df.Embarked = train_df.Embarked.fillna(method='ffill')
enc.fit(train_df.Embarked)
train_df.Embarked = enc.transform(train_df.Embarked)


# In[ ]:


enc = LabelEncoder()
enc.fit(test_df.Embarked)
test_df.Embarked = enc.transform(test_df.Embarked)


# In[ ]:


train_df.Age.fillna(train_df.Age.mean(), inplace=True)
test_df.Age.fillna(test_df.Age.mean(), inplace=True)
train_df.Age = train_df.Age.astype(int)
test_df.Age = test_df.Age.astype(int)


# In[ ]:


train_df.loc[ train_df['Age'] <= 16, 'Age'] = 0
train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1
train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2
train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3
train_df.loc[ train_df['Age'] > 64, 'Age'] = 4


# In[ ]:


test_df.loc[ test_df['Age'] <= 16, 'Age'] = 0
test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32), 'Age'] = 1
test_df.loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48), 'Age'] = 2
test_df.loc[(test_df['Age'] > 48) & (test_df['Age'] <= 64), 'Age'] = 3
test_df.loc[ test_df['Age'] > 64, 'Age'] = 4


# In[ ]:


test_df.Fare = test_df.Fare.fillna(test_df.Fare.mean())
train_df.loc[ train_df['Fare'] <= 7.91, 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1
train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare']   = 2
train_df.loc[ train_df['Fare'] > 31, 'Fare'] = 3
train_df['Fare'] = train_df['Fare'].astype(int)


# In[ ]:


test_df.loc[ test_df['Fare'] <= 7.91, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare']   = 2
test_df.loc[ test_df['Fare'] > 31, 'Fare'] = 3
test_df['Fare'] = test_df['Fare'].astype(int)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_df.drop(['Survived'], axis=1), train_df.Survived, test_size=0.2, random_state=0)


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(test_df)
acc_log = round(logreg.score(X_test, Y_test) * 100, 2)
acc_log


# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(test_df)
acc_svc = round(svc.score(X_test, Y_test) * 100, 2)
acc_svc


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(test_df)
acc_knn = round(knn.score(X_test, Y_test) * 100, 2)
acc_knn


# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(test_df)
acc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 2)
acc_gaussian


# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(test_df)
acc_perceptron = round(perceptron.score(X_test, Y_test) * 100, 2)
acc_perceptron


# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(test_df)
acc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 2)
acc_linear_svc


# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(test_df)
acc_sgd = round(sgd.score(X_test, Y_test) * 100, 2)
acc_sgd


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(test_df)
acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
acc_decision_tree


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(test_df)
acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
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


submission = pd.DataFrame({
        "PassengerId": pid,
        "Survived": Y_pred
    })


# In[ ]:


#submission.to_csv('../input/submission.csv', index=False)


# In[ ]:




