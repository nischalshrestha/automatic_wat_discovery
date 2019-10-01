#!/usr/bin/env python
# coding: utf-8

# ### Titanic Solution

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import numpy.random as rnd


# In[ ]:


titanic_train = pd.read_csv("../input/train.csv",sep=',')
test = pd.read_csv("../input/test.csv", sep=',')
train = titanic_train[["Parch", "Sex", "Age", "SibSp", "Fare", "Embarked","Pclass" ,"Survived"]]
train_test = [train, test]
print('Train data')
print(train.head())
print('\n\nTest data')
test.head()



# In[ ]:


print(train.info())
print('\n\nTest info')
print(test.info())


# ####  To find null values in the dataset

# In[ ]:


test.isnull().sum()


# #### Visualisations

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train)


# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train)


# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=train)


# In[ ]:


train_test = [train, test]
train_test


# In[ ]:


train['Age'].isnull().sum()


# In[ ]:


mean = train['Age'].mean()
std = train['Age'].std()
null_count = train['Age'].isnull().sum()
train['Age'][np.isnan(train['Age'])] = rnd.randint(mean-std, mean+std, size= null_count)


# In[ ]:


test['Sex'].isnull().sum()


# In[ ]:


for dataset in train_test:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


for data in train_test:
   data['Embarked']= data['Embarked'].fillna('S')


# In[ ]:


for dataset in train_test:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


for dataset in train_test:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


# In[ ]:


X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape


# In[ ]:


X_train.head()


# In[ ]:


age_mean = X_test['Age'].mean()
X_test = test[["Parch", "Sex", "Age", "SibSp", "Fare", "Embarked","Pclass"]].copy()
X_test['Age'].fillna(age_mean, inplace=True)
X_test.isnull().sum()


# In[ ]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
acc_log_reg


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_log_reg
    })
submission.head()


# In[ ]:




