#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import re
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
fullset = [train, test] 
train.columns
train.head()


# The following EDA and feature engineering are inspired  by Sina: https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
# 

# In[ ]:


# Feature Engineering

# Let's tackle each feature one by one
# PassengerId, leave it there?
# Survived: our target
# Pclass
train.Pclass.isnull().value_counts() # there are no null values
# train.Sex.value_counts()
# train.Sex.isnull().value_counts()
# test.Sex.isnull().value_counts()
for dataset in fullset:
    # Sex: let's convert Sex into binary variable. Non-binary shouldn't exist back then right?
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)
    


# In[ ]:


# for age we need imputation
for dataset in fullset:
    age_mean = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_imp = np.random.randint(age_mean - age_std,age_mean + age_std, size = age_null_count)
    
    dataset['Age'][np.isnan(dataset['Age'])] = age_imp
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['Categorical_Age'] = pd.qcut(dataset['Age'],4)
    
print(train[['Survived','Categorical_Age']].groupby(['Categorical_Age'],as_index = False).mean())
for dataset in fullset:
    dataset.loc[dataset['Age']<=21.0,'Age'] = 0
    dataset.loc[(dataset['Age']>21.0)&(dataset['Age']<=28.0),'Age'] = 1
    dataset.loc[(dataset['Age']>28.0)&(dataset['Age']<=38.0),'Age'] = 2
    dataset.loc[dataset['Age']>38.0,'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)
    
print(dataset['Age'].value_counts())


# In[ ]:


# Create feature FamilySize from sibsp and parch
for dataset in fullset:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# In[ ]:


train['Fare'].describe()


# In[ ]:


# Fare

test['Fare'].isnull().value_counts() # only one missing value in testing set
# we can just use mean imputation
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
for dataset in fullset:
    dataset['Categorical_Fare'] = pd.qcut(dataset['Fare'],4)
    
print(train[['Survived','Categorical_Fare']].groupby('Categorical_Fare',as_index = False).mean())
print(train['Categorical_Fare'].value_counts())
print(test['Categorical_Fare'].value_counts())

for dataset in fullset:
    dataset.loc[dataset['Fare']<=7.91,'Fare'] = 0
    dataset.loc[(dataset['Fare']<=14.454) & (dataset['Fare']>7.91),'Fare'] = 1
    dataset.loc[(dataset['Fare']<=31.0) & (dataset['Fare']>14.454),'Fare'] = 2
    dataset.loc[dataset['Fare']>31.0,'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

print(train['Fare'].value_counts())
    


# In[ ]:


# Cabin Number
train.Cabin.isnull().value_counts()
# perhaps no need for using this feature??


# In[ ]:


# Embarked
train.Embarked.isnull().value_counts() # 2 missing values
test.Embarked.isnull().value_counts()  # no missing values
# impute using median
train['Embarked'] = train['Embarked'].fillna('S')
# train.Embarked.value_counts() # 2 missing values


# converting
embarked_mapping = {'C':0,'Q':1,'S':2}
for dataset in fullset:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)

print(train[['Survived','Embarked']].groupby(['Embarked'],as_index = False).mean())


# In[ ]:


# credit to Sina 
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in fullset:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))


# In[ ]:


# Here let's be creative and get some manual feature different from the original
for dataset in fullset:
    dataset['Title'] = dataset['Title'].replace(['Capt','Major','Col'],'Military')
    dataset['Title'] = dataset['Title'].replace(['Countess','Don','Dona','Jonkheer','Lady','Master','Sir'],'Nobility')
    dataset['Title'] = dataset['Title'].replace(['Dr','Rev'],'Educated')
    #common sense replacement
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(train[['Survived','Title']].groupby(['Title'],as_index = False).mean())
print(test['Title'].value_counts())


# In[ ]:


title_mapping = {'Educated': 0, 'Military':1,'Miss':2,'Mr':3,'Mrs':4,'Nobility':5}
for dataset in fullset:
    dataset['Title'] = dataset['Title'].map(title_mapping).astype(int)
    
print(train['Title'].value_counts())


# In[ ]:


# preparing training and testing dataset
train_X = train.drop(['Survived','Name','Ticket','Categorical_Age','Categorical_Fare',
                     'PassengerId','SibSp','Parch','Cabin'],axis = 1)
# test_X = test.drop(['Name','Ticket'],axis = 1)
train_y = train['Survived']
test_X = test.drop(['Name','Ticket','Categorical_Age','Categorical_Fare',
                     'PassengerId','SibSp','Parch','Cabin'],axis = 1)
# retain only numpy array
train_X = train_X.values
train_y = train_y.values
test_X = test_X.values



# Investigate different classifiers:
# * Logistic Regression
# * SVM
# * Decision Tree
# * Random Forest
# * AdaBoost
# * Gradient Boosting Classifier
# * Multilayer perceptron
# * Gaussian Naive Bayes
# * Linear Discriminant Analysis
# * K-Nearest Neighbor
# 
# 
# 

# In[ ]:


# import model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

sss = StratifiedShuffleSplit(n_splits = 10,test_size= 0.1,random_state = 10)
sss.split(train_X,train_y)
classifiers = [SVC(),
               RandomForestClassifier(),
               AdaBoostClassifier(),
               GradientBoostingClassifier(),
               LogisticRegression(),
               GaussianNB(),
               KNeighborsClassifier(),
               LinearDiscriminantAnalysis(),
               MLPClassifier(),
               DecisionTreeClassifier()]

acc_table = {} # a dictionary store the prediction
for train_index, test_index in sss.split(train_X,train_y):
    train_X_cv, test_X_cv = train_X[train_index],train_X[test_index]
    train_y_cv, test_y_cv = train_y[train_index],train_y[test_index]
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(train_X_cv,train_y_cv)
        predict_y = clf.predict(test_X_cv)
        acc = accuracy_score(test_y_cv,predict_y)
        if name in acc_table:
            acc_table[name] += acc
        else:
            acc_table[name] = acc

for name in acc_table:
    acc_table[name] = acc_table[name]/10.0
print(acc_table)


# In[ ]:


# print()
# acc_df = pd.DataFrame(acc_table.items(),columns = ['Classifier','Accuracy'])

acc_df = pd.DataFrame(list(acc_table.items()),columns = ['Classifier','Accuracy'])
# acc_df.index.name = 'Classifier'
# acc_df.reset_index()
acc_df = acc_df.sort_values('Accuracy',ascending = 0)
# acc_df


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=acc_df, color="b")

