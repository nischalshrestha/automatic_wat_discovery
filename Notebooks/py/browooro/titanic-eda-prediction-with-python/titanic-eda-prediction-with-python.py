#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config BackendInline.figure_format = 'retina' # Data Visualization")

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn import ensemble, gaussian_process, linear_model, naive_bayes, neighbors, svm # Machine Learning


# In[3]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_id = test['PassengerId']
dataframes = [train, test]


# ### A first look at the dataset
# 
# [Age, Embarked] has some null values, and [Cabin] has a significant amount of null values.

# In[4]:


for data in dataframes:
    print(data.info())
    print('____________________\n')


# In[5]:


train.head()


# In[6]:


train.describe()


# ### Data Cleaning
# There are 2 main goals for cleaning this dataset:
# 1. dealing with null values
# 2. gaining more information on text data.

# In[7]:


def ticket_map(x):
    try:
        int(x)
        return 1 
    except:
        return 0

ss = StandardScaler()

# Data Cleaning
for data in dataframes:
    data['Sex'] = data['Sex'].map(lambda x: 0 if str(x) == 'male' else 1) # Changing sex (M/F) into binary values
    data['Age'] = data['Age'].fillna(data['Age'].median()) # Filling null values of age as median age
    data['Fare'] = data['Fare'].fillna(data['Fare'].median()) # the test dataset has 1 null value.
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0]) # Filling null values in Embarked as the mode.
    data['FamSize'] = data['SibSp'] + data['Parch'] # New feature "FamSize"
    data['Alone'] = data['FamSize'].map(lambda x: 1 if int(x) == 0 else 0) # binary data on if the person was alone on the ship
    data['Title'] = data['Name'].str.split(', ', expand=True)[1].str.split('. ', expand=True)[0] # splitting new features on people's titles
#     data['Ticket_code'] = data['Ticket'].apply(ticket_map) # dividing type of ticket with just numbers vs. text + numbers
    data[['Age', 'Fare']] = ss.fit_transform(data[['Age', 'Fare']]) # standardizing Age and Fare for machine learning models.
    data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1, inplace=True)


# In[8]:


train.head()


# In[9]:


train.Title.value_counts()


# ### Feature Engineering
# 
# There's a large variety of titles from the people on this ship.
# 
# I will make new features for [Embarked] & [Titles], with grouping the assumed typos and professional groups of people.

# In[10]:


for data in dataframes:
    data['Embark_S'] = data['Embarked'].map(lambda x: 1 if str(x) == 'S' else 0)
    data['Embark_C'] = data['Embarked'].map(lambda x: 1 if str(x) == 'C' else 0)
    data['Embark_Q'] = data['Embarked'].map(lambda x: 1 if str(x) == 'Q' else 0)
    data['Mr'] = data['Title'].map(lambda x: 1 if str(x) == 'Mr' else 0)
    data['Miss'] = data['Title'].map(lambda x: 1 if str(x) in ['Miss', 'Mlle', 'Ms'] else 0)
    data['Mrs'] = data['Title'].map(lambda x: 1 if str(x) in ['Mrs', 'Mme'] else 0)
    data['Master'] = data['Title'].map(lambda x: 1 if str(x) == 'Master' else 0)
    data['Professional'] = data['Title'].map(lambda x: 1 if str(x) in ['Dr', 'Major', 'Rev', 'Col', 'Capt'] else 0)
    data['Misc'] = data['Title'].map(lambda x: 1 if str(x) not in ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Major', 'Rev', 'Col', 'Capt'] else 0)
    data.drop(['Title', 'Embarked'], axis=1, inplace=True)


# In[11]:


train.head()


# In[12]:


plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), annot=True, cmap='Blues')


# In[13]:


plt.figure(figsize=(8,6))

plt.hist([train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']], 
         stacked=False, bins = 20, color = ['b','r'], label = ['Survived','Dead'])
plt.title('Histogram of Fare by Survival')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.legend()


# Looks like there are a lot of outliers in fares, and most of the deaths occured with the lower-fare passengers.

# In[14]:


plt.figure(figsize=(8,6))

plt.hist([train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 
         stacked=False, bins = 20, color = ['b','r'], label = ['Survived','Dead'])
plt.title('Histogram of Age by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()


# A lot of the deaths occured within passengers in the mean age (around 30), and younger passengers seems more likely to survive.
# 

# In[15]:


plt.figure(figsize=(15,8))
sns.violinplot(x = 'Sex', y = 'Age', hue = 'Survived', data = train, split = True)
plt.xticks([0,1],['Male', 'Female'])


# In[16]:


plt.figure(figsize=(15,8))
sns.violinplot(x = 'Sex', y = 'FamSize', hue = 'Survived', data = train, split = True)
plt.xticks([0,1],['Male', 'Female'])


# There's a high spike in survival from females with small family size, while generally people with large family size had a more difficult time surviving.

# In[17]:


X = train.drop('Survived', axis=1)
y = train.Survived


# I chose to run a voting classifier model, with using multiple models and voting on if each passenger survived or died in the predictions.
# 
# Source: https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy

# In[18]:


voting_estimates = [
    ('ada', ensemble.AdaBoostClassifier(n_estimators=200)),
    ('bc', ensemble.BaggingClassifier(n_estimators=200)),
    ('etc', ensemble.ExtraTreesClassifier(n_estimators=200)),
    ('gbc1', ensemble.GradientBoostingClassifier(n_estimators=200)),
    ('gbc2', ensemble.GradientBoostingClassifier(n_estimators=500)),
    ('rfc', ensemble.RandomForestClassifier(n_estimators=200)),
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    ('lr', linear_model.LogisticRegressionCV()),
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    ('knn5', neighbors.KNeighborsClassifier()),
    ('svc', svm.SVC(probability=True))]


# In[19]:


vote_soft = ensemble.VotingClassifier(estimators = voting_estimates , voting = 'soft')
vote_hard = ensemble.VotingClassifier(estimators = voting_estimates , voting = 'hard')
print(cross_val_score(vote_soft, X, y, cv=3).mean())
print(cross_val_score(vote_hard, X, y, cv=3).mean())
vote_soft.fit(X, y)
vote_hard.fit(X, y)


# In[20]:


prediction = vote_hard.predict(test)
sub_dict = {'PassengerId':test_id, 'Survived':prediction}
submission = pd.DataFrame(sub_dict)
submission.to_csv(path_or_buf = 'Submission.csv', index = False)

