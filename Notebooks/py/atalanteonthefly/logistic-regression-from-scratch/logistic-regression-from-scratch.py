#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[3]:


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../input/'


# In[4]:


train_csv = pd.read_csv(DATA_DIR + 'train.csv')
test_csv = pd.read_csv(DATA_DIR + 'test.csv')

print('There are %s examples in the training set and %s examples in the test set' % (train_csv.shape[0], test_csv.shape[0]))
print('\n')
print('The different variables that we have at our disposal are : %s' % ', '.join(list(train_csv.columns)))


# # Who Should Live And Die
# 
# Based on some a priori we can assume that some categories of people are, unfortunately, more inclined to to die that others.

# In[5]:


plt.figure()
plt.suptitle('Proportion of male and female that survived')
g = sns.countplot(x="Sex", hue='Survived', data=train_csv);
plt.show()


# We can see that if you were a woman on board you had far more luck to stay alive.

# In[6]:


plt.figure()
plt.suptitle('Proportion of people that survived depending on their socio-economic status')
g = sns.countplot(x="Pclass", hue='Survived', data=train_csv);
plt.show()


# Again we can see that chances are not equal for everyone. If you are from a 'lower' status you had far more chance to die that if you came from a 'upper' status.

# # Model Building

# ## Useful functions

# In[7]:


def label_encoding(dataframe, labels):
    """
    Encode categorical variable into numerical values
    """

    le = LabelEncoder()
    for label in labels:
        le.fit(dataframe[label])
        dataframe[label] = le.transform(dataframe[label])

    return dataframe

def normalize_features(X_train):
    """
    Normalize the features by substracting the mean 
    and dividing by the standard deviation
    """

    for features in X_train:
        feats = X_train[features].tolist()
        mean = np.mean(feats)
        std = np.std(feats)
        feats = (feats - mean)/std
        X_train[features] = feats

    return X_train

def get_training_data():
    """
    Clean the data by processing the nan values
    and normalizing the features
    """
    train_csv = pd.read_csv(DATA_DIR + 'train.csv')

    train_csv['Cabin'] = train_csv['Cabin'].fillna('C0')
    train_csv['Embarked'] = train_csv['Embarked'].fillna('0')
    train_csv['Age'] = train_csv['Age'].fillna(train_csv['Age'].mean())
    train_csv = label_encoding(train_csv, ['Sex', 'Ticket', 'Cabin', 'Embarked'])

    X_train = train_csv[['Pclass', 'Sex', 'Age',  'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    Y_train = train_csv['Survived']

    normalize_features(X_train)

    return X_train.as_matrix(), Y_train.as_matrix()

def get_testing_data():

    test_csv = pd.read_csv(DATA_DIR + 'test.csv')

    test_csv['Cabin'] = test_csv['Cabin'].fillna('C0')
    test_csv['Embarked'] = test_csv['Embarked'].fillna('0')
    test_csv['Age'] = test_csv['Age'].fillna(test_csv['Age'].mean())
    test_csv['Fare'] = test_csv['Fare'].fillna(test_csv['Fare'].mean())
    test_csv = label_encoding(test_csv, ['Sex', 'Ticket', 'Cabin', 'Embarked'])

    X_test = test_csv[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]

    normalize_features(X_test)

    return X_test.as_matrix(), test_csv['PassengerId']


# ## Logistic regression training 

# In[8]:


X_train, Y_train = get_training_data()

X_train, Y_train = get_training_data()

# Hyperparameters initialization
lr = 0.05

# Parameters initialization
weights = np.random.normal(0, 0.1, 9)
biais = random.normalvariate(0, 0.1)

m = X_train.shape[0]
for epoch in range(300):

    # Forward pass
    Z = np.dot(X_train, weights) + biais
    A = 1 / (1 + np.exp(-Z))

    #Loss Computation
    J = np.sum(-(Y_train * np.log(A) + (1 - Y_train) * np.log(1 - A))) / m

    # Gradient computation
    dZ = A - Y_train
    dw = np.dot(dZ, X_train) / m
    db = np.sum(dZ) / m

    # Update weights
    weights = weights - lr * dw
    biais = biais - lr * db
    
    if epoch % 10 == 0:
        print("epoch %s - loss %s" % (epoch, J))


# ## Logistic regression prediction 

# In[9]:


X_test, PassengerId = get_testing_data()

preds = []
for feats in X_test:

   z = np.dot(feats, weights) + biais
   a = 1 / (1 + np.exp(-z))

   if a > 0.5:
       preds.append(1)
   elif a <= 0.5:
       preds.append(0)
     
sample_ids = np.random.choice(PassengerId, 10)

for id, value in enumerate(sample_ids):
   print('Passenger id : %s - Survived : %s' % (value, preds[id]))
   
gendermodel_csv = pd.read_csv(DATA_DIR + 'gendermodel.csv')
accuracy = accuracy_score(list(gendermodel_csv['Survived']), preds)
print('\n')
print('The accuracy of the model is of %s : ' % accuracy)

