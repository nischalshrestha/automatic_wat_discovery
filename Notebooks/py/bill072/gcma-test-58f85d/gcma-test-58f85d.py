#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import math
import time
# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing


# In[ ]:


# get train & test csv files as a DataFrame
data_train = pd.read_csv("../input/train.csv")
data_test    = pd.read_csv("../input/test.csv")
data_all = pd.concat([data_train, data_test])
# preview the data
data_all.head()


# In[ ]:


data_train.info()
print("----------------------------")
data_test.info()


# **Generate Features**

# In[ ]:


# Parse Title
data_all['Title'] = data_all.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
data_all['Title'] = data_all['Title'].replace('Mlle', 'Miss')
data_all['Title'] = data_all['Title'].replace('Ms', 'Miss')
data_all['Title'] = data_all['Title'].replace('the', 'Miss')
data_all['Title'] = data_all['Title'].replace('Mme', 'Miss')
data_all['Title'] = data_all['Title'].replace('the', 'Miss')
data_all['Title'] = data_all['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

sns.factorplot('Title','Survived', data=data_all,size=4,aspect=3)


# In[ ]:


# One hot encoding of Title
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
data_all['Title'] = data_all['Title'].map(title_mapping)
data_all['Title'] = data_all['Title'].fillna(0)
data_all['Title_Mr'] = (data_all['Title']==1) + 0
data_all['Title_Miss'] = (data_all['Title']==2) + 0
data_all['Title_Mrs'] = (data_all['Title']==3) + 0
data_all['Title_Master'] = (data_all['Title']==4) + 0
data_all['Title_Rare'] = (data_all['Title']==5) + 0


# In[ ]:


# split Sex to Male and Female
sex_mapping = {"female":1, "male":2}
data_all['Sex'] = data_all['Sex'].map(sex_mapping)
data_all['Male'] = (data_all['Sex']==2)+0
data_all['Female'] = (data_all['Sex']==1)+0


# In[ ]:


# parse Embarked
# plot



# In[ ]:


#one-hot vector for pclass


# In[ ]:


#Cabin
data_all['Cabin'] = data_all.Cabin.fillna('ZZ')
data_all['Cabin'] = data_all['Cabin'].apply(lambda x: x.split()[0][0])
data_all['Cabin_Z'] = (data_all['Cabin']=='Z')+0
data_all['Cabin_C'] = (data_all['Cabin']=='C')+0
data_all['Cabin_E'] = (data_all['Cabin']=='E')+0
data_all['Cabin_G'] = (data_all['Cabin']=='G')+0
data_all['Cabin_D'] = (data_all['Cabin']=='D')+0
data_all['Cabin_A'] = (data_all['Cabin']=='A')+0
data_all['Cabin_B'] = (data_all['Cabin']=='B')+0
data_all['Cabin_F'] = (data_all['Cabin']=='F')+0
data_all['Cabin_T'] = (data_all['Cabin']=='T')+0


# In[ ]:


# fill na for Fare
# find a better way to fill the nan value
data_all['Fare'] = data_all['Fare'].fillna(0)


# In[ ]:


# is child


# In[ ]:


# fill nam for Age
guess_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = data_all[(data_all['Sex'] == i+1) & (data_all['Pclass'] == j+1)]['Age'].dropna()
        # age_mean = guess_df.mean()
        # age_std = guess_df.std()
        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
        age_guess = guess_df.median()
# Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
for i in range(0, 2):
    for j in range(0, 3):
        data_all.loc[ (data_all.Age.isnull()) & (data_all.Sex == i+1) & (data_all.Pclass == j+1),                    'Age'] = guess_ages[i,j]

data_all['Age'] = data_all['Age'].astype(int)


# In[ ]:


# number of family
data_all['NumFamily'] = data_all['SibSp'] + data_all['Parch']


# In[ ]:


# is Mathor
data_all['Mother'] = ((data_all['Female']==1) & (data_all['Parch']>0) & (data_all['Age'] > 18) & (data_all['Title_Miss']==0)) + 0


# In[ ]:


# Drop the column that we don't need
data_all = data_all.drop(['Name', 'Ticket', 'Cabin', 'Survived', 'Title', 'Pclass', 'Sex', 'Embarked',  'PassengerId'], axis=1)
label = data_train['Survived']


# In[ ]:


# standardrization
scaler = preprocessing.StandardScaler().fit(data_all)
data_all = scaler.transform(data_all)


# In[ ]:


data_train = data_all[0:891].T
data_test = data_all[891:].T
data_test_ori = data_test
data_test.shape


# **Train Calssifier**

# In[ ]:


def loadTrainDev(X, Y, numTrain, seed=0):
    m = X.shape[1]
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
    
    X_train = shuffled_X[: , 0:numTrain]
    Y_train = shuffled_Y[: , 0:numTrain]
    X_dev = shuffled_X[ : , numTrain:]
    Y_dev = shuffled_Y[ : , numTrain:]
    
    return (X_train, Y_train, X_dev, Y_dev)

# get train and dev data set
(X_train, Y_train, X_dev, Y_dev) = loadTrainDev(data_train, label.as_matrix().reshape(1, -1), 840, seed=int(time.time()))
print(X_train.shape)
print(Y_train.shape)
print(X_dev.shape)
print(Y_dev.shape)


# In[ ]:


# SVC
svc = SVC()
svc.fit(X_train.T, Y_train.T)
svc_dev_pred = svc.predict(X_dev.T).reshape(X_dev.T.shape[0],1)
svc_test_pred = svc.predict(data_test.T).reshape(data_test.T.shape[0], 1)
acc_svc = round(svc.score(X_dev.T, Y_dev.T) * 100, 2)
acc_svc


# In[ ]:


# KNN
N_ITERATION = 300
N_SAMPLES = 4
bestParameters = None
acc_knn = 0
best_n_neighbors = 0;
for i in range(0,N_ITERATION):
    np.random.seed(int(time.time()))
    n_neighbors = np.random.randint(1,700)
    #print("n neighbors: ", n_neighbors)
    acc_average = 0
    for j in range(0, N_SAMPLES):
        (X_train_iter, Y_train_iter, X_dev_iter, Y_dev_iter) = loadTrainDev(data_train, label.as_matrix().reshape(1, -1), 700, seed=int(time.time()))
        knn = KNeighborsClassifier(n_neighbors = n_neighbors)
        knn.fit(X_train_iter.T, Y_train_iter.ravel().T)
        acc_knn_sample = round(knn.score(X_dev_iter.T, Y_dev_iter.T) * 100, 2)
        acc_average = acc_average + acc_knn_sample
        
    acc_average = acc_average / N_SAMPLES
    #print("accuracy: ", acc_average)
    #print(" ")
    if acc_average > acc_knn:
        acc_knn = acc_average
        best_n_neighbors = n_neighbors
        
print("best accuracy: ", acc_knn)
print("number of neighbors: ", best_n_neighbors)


print(best_n_neighbors)
knn = KNeighborsClassifier(n_neighbors = best_n_neighbors)
knn.fit(X_train.T, Y_train.ravel().T)
knn_dev_pred = knn.predict(X_dev.T).reshape(X_dev.T.shape[0],1)
knn_test_pred = knn.predict(data_test.T).reshape(data_test.T.shape[0], 1)
acc_knn = round(knn.score(X_dev.T, Y_dev.T) * 100, 2)

print("accuracy: ", acc_knn)


# In[ ]:


# Random Forest


# In[ ]:


# Logistic Regression


# In[ ]:


# Can you add more classifiers?


# **Output the result with Stacking method**

# In[ ]:


def mjVote(preds, weights):
    (m, n) = preds.shape
    rst = np.zeros(m)
    for i in range(0, m):
        rst[i] = np.argmax(np.bincount(preds[i,:], weights=weights))
    return rst.astype(np.int32).reshape(m,1)
def accuracy(pred, Y):
    N = Y.shape[1]
    truePred = (pred == (Y.T)) + 0
    return round(truePred.sum()/N*100, 2)
#weight
weights = np.array([acc_svc, acc_knn], np.float)
print("Weights: ")
print(weights)

preds_dev = np.concatenate((svc_dev_pred, knn_dev_pred), axis=1)
vote_preds_dev = mjVote(preds_dev, weights)
print("dev accuracy: ")
print(accuracy(vote_preds_dev, Y_dev))

preds_test = np.concatenate((svc_test_pred, knn_test_pred), axis=1)
vote_preds_test = mjVote(preds_test, weights)

data_test_ori = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({
        "PassengerId": data_test_ori["PassengerId"]
    })
submission['Survived'] = vote_preds_test[:,0]
submission.to_csv('submission.csv', index=False)


# In[ ]:




