#!/usr/bin/env python
# coding: utf-8

# **A small road map on solving this problem**

# 1st you should browse the data,  check the size, the nature of it, and read the requirements.

# 2nd you need to to load the data, 

# 3rd do a fast cleaning for the data to be able to implement a fast learning algorithm,

# 4th after getting the first result, even if it is a dirty implementation, it will be a good start to go through the problem again check the data cleaning, feature engineering needed, tweak the parameters , try another algorithm to compare the results.

# In[ ]:


# Import the needed referances
import pandas as pd
import numpy as np
import csv as csv

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

#Shuffle the datasets
from sklearn.utils import shuffle

#Learning curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#import seaborn as sns
#Output plots in notebook
#%matplotlib inline 


# In[ ]:


#loading the data sets from the csv files
print('--------load train & test file------')
train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')
print("------finish loading --------------------")


# In[ ]:


# print data sets information 
print('----train dataset information-------')
train_dataset.info()
print('----test dataset information--------')
test_dataset.info()
print("------------------------------------")


# In[ ]:


#Check for missing data & list them 
nas = pd.concat([train_dataset.isnull().sum(), test_dataset.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset'])
nas[nas.sum(axis=1) > 0]


# In[ ]:


# Data sets cleaing, fill nan (null) where needed and delete uneeded columns
print('----Strat data cleaning ------------')

#manage Age
train_random_ages = np.random.randint(train_dataset["Age"].mean() - train_dataset["Age"].std(),
                                          train_dataset["Age"].mean() + train_dataset["Age"].std(),
                                          size = train_dataset["Age"].isnull().sum())

test_random_ages = np.random.randint(test_dataset["Age"].mean() - test_dataset["Age"].std(),
                                          test_dataset["Age"].mean() + test_dataset["Age"].std(),
                                          size = test_dataset["Age"].isnull().sum())

train_dataset["Age"][np.isnan(train_dataset["Age"])] = train_random_ages
test_dataset["Age"][np.isnan(test_dataset["Age"])] = test_random_ages
train_dataset['Age'] = train_dataset['Age'].astype(int)
test_dataset['Age']    = test_dataset['Age'].astype(int)

# Embarked 
train_dataset["Embarked"].fillna('S', inplace=True)
test_dataset["Embarked"].fillna('S', inplace=True)
train_dataset['Port'] = train_dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_dataset['Port'] = test_dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
del train_dataset['Embarked']
del test_dataset['Embarked']

# Fare
train_dataset["Fare"].fillna(train_dataset["Fare"].median(), inplace=True)
test_dataset["Fare"].fillna(test_dataset["Fare"].median(), inplace=True)

train_dataset['Fare']    = train_dataset['Fare'].astype(int)
test_dataset['Fare'] = test_dataset['Fare'].astype(int)

# Map Sex to a new column Gender as 'female': 0, 'male': 1
train_dataset['Gender'] = train_dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_dataset['Gender'] = test_dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
# Delete Sex column from datasets
del train_dataset['Sex']
del test_dataset['Sex']

# Delete Ticket column from datasets  (No need for them in the analysis)
del train_dataset['Ticket']
del test_dataset['Ticket']

# Cabin has a lot of nan values, so i will remove it
del train_dataset['Cabin']
del test_dataset['Cabin']


# ** Engineer New Features **

# In[ ]:


# engineer a new Title feature
# Get titles from the names
train_dataset['Title'] = train_dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_dataset['Title'] = test_dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# group them
full_dataset = [train_dataset, test_dataset]
for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev'], 'Officer')
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Sir', 'Jonkheer', 'Dona'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Get the average survival rate of different titles
train_dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

##engineer the family size feature
for dataset in full_dataset:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
## Create new column "FamilySizeGroup" and assign "Alone", "Small" and "Big"
for dataset in full_dataset:
    dataset['FamilySizeGroup'] = 'Small'
    dataset.loc[dataset['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'
    dataset.loc[dataset['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Big'

## Get the average survival rate of different FamilySizes
train_dataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()

## engineer a new ischild feature
for dataset in full_dataset:
    dataset['IsChild'] = 0
    dataset.loc[dataset['Age'] <= 10, 'IsChild'] = 1
    
for dataset in full_dataset:
    dataset.loc[(dataset['Age'] <= 10), 'AgeGroup'] = 1
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'AgeGroup'] = 2
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'AgeGroup'] = 3
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'AgeGroup'] = 4
    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'AgeGroup'] = 5
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'AgeGroup'] = 6
    dataset.loc[(dataset['Age'] > 70) & (dataset['Age'] <= 80), 'AgeGroup'] = 7
    dataset.loc[(dataset['Age'] > 60) & (dataset['Age'] <= 70), 'AgeGroup'] = 8
    
# Convert to integer type
#full_dataset['AgeGroup'] = full_dataset['AgeGroup'].astype(int)
#full_dataset['AgeGroup'] = full_dataset['AgeGroup'].astype(int)

    


# In[ ]:


# map the new features
title_mapping = {"Mr": 0, "Officer": 1, "Master": 2, "Miss": 3, "Royal": 4, "Mrs": 5}
family_mapping = {"Small": 0, "Alone": 1, "Big": 2}
for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['FamilySizeGroup'] = dataset['FamilySizeGroup'].map(family_mapping)
    

# Delete Name column from datasets (No need for them in the analysis)
del train_dataset['Name']
del test_dataset['Name']

del train_dataset['SibSp']
del test_dataset['SibSp']

del train_dataset['Parch']
del test_dataset['Parch']

del train_dataset['FamilySize']
del test_dataset['FamilySize']

print('----Finish data cleaning ------------')


# In[ ]:


train_dataset.head()


# In[ ]:


##Shuffling the datasets
#train_dataset = shuffle(train_dataset)
#test_dataset = shuffle(test_dataset)
#print('Finish Shuffling the datasets')


# In[ ]:


# create a validation data set arround 20 % 
train_dataset, valid_dataset = np.split(train_dataset.sample(frac=1), [int(.8*len(train_dataset))])
print(len(train_dataset))
print(len(valid_dataset))


# In[ ]:


del train_dataset['PassengerId']
del valid_dataset['PassengerId']

X_train = train_dataset.drop("Survived",axis=1).as_matrix()
Y_train = train_dataset["Survived"].as_matrix()

X_val = valid_dataset.drop("Survived",axis=1).as_matrix()
Y_val = valid_dataset["Survived"].as_matrix()

X_test  = test_dataset.drop("PassengerId",axis=1).copy().as_matrix()

print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)


# In[ ]:


# Learning curve
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
logreg_model = LogisticRegression(C=1)
def Learning_curve_model(X, Y, model, cv, train_sizes):

    plt.figure()
    plt.title("Learning curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")


    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=cv, n_jobs=4, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
                     
    plt.legend(loc="best")
    return plt

plot_lc      = 1   # 1--display learning curve/ 0 -- don't display

#learn curve
if plot_lc==1:
    train_size=np.linspace(.1, 1.0, 15)
    Learning_curve_model(X_train,Y_train , logreg_model, cv, train_size)


# ##Fixing##: Adding features: Fixes high bias , Adding polynomial features: Fixes high bias,Decreasing λ: Fixes high bias**

# In[ ]:


# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

result_val =svc.score(X_val, Y_val)
result_train = svc.score(X_train, Y_train)
print('taring score = %s , while validation score = %s' %(result_train , result_val))


# In[ ]:


# Logistic Regression
logreg = LogisticRegression(C=1) #(C=0.1, penalty='l1', tol=1e-6)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

result_val =logreg.score(X_val, Y_val)
result_train = logreg.score(X_train, Y_train)
print('taring score = %s , while validation score = %s' %(result_train , result_val))


# In[ ]:


# Random Forests
random_forest = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

result_val =random_forest.score(X_val, Y_val)
result_train = random_forest.score(X_train, Y_train)
print('taring score = %s , while validation score = %s' %(result_train , result_val))


# In[ ]:


cof_df = pd.DataFrame(train_dataset.columns.delete(0))
cof_df.columns = ['Features']
cof_df["Coefficient Estimation"] = pd.Series(logreg.coef_[0])

#PRINT
print(cof_df)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_dataset["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
print('Exported')

