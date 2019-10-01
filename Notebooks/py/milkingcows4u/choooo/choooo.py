#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load packages
import numpy as np  
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import re
import seaborn as sns
sns.set_style('whitegrid')

from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

print ("Read in packages from numpy, pandas, sklearn, seaborn & matplotlib")


# In[ ]:


# Load training data
train_set = pd.read_csv("../input/train.csv")
test_set  = pd.read_csv("../input/test.csv")
print ("Read in training, test data as Panda dataframes")

# Review input features - Part 1
print ("\n\n---------------------")
print ("TRAIN SET INFORMATION")
print ("---------------------")
print ("Shape of training set:", train_set.shape, "\n")
print ("Column Headers:", list(train_set.columns.values), "\n")
print (train_set.describe(), "\n\n")
print (train_set.dtypes)

print ("\n\n--------------------")
print ("TEST SET INFORMATION")
print ("--------------------")
print ("Shape of test set:", test_set.shape, "\n")
print ("Column Headers:", list(test_set.columns.values), "\n")
print (test_set.describe(), "\n\n")
print (test_set.dtypes)


# In[ ]:


# Review input features (train set) - Part 2A
missing_values = []
nonumeric_values = []

print ("TRAINING SET INFORMATION")
print ("========================\n")

for column in train_set:
    # Find all the unique feature values
    uniq = train_set[column].unique()
    print ("'{}' has {} unique values" .format(column,uniq.size))
    if (uniq.size > 25):
        print("~~Listing up to 25 unique values~~")
    print (uniq[0:24])
    print ("\n-----------------------------------------------------------------------\n")
    
    # Find features with missing values
    if (True in pd.isnull(uniq)):
        s = "{} has {} missing" .format(column, pd.isnull(train_set[column]).sum())
        missing_values.append(s)
    
    # Find features with non-numeric values
    for i in range (1, np.prod(uniq.shape)):
        if (re.match('nan', str(uniq[i]))):
            break
        if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)', str(uniq[i]))):
            nonumeric_values.append(column)
            break
  
print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print ("Features with missing values:\n{}\n\n" .format(missing_values))
print ("Features with non-numeric values:\n{}" .format(nonumeric_values))
print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
# Review input features (test set) - Part 2B
missing_values = []
nonumeric_values = []

print ("TEST SET INFORMATION")
print ("====================\n")

for column in test_set:
    # Find all the unique feature values
    uniq = test_set[column].unique()
    print ("'{}' has {} unique values" .format(column,uniq.size))
    if (uniq.size > 25):
        print("~~Listing up to 25 unique values~~")
    print (uniq[0:24])
    print ("\n-----------------------------------------------------------------------\n")
    
    # Find features with missing values
    if (True in pd.isnull(uniq)):
        s = "{} has {} missing" .format(column, pd.isnull(test_set[column]).sum())
        missing_values.append(s)
    
    # Find features with non-numeric values
    for i in range (1, np.prod(uniq.shape)):
        if (re.match('nan', str(uniq[i]))):
            break
        if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)', str(uniq[i]))):
            nonumeric_values.append(column)
            break
  
print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print ("Features with missing values:\n{}\n\n" .format(missing_values))
print ("Features with non-numeric values:\n{}" .format(nonumeric_values))
print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


# Notes about input features
# --------------------------  
# 
# ###Size of training data
# Shape of dataframe: (891, 11+1) 
#  
# ###Size of test data
# Shape of dataframe: (418, 11) 
# 
# ###Output Feature (1)
# Survived (0 | 1)
#  
# ###Input Features (11)  
# PassengerId [1 2 3 ... ]      
# Pclass      [1 2 3]  
# Name        ['Braund, Mr. Owen Harris' 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)' 'Heikkinen, Miss. Laina' ...]   
# Sex         ['male' 'female']  
# Age         [22. 38. 26. ...]   
# SibSp       [0 1 2 3 4 5 8]  
# Parch       [0 1 2 3 4 5 6 (9)]  
# Ticket   ['A/5 21171' 'PC 17599' 'STON/O2. 3101282' ... ]  
# Fare     [7.25  71.2833  7.925 ... ]  
# Cabin    [nan 'C85' 'C123' 'E46' ... ]  
# Embarked ['S' 'C' 'Q' nan]
# 
# ###Features w/ missing values (3 train, 3 test)
# Cabin (687, 327)  
# Age (177, 86)  
# Embarked (2, 0)  
# Fare (0, 1)
# 
# ###Features w/ non-numeric values (5)
# Name  
# Sex  
# Ticket  
# Cabin  
# Embarked

# In[ ]:


# Feature Cleaning
# Convert non-numeric values for Sex, Embarked
# male=0, female=1
train_set.loc[train_set["Sex"] == "male", "Sex"]   = 0
train_set.loc[train_set["Sex"] == "female", "Sex"] = 1

test_set.loc[test_set["Sex"] == "male", "Sex"]   = 0
test_set.loc[test_set["Sex"] == "female", "Sex"] = 1

# Handle Parch=9 found only in test
# replace by value 6 which is the closest available in training data
test_set.loc[test_set["Parch"] == 9, "Parch"] = 6

# S=0, C=1, Q=2
train_set.loc[train_set["Embarked"] == "S", "Embarked"] = 0
train_set.loc[train_set["Embarked"] == "C", "Embarked"] = 1
train_set.loc[train_set["Embarked"] == "Q", "Embarked"] = 2

test_set.loc[test_set["Embarked"] == "S", "Embarked"] = 0
test_set.loc[test_set["Embarked"] == "C", "Embarked"] = 1
test_set.loc[test_set["Embarked"] == "Q", "Embarked"] = 2

# Substitute missing values for Age, Embarked & Fare
train_set["Age"]      = train_set["Age"].fillna(train_set["Age"].median())
train_set["Fare"]     = train_set["Fare"].fillna(train_set["Fare"].median())
train_set["Embarked"] = train_set["Embarked"].fillna(train_set["Embarked"].median())

test_set["Age"] = test_set["Age"].fillna(test_set["Age"].median())
test_set["Fare"] = test_set["Fare"].fillna(test_set["Fare"].median())

print ("Converted non-numeric features for Sex & Embarked...\nSubstituted missing values for Age, Embarked & Fare")


# In[ ]:


# Pclass - Visualize the features and their impact on outcomes
# Two subplots, the axes array is 1-d
print ("VISUALIZING TRAINING DATA - FEATURES vs OUTCOME")
print ("===============================================\n\n")

f, ((axis1,axis2), (axis3,axis4), (axis5,axis6)) = plt.subplots(3,2, sharey=True, figsize=(20,30))

sns.barplot(x='Pclass', y='Survived', data=train_set, order=[1, 2, 3], ax=axis1, palette="Blues_d")
axis1.set_xticklabels(['First', 'Second', 'Third'], rotation=0)

sns.barplot(x='Sex', y='Survived', data=train_set, order=[0, 1], ax=axis2, palette="Blues_d")
axis2.set_xticklabels(['M', 'F'], rotation=0)

sns.barplot(x='Embarked', y='Survived', data=train_set, order=[0, 1, 2], ax=axis3, palette="Blues_d")
axis3.set_xticklabels(['S', 'C', 'Q'], rotation=0)

sns.barplot(x='SibSp', y='Survived', data=train_set, order=[0, 1, 2, 3, 4, 5, 8], ax=axis4, palette="Blues_d")

sns.barplot(x='Parch', y='Survived', data=train_set, order=[0, 1, 2, 3, 4, 5, 6], ax=axis5, palette="Blues_d")


# Print survival rate for each feature
for feature in ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']:
    feature_survived = pd.crosstab(train_set[feature], train_set["Survived"])
    feature_survived_frac = feature_survived.apply(lambda r: r/r.sum(), axis=1)
    print ("Tables for {}\n\n{}\n\n{}\n" .format(feature, feature_survived, feature_survived_frac))
    print ("-----------------------------------\n")


# In[ ]:


# Longer X-Axis for age, fare
f, (axis1,axis2,axis3,axis4) = plt.subplots(4,1, sharey=True, figsize=(20,15))
survival_by_age = train_set[['Age', 'Survived']].groupby(['Age'],as_index=False).mean() 
sns.barplot(x='Age', y='Survived', data=survival_by_age, ax=axis1, palette="Blues_d")

train_set['age_group'] = train_set.apply(lambda r: int(r.Age/2.5), axis=1)
test_set['age_group']  = test_set.apply(lambda r: int(r.Age/2.5), axis=1)
survival_by_agegroup = train_set[['age_group', 'Survived']].groupby(['age_group'],as_index=False).mean() 
sns.barplot(x='age_group', y='Survived', data=survival_by_agegroup, ax=axis2, palette="Blues_d")

survival_by_fare = train_set[['Fare', 'Survived']].groupby(['Fare'],as_index=False).mean() 
sns.barplot(x='Fare', y='Survived', data=survival_by_fare, ax=axis3, palette="Blues_d")

train_set['fare_group'] = train_set.apply(lambda r: int(r.Fare/6.0), axis=1)
test_set['fare_group']  = test_set.apply(lambda r: int(r.Fare/6.0), axis=1)
survival_by_faregroup = train_set[['fare_group', 'Survived']].groupby(['fare_group'],as_index=False).mean() 
sns.barplot(x='fare_group', y='Survived', data=survival_by_faregroup, ax=axis4, palette="Blues_d")


# In[ ]:


Notes about input features
--------------------------  

###Size of training data


# In[ ]:


# Features used for training
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'age_group', 'fare_group']

# Train / Test split for original training data
# Withold 5% from train set for testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    train_set[predictors], train_set["Survived"], test_size=0.05, random_state=0)

print ("Original Training Set: {}\nTraining Set: {}\nTesting Set(witheld): {}" .format(train_set.shape, X_train.shape,X_test.shape))


# Normalize features - both training & test (withheld & final)
scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
final_test_transformed  = scaler.transform(test_set[predictors])

print ("Transformed training, test sets (withheld & final)")

# Scoring Metric - Accuracy
print ("Use accuracy as the score function")


# In[ ]:


X_new = np.delete(X_train_transformed, [1,3], axis=1)
X_new.shape
X_train_transformed.shape


# In[ ]:


# Assess Feature importance
# Initialize the algorithm
# Defaults to mean accuracy as score
feature_labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_train1 = np.delete(X_train_transformed, [7,8], axis=1)

alg = RandomForestClassifier(random_state=1, n_estimators=10000, min_samples_split=50, min_samples_leaf=1)
clf = alg.fit(X_train1, y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train1.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                             feature_labels[indices[f]], 
                             importances[indices[f]]))

labels_reordered = [ feature_labels[i] for i in indices]
    
plt.title('Feature Importances')
plt.bar(range(X_train1.shape[1]), 
         importances[indices],
         color='lightblue', 
         align='center')
plt.xticks(range(X_train1.shape[1]), labels_reordered, rotation=90)
plt.xlim([-1, X_train1.shape[1]])
plt.tight_layout()
plt.show()


# In[ ]:


# Assess Feature importance
# Initialize the algorithm
# Defaults to mean accuracy as score
feature_labels = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'age_group', 'fare_group']
X_train2 = np.delete(X_train_transformed, [2,5], axis=1)

alg = RandomForestClassifier(random_state=1, n_estimators=10000, min_samples_split=50, min_samples_leaf=1)
clf = alg.fit(X_train2, y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train2.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                             feature_labels[indices[f]], 
                             importances[indices[f]]))

labels_reordered = [ feature_labels[i] for i in indices]
    
plt.title('Feature Importances')
plt.bar(range(X_train2.shape[1]), 
         importances[indices],
         color='lightblue', 
         align='center')
plt.xticks(range(X_train2.shape[1]), labels_reordered, rotation=90)
plt.xlim([-1, X_train2.shape[1]])
plt.tight_layout()
plt.show()


# In[ ]:


# Use a simple model
# Initialize the algorithm
# Defaults to mean accuracy as score
alg = RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=5, min_samples_leaf=3)
clf = alg.fit(X_train_transformed, y_train)

# Scores
train_score = clf.score(X_train_transformed, y_train)
test_score  = clf.score(X_test_transformed, y_test)
print ("Train Score: %0.3f\nTest Score: %0.3f" %(train_score, test_score))


# In[ ]:


# Use Cross Validation
scores = cross_validation.cross_val_score(clf, X_train_transformed, y_train, cv=3)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
# Use GridSearchCV
# Specify parameters
param_grid = {"n_estimators": [200, 300, 500],
              "max_depth": [None],
              "max_features": [5],
              "min_samples_split": [9],
              "min_samples_leaf": [6],
              "bootstrap": [True],
              "criterion": ["gini"]}
             
clf = RandomForestClassifier()

grid_search = GridSearchCV(clf, param_grid=param_grid)
grid_search.fit(X_train_transformed, y_train)
print (grid_search.best_estimator_) 

# Scores
train_score = grid_search.score(X_train_transformed, y_train)
test_score  = grid_search.score(X_test_transformed, y_test)
print ("Train Score: %0.3f\nTest Score: %0.3f" %(train_score, test_score))


# In[ ]:


# Use Random Forest with Best Parameters
clf_final = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=5, max_leaf_nodes=None,
            min_samples_leaf=6, min_samples_split=9,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
clf_final.fit(X_train_transformed, y_train)

# Scores
train_score = clf_final.score(X_train_transformed, y_train)
test_score  = clf_final.score(X_test_transformed, y_test)
print ("Train Score: %0.3f\nTest Score: %0.3f" %(train_score, test_score))

#CV
scores = cross_validation.cross_val_score(clf_final, X_train_transformed, y_train, cv=3)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


# Make Predictions using Test Set
predictions = clf_final.predict(final_test_transformed)

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('titanic_rf4.csv', index=False)

submission.head(15)

