#!/usr/bin/env python
# coding: utf-8

# Hey everyone! Welcome to my submission for the Titanic kernel on Kaggle. This is my first Kaggle submission so any feedback would be welcome. 
# 
# **Overview**
# 
# The data has been split into two groups:
# 
#     training set (train.csv)
#     test set (test.csv)
# 
# The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
# 
# The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
# 
# 
# **Data Dictionary**
# 
# Variable	Definition	Key
# survival 	Survival 	0 = No, 1 = Yes
# pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
# sex 	Sex 	
# Age 	Age in years 	
# sibsp 	# of siblings / spouses aboard the Titanic 	
# parch 	# of parents / children aboard the Titanic 	
# ticket 	Ticket number 	
# fare 	Passenger fare 	
# cabin 	Cabin number 	
# embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton
# 
# **Variable Notes**
# 
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import warnings; warnings.simplefilter('ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Extracting features from data
def extract_features(X):
    """
    Format data and prepare features for use in model. One hot encodes categorical data and remove columns for features 
    are not used. 

    Parameters
    ----------
    
    X : pandas dataframe that contains either test set or train set data. Shape is (m, n) where
    m is the number of training examples and n is the number of features. 
    
    """
    X = X.drop("PassengerId", axis=1)
    X = X.drop("Ticket", axis=1)
    X = X.drop("Cabin", axis=1)
    
    # Adding polynomial features
    X["Age2"] = X["Age"] ** 2
    #X["Fare2"] = X["Fare"] ** 2
    #X["Pclass2"] = X["Pclass"] ** 2

    
    male_titles = set(["Mr", "Don", "Sir"])
    female_titles = set(["Miss", "Ms", "Mrs", "Mme", "Mdm", "Lady"])
    professionals = set(["Dr", "Rev", "Master"])
    military = set(["Col", "Major", "Capt"])
    royalty = set(["the Countess", "Jonkheer"])
    
    names = X["Name"]
    for i in range(len(names)): 
        name_tokens = names[i].split(", ") 
        passenger_title = name_tokens[1].split(".")[0]
        if passenger_title in male_titles:
            names[i] = 1
        elif passenger_title in female_titles:
            names[i] = 2
        elif passenger_title in professionals:
            names[i] = 3
        #elif passenger_title in royalty:
        #    names[i] = 4
        elif passenger_title in military:
            names[i] = 5
        else:
            names[i] = 6
    
    X["Name"].update(names)
    
    # One hot encoding of categorical data
    X = pd.get_dummies(X)    
    
    X.fillna(0, inplace=True)
    X['Fam'] = X['SibSp'] + X['Parch']  # assigned to a column
    return X


# In[ ]:


train_data = pd.read_csv('../input/train.csv')

X_train = extract_features(train_data)


# In[ ]:


# Extract y vector from train_set
y = X_train["Survived"]
X_train = X_train.drop("Survived", axis=1)


# In[ ]:


X_train


# In[ ]:


scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(X_train)

X_train = scaler.transform(X_train)


# In[ ]:


# Train model
model = SVC()

model.fit(X_train, y)


# In[ ]:


# Format test data

test_data = pd.read_csv('../input/test.csv')

X_test = extract_features(test_data)


# In[ ]:


# Use model to make predictions on test set
X_test_scaled = scaler.transform(X_test)

predictions = model.predict(X_test_scaled)


# In[ ]:


# Write predictions to a CSV file
csvfile = open('output.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

csvwriter.writerow(["PassengerId", "Survived"])
for x in range(len(predictions)):
    csvwriter.writerow([test_data["PassengerId"][x], predictions[x]])

    
csvfile.close()


# In[ ]:


# Calculate train set error
train_predictions = model.predict(X_train)
train_error = sum(abs(train_predictions - y.values)) / len(y.values)

print("Train set error: ", train_error)


# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curve for SVM with RBF Kernel"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = SVC()
plot_learning_curve(estimator, title, X_train, y, ylim=(0.7, 1.01), cv=cv)

title = "Learning Curve for LinearSVC"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = LinearSVC()
plot_learning_curve(estimator, title, X_train, y, ylim=(0.7, 1.01), cv=cv)

plt.show()

