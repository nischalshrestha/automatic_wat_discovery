#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Workflow stages
The competition solution workflow goes through seven stages described in the Data Science Solutions book.
1) Question or problem definition.
2) Acquire training and testing data.
3) Wrangle, prepare, cleanse the data.
4) Analyze, identify patterns, and explore the data.
5) Model, predict and solve the problem.
6) Visualize, report, and present the problem solving steps and final solution.
7) Supply or submit the results.
'''
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#--------------------------------------------------------
# 1) Question or problem definition.
#--------------------------------------------------------

#--------------------------------------------------------
# 2) Acquire training and testing data.
#--------------------------------------------------------
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

#--------------------------------------------------------
# 3) Wrangle, prepare, cleanse the data.
#--------------------------------------------------------
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# We decide to retain the new Title feature for model training.
for dataset in combine: dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# We can replace many titles with a more common name or classify them as Rare
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
# We can convert the categorical titles to ordinal.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.    
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

#--------------------------------------------------------
# 4) Analyze, identify patterns, and explore the data.
#--------------------------------------------------------

# Converting a categorical feature
for dataset in combine: dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
# Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.    
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
    for i in range(0, 2):
        for j in range(0, 3): dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)

#Let us create Age bands and determine correlations with Survived.
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

# Let us replace Age with ordinals based on these bands.
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

# We can not remove the AgeBand feature.
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# Create new feature combining existing features

# We can create a new feature for FamilySize
for dataset in combine: dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
# We can create another feature called IsAlone.
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
# Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# We can also create an artificial feature combining Pclass and Age.
for dataset in combine: dataset['Age*Class'] = dataset.Age * dataset.Pclass
    
# Completing a categorical feature
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine: dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# Converting categorical feature to numeric
for dataset in combine: dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
# Quick completing and converting a numeric feature
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# We can not create FareBand.
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

# Convert the Fare feature to ordinal values based on the FareBand.
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

#--------------------------------------------------------
# 5) Model, predict and solve the problem.
#--------------------------------------------------------
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

# Logistic Regression
logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(X_train, Y_train)
Y_pred_logreg = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

# Support Vector Machines
svc = SVC(gamma = 'auto') #gamma will change from 'auto' to 'scale'
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gaussian = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

# Perceptron
perceptron = Perceptron(max_iter=1000)
perceptron.fit(X_train, Y_train)
Y_pred_perceptron = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# Linear SVC
linear_svc = LinearSVC(max_iter=300000) #ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
linear_svc.fit(X_train, Y_train)
Y_pred_linear_svc = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=1000)
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_decision_tree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_random_forest = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

#--------------------------------------------------------
# 6) Visualize, report, and present the problem solving steps and final solution.
#--------------------------------------------------------
# Model evaluation
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
# Model	Score
# 3	Random Forest	86.76
# 8	Decision Tree	86.76
# 1	KNN	84.74
# 0	Support Vector Machines	83.84
# 2	Logistic Regression	80.36
# 6	Stochastic Gradient Decent	79.69
# 7	Linear SVC	78.90
# 5	Perceptron	76.99
# 4	Naive Bayes	72.28

# Ensemble of 5 best predictions
def find_mean_predicton(ndarrays=[], weights=[]):
    # 1) dublicate some arrays
    for w in weights: 
        array = ndarrays[weights.index(w)]
        for x in range(w-1): ndarrays.append(array)
    # 2) find mean = sum / len
    s = np.zeros(len(ndarrays[0]))  
    for x in ndarrays: s += x
    return s/len(ndarrays) # ndarray([...])

# this ensenble gives 0.77511
# Y_pred = find_mean_predicton([Y_pred_random_forest, Y_pred_decision_tree, Y_pred_knn, Y_pred_svc, Y_pred_logreg], weights=[3,3,2,1,1]).round() #array([ 13, 133, 1333, 4, 8])

# Random forest gives 0.78468
Y_pred = Y_pred_random_forest

# Decision gives 0.78468
# Y_pred = Y_pred_decision_tree

# KNN gives 0.77990
# Y_pred = Y_pred_knn

# this ensenble gives 0.77511
# Y_pred = find_mean_predicton([Y_pred_random_forest, Y_pred_decision_tree, Y_pred_knn], weights=[3,3,1]).round() #array([ 13, 133, 1333, 4, 8])

#--------------------------------------------------------
# 7) Supply or submit the results.
#--------------------------------------------------------
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": Y_pred.astype(int)})
#submission.to_csv('../output/gender_submission.csv', index=False)
submission


# In[ ]:




