#!/usr/bin/env python
# coding: utf-8

# # Changes/Improvs to be done:
# - Proper Formatting
# - Adding Title as a Feature
# - HyperParameter tuning

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import sklearn
import re

from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.preprocessing import OneHotEncoder, LabelEncoder




import os
print(os.listdir("../input"))



# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

toy = train.copy(deep=True)

data_cleaner = [train,test]


# In[ ]:


# print(train.info())
# print(toy.info())

print(train.describe())

print(toy.sample(10))


# In[ ]:


print("missing values:")


print(train.isnull().sum())
print(test.isnull().sum())

print("cleaning..")


for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)
    
    
print(train.isnull().sum())
print(test.isnull().sum())


# In[ ]:


train['has_cabin'] = train['Cabin'].apply(lambda x:0 if type(x) == float else 1)
test['has_cabin'] = test['Cabin'].apply(lambda x:0 if type(x) == float else 1)
print(train.sample(10))
print(test.sample(10))
print(data_cleaner)


# In[ ]:


for dataset in data_cleaner:
    dataset['FamilySize']  = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['isAlone'] = 1 
    dataset['isAlone'].loc[dataset['FamilySize'] > 1] = 0
    dataset['AgeBin'] = pd.cut(dataset['Age'], 5)
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)


# In[ ]:


label = LabelEncoder()

for dataset in data_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])


# In[ ]:


for dataset in data_cleaner:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# In[ ]:


for dataset in data_cleaner:
    dataset['Title'] = dataset['Title'].replace(['Don', 'Capt', 'Col', 'Major', 'Sir', 'Jonkheer', 'Rev', 'Dr'], 'Honored')
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Dona', 'Mme', 'Countess'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])


# In[ ]:


train.sample(10)


# In[ ]:


drop_list = ['Cabin','Name','PassengerId', 'Sex', 'Age', 'Fare', 'Ticket', 'Embarked','Title']
train = train.drop(drop_list, axis=1)
# test = test.drop(drop_list,axis=1)

train = train.drop(['AgeBin', 'FareBin'], axis=1)


# In[ ]:



train.sample(15)



# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


Target = ['Survived']
train_features = ['Pclass', 'SibSp', 'Parch', 'has_cabin', 'FamilySize', 'isAlone', 'Sex_Code', 'Embarked_Code', 'AgeBin_Code', 'FareBin_Code', 'Title_Code']
data_train = train[train_features]
data_train.head(10)

MLA_predict = train[Target]
MLA_predict.head(10)


# In[ ]:


MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    

    XGBClassifier()    
    ]




cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) 


MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

row_index = 0
for alg in MLA:

    
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    
    cv_results = model_selection.cross_validate(alg, train[train_features], train[Target], cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   
    

    
    alg.fit(train[train_features], train[Target])
    MLA_predict[MLA_name] = alg.predict(train[train_features])
    
    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#MLA_predict


# In[ ]:


MLA_predict


# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(MLA_predict.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


vote_est = [
    
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),

    
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    
    
    ('lr', linear_model.LogisticRegressionCV()),
    
    
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    
    
    ('knn', neighbors.KNeighborsClassifier()),
    
    
    ('svc', svm.SVC(probability=True)),
    

   ('xgb', XGBClassifier())

]



vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, train[train_features], train[Target], cv  = cv_split)
vote_hard.fit(train[train_features], train[Target])


# In[ ]:


test.describe()


# In[ ]:


df = pd.DataFrame()


# In[ ]:


test['Survived'] = vote_hard.predict(test[train_features])


# In[ ]:


test.head(10)


# In[ ]:


submit = test[['PassengerId', 'Survived']]
submit.head(10)


# In[ ]:


submit.to_csv("../working/submit.csv", index=False)

