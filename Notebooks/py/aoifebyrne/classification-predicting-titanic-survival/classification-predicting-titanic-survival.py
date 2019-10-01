#!/usr/bin/env python
# coding: utf-8

# ## Introduction ##
# 
# This notebook is written in Python. 
# 
# Steps:
# 
#  1. Explore and visualize the data.
#  2. Feature engineering and imputing missing data
#  3. Compare the accuracy of classifiers
#  4. Predict survival using an ensemble of classifiers
# 
# ###Question and problem definition###
# 
# "  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. "
# 
# "In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy."

# In[ ]:


#Import libraries and data

import numpy as np
import pandas as pd
import seaborn as sns
import re as re
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

full_data = [train, test]


# 

# In[ ]:


print(train.head())
train.describe()
train.info()


# 

# 

# In[ ]:


from numpy import corrcoef

corrcoef(train["PassengerId"], train["Survived"])


# 

# In[ ]:


print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())


# 

# In[ ]:


print (train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean())


# In[ ]:


train["Famsize"] = train["Parch"]+ train["SibSp"]+1
test["Famsize"] = test["Parch"]+ test["SibSp"]+1


print (train[["Famsize", "Survived"]].groupby(['Famsize'], as_index=False).count())


# 

# In[ ]:


#No parents

print(train.loc[(train["Parch"]==0) & (train["Age"]<18)])

train["LoneChild"] = 0
train.loc[(train["Parch"]==0) & (train["Age"]<18), "LoneChild"] = 1


# In[ ]:


print (train[["LoneChild", "Survived"]].groupby(['LoneChild'], as_index=False).mean())
sns.factorplot(x="LoneChild", y="Survived", data=train)


# 

# In[ ]:


train = train.drop("LoneChild", 1)


# 

# In[ ]:


for dataset in [train, test]:
    dataset['Alone'] = 0
    dataset.loc[dataset['Famsize'] == 1, 'Alone'] = 1
    
sns.factorplot(x="Alone", y="Survived", data=train)


# 

# In[ ]:


sns.factorplot('Embarked','Survived', data=train)


# In[ ]:


print("Mean")
print (train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean())

print("Count")
print (train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).count())


# In[ ]:


sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0])


# In[ ]:


train["Embarked"] = train["Embarked"].fillna('S')
test["Embarked"] = test["Embarked"].fillna('S')


# 

# In[ ]:


for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())


# 

# In[ ]:


for dataset in [train, test]:
    avg_age = dataset['Age'].mean()
    std_age = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    random_age = np.random.randint(avg_age - std_age , avg_age + std_age , size=age_null_count)
    dataset['Age'][dataset['Age'].isnull()] = age_null_count
    dataset['Age'] = dataset['Age'].astype(int)
    


# In[ ]:


np.corrcoef(train["Age"], train["Survived"])


# In[ ]:


# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(train, hue="Survived",aspect=3)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()


# 

# In[ ]:


for dataset in [train, test]:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(train['Title'].value_counts())


# In[ ]:


#cleaning up the title column

for data_set in [train, test]:
    data_set['Title'] = data_set['Title'].replace('Mlle', 'Ms')
    data_set['Title'] = data_set['Title'].replace('Miss', 'Ms')
    data_set['Title'] = data_set['Title'].replace('Mme', 'Mrs')
    data_set['Title'] = data_set['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')


# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder

categories = ['Embarked','Sex','Title']

for cat in categories:
    train[cat] = LabelEncoder().fit_transform(train[cat])
    test[cat] = LabelEncoder().fit_transform(test[cat])


# In[ ]:


drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp',                 'Parch']

train = train.drop(drop_elements, axis = 1)
test = test.drop(drop_elements, axis = 1)


# In[ ]:


#check everything looks good

train.info()


# In[ ]:


###Comparing algorithms#


# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    [KNeighborsClassifier(3),'KNN'],
    [SVC(probability=True), 'SVC'],
    [DecisionTreeClassifier(),'Decision Tree'],
    [RandomForestClassifier(),'Random Forest'],
    [AdaBoostClassifier(),'ADA booster'],
    [GradientBoostingClassifier(),'Gradient Booster'],
    [GaussianNB(),'Gaussian Nb'],
    [LinearDiscriminantAnalysis(),'Linear Discriminant Analysis'],
    [QuadraticDiscriminantAnalysis(),'Quadratic Discrimination'],
    [LogisticRegression(),'Logistic Regression']]


X = train.drop("Survived",axis=1)
y = train["Survived"]
X_test  = test



scores = []

for clf in classifiers:
    
    clf = clf[0]
    
    clf.fit(X,y)
    y_pred = clf.predict(X_test)
    
    cv_scores = cross_val_score(clf, X, y, cv=5)

    #score = clf.score(X,y)
    scores.append(cv_scores.mean())
    


# In[ ]:


#viewing classifier scores

names = [clf[1] for clf in classifiers]


np.column_stack((names, scores))


# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

test["PassengerId"] = test["PassengerId"].astype(int)

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Famsize", "Title", "Alone"]

algorithms = [GaussianNB(), LinearDiscriminantAnalysis(), GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
LogisticRegression(random_state=1),RandomForestClassifier(random_state=1, n_estimators = 50, min_samples_split=4, min_samples_leaf=2)]


predictions = []
train_target = train["Survived"]
full_test_predictions = []

    # Make predictions for each algorithm on each fold
for alg in algorithms:
        # Fit the algorithm on the training data
    alg.fit(train[predictors], train_target)
        # Select and predict on the test fold 
        # We need to use .astype(float) to convert the dataframe to all floats and avoid an sklearn error
    test_predictions = alg.predict_proba(test[predictors])[:,1]
    full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme&#8212;just average the predictions to get the final classification
test_predictions = ( sum(full_test_predictions) / len(full_test_predictions) )
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction
test_predictions[test_predictions <= .5] = 0
test_predictions[test_predictions > .5] = 1
predictions.append(test_predictions)

# Put all the predictions together into one array
predictions = np.concatenate(predictions, axis=0).astype(int)



submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv('titanic-predictions-4.csv', index = False)


# In[ ]:




