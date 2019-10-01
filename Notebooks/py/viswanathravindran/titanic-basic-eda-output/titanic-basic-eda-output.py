#!/usr/bin/env python
# coding: utf-8

# Initial EDA for the Titanic dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#Count of values in the dataset
print ('The training dataset row count is', len(train))
print ('The Test dataset row count is', len(test))


# In[ ]:


# identifying the Number of 
print ('This gives the number of missing values in training dataset', train.count())
print ('This gives the number of missing values in test dataset', test.count())


# In[ ]:


# Replacing the missing values
# train - Age, Cabin, Embarked
# test - Age, Fare, Cabin

# 1. Replace the Age in Train
tr_avage = train.Age.mean()
tr_sdage = train.Age.std()
tr_misage = train.Age.isnull().sum()
rand_age = np.random.randint(tr_avage - tr_sdage, tr_avage + tr_sdage, size=tr_misage)
train['Age'][np.isnan(train['Age'])] = rand_age
train['Age'] = train['Age'].astype(int)

# 2. Replace the Age in Test
te_avage = test.Age.mean()
te_sdage = test.Age.std()
te_misage = test.Age.isnull().sum()
rand_age = np.random.randint(te_avage - te_sdage, te_avage + te_sdage, size=te_misage)
test['Age'][np.isnan(test['Age'])] = rand_age
test['Age'] = test['Age'].astype(int)

# 3. Replace the Embarked in Train
# Distribution of Embarked in train S-644, C-168, Q-77
train['Embarked'] = train['Embarked'].fillna('S')

# 4. Treat the cabin for both test and train as a new varibale "Is_Cabin"
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# 5. Reaplce the Fare in test with a median value
med =  test.Fare.median()
test['Fare'] =  test['Fare'].fillna(med)


# In[ ]:


# Create new Features - 1. FamilySize 2. Solo traveller 3. Age bucket

# 1. FamilySize
train['FamilySize'] = train['SibSp'] + train['Parch']
test['FamilySize'] = test['SibSp'] + test['Parch']


# In[ ]:


# 2. Create New Feature Solo Traveller
train['Solo'] = train['FamilySize'].apply(lambda x: 0 if x>0 else 1)
test['Solo'] = test['FamilySize'].apply(lambda x: 0 if x>0 else 1)


# In[ ]:


# 3. Create the Age Bucket

# For Train
train['Age'] = train['Age'].astype(int)


def Age(row):
    if row['Age'] < 16:
        return 'VY'
    elif row['Age'] < 32:
        return 'Y'
    elif row['Age'] < 48:
        return 'M'
    elif row['Age'] < 64:
        return 'O'
    else:
        return 'VO'
    
train['CategoricalAge'] = train.apply(lambda row: Age(row), axis=1)
test['CategoricalAge'] = test.apply(lambda row: Age(row), axis=1)


# In[ ]:


print (train.head())


# In[ ]:


# Final Feature Selection Droping the ones which may look not necessary
drop_list = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Age']
ftrain = train.drop(drop_list, axis = 1)
ftest = test.drop(drop_list, axis = 1)


# Steps below are for Feature Engineering:
# 1. Identify and treat the missing values
# 2. Identify the relationship between each variable against the survival
# 3. Final set of features to be considered for running machine learning
# My credit goes to Sina's very comprehensive guide for feature engineering ideas. Please go through his work to see too : [Titanic Best Working][1] Classfier & [Classifier by barryhunt][2] 
# 
# 
#   [1]: https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier
#   [2]: https://www.kaggle.com/barryhunt/titanic/introduction-to-ensembling-stacking-in-pyth-ace527

# For us to proceed further Identify the set of hypothesis on the data  that you would want to test, Some of mine were:
# 
#  1. Passenger with a higher Pclass(1-Upper,2-Middle&3-Lower) should have paid a higher fare and would have higher survival rate.
#  2. While the variable denotes embankment location (C= Cherbourg, Q= Queenstown, S=
#     Southampton). Has the boarding location for passengers given a better chance of survival. The boarding order was Southampton -> Cherbourg -> Queenstown. 
#  3. Is there a case where younger members are able to somehow find a way to survive better, regardless of the Gender

# In[ ]:


# looking up PClass, Fare with Survival
# the result of the Hypothesis reads as Passengers with better Pclass survived better and strangely the corelation 
# does extend to the Fareclass
print (ftrain[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (ftrain[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
print (ftrain[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


# In[ ]:


# Before Visualization we will have to convert the Categorical Variables into clasess
# 1. Map the variable Sex
ftrain['Sex'] = ftrain['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
ftest['Sex'] = ftest['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# 2. Map the variable Embarked
ftrain['Embarked'] = ftrain['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
ftest['Embarked'] = ftest['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# 3. Map the Categorical Age
ftrain['CategoricalAge'] = ftrain['CategoricalAge'].map( {'VY': 0, 'Y': 1, 'M': 2, 'O': 3, 'VO': 4} ).astype(int)
ftest['CategoricalAge'] = ftest['CategoricalAge'].map( {'VY': 0, 'Y': 1, 'M': 2, 'O': 3, 'VO': 4} ).astype(int)


# In[ ]:


print (ftrain.head())
print (ftest.head())


# In[ ]:


# Creating the X and Y for both Train and Test
y_train = ftrain['Survived'].ravel()
ftrain = ftrain.drop(['Survived'], axis=1)
x_train = ftrain.values # Creates an array of the train data
x_test = ftest.values # Creats an array of the test data
#Xtrain = ftrain['Pclass','Sex', 'Fare', 'Embarked', 'Has_cabin', 'FamilySize', 'Solo', 'CategoricalAge']
#Ytrain = ftrain['Survived']


# In[ ]:


# Visualization for the Data in Train
pd.tools.plotting.scatter_matrix(ftrain.loc[:, ["Pclass", "Sex", "Embarked", "Has_Cabin", "FamilySize"]], diagonal="kde")
plt.show()


# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(ftrain.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


# Using the Classifiers
from sklearn.metrics import accuracy_score, log_loss
#from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

clf1= RandomForestClassifier()
clf1.fit(x_train, y_train)
pred1 = clf1.predict(x_test)


# In[ ]:


final_sub1 = pd.DataFrame({ 'PassengerId': test.PassengerId,
                            'Survived': pred1 })
final_sub1.to_csv("Sub2.csv", index=False)


# In[ ]:




