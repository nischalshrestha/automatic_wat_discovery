#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Inporting Data series

# In[ ]:


training = pd.read_csv("../input/train.csv")
testing = pd.read_csv("../input/test.csv")


# Lets see first 6 rows

# In[ ]:


training.head(6)


# In[ ]:


testing.head(6)


# In[ ]:


training.shape


# In[ ]:


testing.shape


# In[ ]:


training.info()


# In[ ]:


testing.info()


# In[ ]:


training.isnull().sum()


# In[ ]:


testing.isnull().sum()


# VISUALIZATION

# In[ ]:



def  bar_chart(feature):
    survived = training[training['Survived']==1][feature].value_counts()
    dead = training[training['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


bar_chart('Sex')


# In[ ]:


bar_chart('Pclass')


# In[ ]:


bar_chart('SibSp')


# In[ ]:


bar_chart('Parch')


# In[ ]:


bar_chart('Embarked')


# In[ ]:


from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")


# In[ ]:


train_test_data = [training, testing] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


training['Title'].value_counts()


# In[ ]:



testing['Title'].value_counts()


# In[ ]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


training.head()


# In[ ]:


testing.head()


# In[ ]:


bar_chart('Title')


# In[ ]:


# delete unnecessary feature from dataset
training.drop('Name', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)


# In[ ]:



training.head()


# In[ ]:


testing.head()


# In[ ]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


bar_chart('Sex')


# # Age
# Let's use Title's median age for missing Age

# In[ ]:



# fill missing age with median age for each title (Mr, Mrs, Miss, Others)# fill m 
training["Age"].fillna(training.groupby("Title")["Age"].transform("median"), inplace=True)
testing["Age"].fillna(testing.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


training.head(30)
training.groupby("Title")["Age"].transform("median")


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, training['Age'].max()))
facet.add_legend()
 
plt.show()


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, training['Age'].max()))
facet.add_legend()
plt.xlim(0, 20)


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, training['Age'].max()))
facet.add_legend()
plt.xlim(20, 30)


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, training['Age'].max()))
facet.add_legend()
plt.xlim(30, 40)


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, training['Age'].max()))
facet.add_legend()
plt.xlim(40, 60)


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, training['Age'].max()))
facet.add_legend()
plt.xlim(60)


# In[ ]:


training.info()


# In[ ]:


testing.info()


# # Binning
# Binning/Converting Numerical Age to Categorical Variable
# 
# feature vector map:
# child: 0
# young: 1
# adult: 2
# mid-age: 3
# senior: 4

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# In[ ]:


training.head()


# In[ ]:


bar_chart('Age')


# # Embarked
#  filling missing values

# In[ ]:


Pclass1 = training[training['Pclass']==1]['Embarked'].value_counts()
Pclass2 = training[training['Pclass']==2]['Embarked'].value_counts()
Pclass3 = training[training['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# 
# more than 50% of 1st class are from S embark
# 
# more than 50% of 2nd class are from S embark
# 
# more than 50% of 3rd class are from S embark
# 
# 
# fill out missing embark with S embark

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:



training.head()


# In[ ]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# # Fare
# 

# In[ ]:



# fill missing Fare with median fare for each Pclass# fill m 
training["Fare"].fillna(training.groupby("Pclass")["Fare"].transform("median"), inplace=True)
testing["Fare"].fillna(testing.groupby("Pclass")["Fare"].transform("median"), inplace=True)
training.head(50)


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, training['Fare'].max()))
facet.add_legend()
 
plt.show()


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, training['Fare'].max()))
facet.add_legend()
plt.xlim(0, 20)


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, training['Fare'].max()))
facet.add_legend()
plt.xlim(0, 30)


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, training['Fare'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[ ]:


training.head()


# # Cabin

# In[ ]:


training.Cabin.value_counts()


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


Pclass1 = training[training['Pclass']==1]['Cabin'].value_counts()
Pclass2 = training[training['Pclass']==2]['Cabin'].value_counts()
Pclass3 = training[training['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


# fill missing Fare with median fare for each Pclass# fill m 
training["Cabin"].fillna(training.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
testing["Cabin"].fillna(testing.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# # FamilySize
# 

# In[ ]:


training["FamilySize"] = training["SibSp"] + training["Parch"] + 1
testing["FamilySize"] = testing["SibSp"] + testing["Parch"] + 1


# In[ ]:


facet = sns.FacetGrid(training, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, training['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[ ]:


training.head()


# In[ ]:


testing.head()


# In[ ]:


features_drop = ['Ticket', 'SibSp', 'Parch']
training = training.drop(features_drop, axis=1)
testing = testing.drop(features_drop, axis=1)
training = training.drop(['PassengerId'], axis=1)


# In[ ]:


train_data = training.drop('Survived', axis=1)
target = training['Survived']

train_data.shape, target.shape


# In[ ]:


train_data.head(10)


# # Modelling

# In[ ]:


# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np


# In[ ]:


training.info()


# # Cross Validation (K-fold)

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# # kNN

# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# kNN Score
round(np.mean(score)*100, 2)


# # Decision Tree

# In[ ]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# decision tree Score
round(np.mean(score)*100, 2)


# # Ramdom Forest

# In[ ]:


clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# Random Forest Score
round(np.mean(score)*100, 2)


# #  Naive Bayes

# In[ ]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# Naive Bayes Score
round(np.mean(score)*100, 2)


# # SVM

# In[ ]:


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# # Testing

# In[ ]:


clf = SVC()
clf.fit(train_data, target)

test_data = testing.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": testing["PassengerId"],
        "Survived": prediction
    })
os.chdir("/kaggle/working/")

submission.to_csv('/kaggle/working/submission.csv', index=False)


# In[ ]:


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.info()
submission.head(418)


# In[ ]:


print(os.listdir("../working/"))


# resources
# 
# https://www.kaggle.com/enespolat/kernels/notebooks/new?forkParentScriptVersionId=5148993
# https://chrisalbon.com/python/data_wrangling/pandas_dataframe_count_values/
