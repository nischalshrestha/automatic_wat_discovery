#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# ### Predict survival on the Titanic
# - Defining the problem statement
# - Collecting the data
# - Exploratory data analysis
# - Feature engineering
# - Modelling
# - Testing

# ## 1. Defining the problem statement
# Complete the analysis of what sorts of people were likely to survive.  
# In particular, we ask you to apply the tools of machine learning to predict which passengers survived the Titanic tragedy.

# In[ ]:


from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")


# ## 2. Collecting the data
# 
# training data set and testing data set are given by Kaggle
# you can download from  
# my github [https://github.com/minsuk-heo/kaggle-titanic/tree/master](https://github.com/minsuk-heo/kaggle-titanic)  
# or you can download from kaggle directly [kaggle](https://www.kaggle.com/c/titanic/data)  
# 
# ### load train, test dataset using Pandas

# In[ ]:


import pandas as pd

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train.head()


# ## 3. Exploratory data analysis

# ## Data Dictionary
# - Survived : 0 = No, 1 = Yes
# - Pclass : Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# - Sibsp : # of siblings / spouses aboard the Titanic
# - Parch : # of parents / children aboard the Titanic
# - Ticket : Ticket number
# - Cabin : Cabin number
# - Embarked : Port of Embarkation C = CherBourg, Q = Queenstown, S = Southampton

# ### Total rows and columns
# We can see that three are 891 rows and 12 columns in our training dataset.

# In[ ]:


df_test.head()


# In[ ]:


df_train.shape 


# In[ ]:


df_test.shape


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# There are 177 rows with missing Age, 687 rows with missing Cabin and 2 rows with missing Embarked information

# ## Import python lib for visualization

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# ## Bar Chart for Categorical Features
# - Pclass
# - Sex
# - SibSp ( # of siblings and spouse)
# - Parch (  # of parents and children)
# - Embarked
# - Cabin

# In[ ]:


def bar_chart (feature) :
    survived = df_train[df_train['Survived'] == 1][feature].value_counts()
    dead = df_train[df_train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10, 5))


# In[ ]:


bar_chart('Sex')


# The Chart confirms **Women** more likely survived thatn **Men**.[](http://)

# In[ ]:


bar_chart('Pclass')


# The Chart confirms **1st class** more likely survived than **other classes**.  
# The Chart confirms **3rd class** more likely dead than other **classes**.
# 

# In[ ]:


bar_chart('SibSp')


# The Chart confirms **a person aboarded with more than 2 siblings or spouse** more likely survived.  
# The Chart confirms **a person aboarded without siblings or spouse** more likely dead.

# In[ ]:


bar_chart('Parch')


# The Chart confirms **a person aboarded with more than 2 parents or children** more likely survived.  
# The Chart confirms **a person aboarded alon** more likely dead.

# In[ ]:


bar_chart('Embarked')


# The Chart confirms **a person aboarded from C** slightly more likely survived  
# The Chart confirms **a person aboarded from Q** more likely dead  
# The Chart confirms **a person aboarded from S** more likely dead

# ## 4. Feature engineering
# 
# Feature engineering is the process of using domain knowledge of the data to create features (**features vectors**) that make machine learning algorithms work.
# 
# Feature vector is an n-dimesional vector of numerical features that represent some object.  
# Many algorithms in machine learning require a numerical representation of objects, since such representations facilitate processing and statistical analysis.

# In[ ]:


df_train.head()


# ### 4.1 How Titanic Sank?
# Sank from the bow of the ship where third class rooms located conclusion, Pclass is key feature for classifier.

# In[ ]:


Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")


# In[ ]:


df_train.head(10)


# ### 4.2 Name

# In[ ]:


train_test_data = [df_train, df_test] # combining train and test dataset

for dataset in train_test_data :
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand = False)


# In[ ]:


df_train['Title'].value_counts()


# In[ ]:


df_test['Title'].value_counts()


# #### Title map
# Mr : 0  
# Miss : 1  
# Mrs: 2  
# Others: 3

# In[ ]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


bar_chart('Title')


# In[ ]:


# delete unnecessary feature from dataset
df_train.drop('Name', axis = 1, inplace = True)
df_test.drop('Name', axis = 1, inplace = True)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ### 4.3 Sex
# 
# male: 0
# female: 1

# In[ ]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data :
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


bar_chart('Sex')


# ### 4.4 Age

# #### 4.4.1 Some age is missing
# Let's use Title's median age for missing Age

# In[ ]:


df_train.head(100)


# In[ ]:


# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
df_train["Age"].fillna(df_train.groupby("Title")["Age"].transform("median"), inplace =True)
df_test["Age"].fillna(df_test.groupby("Title")["Age"].transform("median"), inplace = True)


# In[ ]:


df_train.head(30)
df_train.groupby("Title")["Age"].transform("median")


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived",aspect = 4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df_train['Age'].max()))
facet.add_legend()
 
plt.show() 


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived",aspect = 4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df_train['Age'].max()))
facet.add_legend()
 
plt.xlim(0, 20) 


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived",aspect = 4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df_train['Age'].max()))
facet.add_legend()
 
plt.xlim(20, 30) 


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived",aspect = 4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df_train['Age'].max()))
facet.add_legend()
 
plt.xlim(30, 40) 


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived",aspect = 4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df_train['Age'].max()))
facet.add_legend()
 
plt.xlim(40, 60) 


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived",aspect = 4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df_train['Age'].max()))
facet.add_legend()
 
plt.xlim(60) 


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# #### 4.4.2 Binning
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


df_train.head()


# In[ ]:


bar_chart('Age')


# ### 4.5 Embarked

# #### 4.5.1 filling missing values

# In[ ]:


Pclass1 = df_train[df_train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = df_train[df_train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = df_train[df_train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# more than 50% of 1st class are from S embark  
# more than 50% of 2nd class are from S embark  
# more than 50% of 3rd class are from S embark
# 
# **fill out missing embark with S embark**

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


df_train.head()


# In[ ]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# ### 4.6 Fare

# In[ ]:


# fill missing Fare with median fare for each Pclass
df_train["Fare"].fillna(df_train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
df_test["Fare"].fillna(df_test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
df_train.head(50)


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived", aspect=4)
facet.map(sns.kdeplot,'Fare',shade = True)
facet.set(xlim = (0, df_train['Fare'].max()))
facet.add_legend()
 
plt.show()  


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived", aspect=4)
facet.map(sns.kdeplot,'Fare',shade = True)
facet.set(xlim = (0, df_train['Fare'].max()))
facet.add_legend()
 
plt.xlim(0, 20)


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived", aspect=4)
facet.map(sns.kdeplot,'Fare',shade = True)
facet.set(xlim = (0, df_train['Fare'].max()))
facet.add_legend()
 
plt.xlim(0, 30)


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived", aspect=4)
facet.map(sns.kdeplot,'Fare',shade = True)
facet.set(xlim = (0, df_train['Fare'].max()))
facet.add_legend()
 
plt.xlim(0)


# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[ ]:


df_train.head()


# ### 4.7 Cabin

# In[ ]:


df_train.Cabin.value_counts()


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


Pclass1 = df_train[df_train['Pclass'] == 1]['Cabin'].value_counts()
Pclass2 = df_train[df_train['Pclass'] == 2]['Cabin'].value_counts()
Pclass3 = df_train[df_train['Pclass'] == 3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked = True, figsize = (10,5))


# In[ ]:


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


# fill missing Fare with median fare for each Pclass
df_train["Cabin"].fillna(df_train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
df_test["Cabin"].fillna(df_test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# ### 4.8 FamilySize

# In[ ]:


df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] + 1
df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] + 1


# In[ ]:


facet = sns.FacetGrid(df_train, hue = "Survived",aspect = 4)
facet.map(sns.kdeplot,'FamilySize',shade = True)
facet.set(xlim=(0, df_train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[ ]:


df_train.head()


# In[ ]:


features_drop = ['Ticket', 'SibSp', 'Parch']
df_train = df_train.drop(features_drop, axis=1)
df_test = df_test.drop(features_drop, axis=1)
df_train = df_train.drop(['PassengerId'], axis=1)


# In[ ]:


train_data = df_train.drop('Survived', axis=1)
target = df_train['Survived']

train_data.shape, target.shape


# In[ ]:


train_data.head(10)


# ## 5. Modelling

# In[ ]:


# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np


# In[ ]:


df_train.info()


# ### 6.2 Cross Validation (K-fold)

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)


# ### 6.2.1 kNN

# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_fold, n_jobs = 1, scoring = scoring)
print(score)


# In[ ]:


# kNN Score
round(np.mean(score)*100, 2)


# ### 6.2.2 Decision Tree

# In[ ]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_fold, n_jobs = 1, scoring = scoring)
print(score)


# In[ ]:


# decision tree Score
round(np.mean(score)*100, 2)


# ### 6.2.3 Ramdom Forest

# In[ ]:


clf = RandomForestClassifier(n_estimators = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_fold, n_jobs = 1, scoring = scoring)
print(score)


# In[ ]:


# Random Forest Score
round(np.mean(score)*100, 2)


# ### 6.2.4 Naive Bayes

# In[ ]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_fold, n_jobs = 1, scoring = scoring)
print(score)


# In[ ]:


# Naive Bayes Score
round(np.mean(score)*100, 2)


# ### 6.2.5 SVM

# In[ ]:


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_fold, n_jobs = 1, scoring = scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# ## 7. Testing

# In[ ]:


clf = SVC()
clf.fit(train_data, target)

test_data = df_test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)


# In[ ]:


submission = pd.read_csv('submission.csv')
submission.head()


# ## References
# 
# This notebook is created by learning from the following notebooks:
# 
# - [Mukesh ChapagainTitanic Solution: A Beginner's Guide](https://www.kaggle.com/chapagain/titanic-solution-a-beginner-s-guide?scriptVersionId=1473689)
# - [How to score 0.8134 in Titanic Kaggle Challenge](http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html)
# - [Titanic: factors to survive](https://olegleyz.github.io/titanic_factors.html)
# - [Titanic Survivors Dataset and Data Wrangling](http://www.codeastar.com/data-wrangling/)
# - [Minsuk Heo](https://www.youtube.com/channel/UCxP77kNgVfiiG6CXZ5WMuAQ)

# In[ ]:




