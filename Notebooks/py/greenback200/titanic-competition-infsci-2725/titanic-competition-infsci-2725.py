#!/usr/bin/env python
# coding: utf-8

# **Assignment 4**
# 
# *By John Falconi, Shreya Shenoy, and Neha Aeri*
#  
#  
# Contents
# 1. Loading libraries
# 2. Analyzing/Visualizing Data
# 3. Cleaning Data & Feature Engineering
# 4. Running Models
# 5. Submission
# 
# 

# **1. Loading Libraries**
# 
# For this section, we simply load our libraries and data into the notebook.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# **2. Analyzing/Visualizing Data**
# 
#   The goal of this section was to study the provided data to see if there were any strong trends or correlation between survival and another column.  Since we found that there were many different columns, we initially wanted our approach to be dropping extraneous noise and categorizing/grouping fields with a very wige range of values to hone in on strong trends.
# 
#   We predicted that a few key traits would be very important to include in our prediction model.  Namely:
#   
# * Females would be more likely to survive than males
# * The higher the social class, the more likely the passenger would survive
# * Those with more family (parents, children, siblings, spouses) would have a higher rate of survival
# * The older a person is, the less likely they will survive
# * The younger a person is, the more likely they will survive
# 
#   Not only do we set out to identify patterns within the data, but our goal is also to identify any flaws or traits the data has that will adversely affect the data and will subsequently require cleaning.  The way we approached analyzing data and identifying patterns was to look at relationships between columns from multiple perspectives.  For instance, perhaps you are more likely to survive as a higher class passenger than a lower class.  However, it is beneficial to look at total survival and demise sums to get an even better picture as to how strong the correlation is.  Lower class deaths dwarfed lower class survivals by an overwhelming amount, which would indicate that it is a *very* strong factor when predicting survival rates.

# In[ ]:


#Let's look at the two datasets to see what types of data are in each column and how many non-null values
# make up each column.

print(train.info())
print(test.info())


# In[ ]:


#It would also be helpful to take a look at a sampling of the data to see the actual values.

train.sample(5)


# In[ ]:


#"Describing" this data could give us a good idea of the general breakdown of each category.  I'm looking
# for total amounts and averages.  We have noted that the "count" in Age is lower than the others, which
# means we should try to impute data if we want to improve the prediction models.

train.describe()


# In[ ]:


#This tells us how many of each gender survive.  We can clearly see that it is far more likely for a 
# woman to survive than a man.

_ = sns.barplot(x="Sex", y="Survived", data=train)


# In[ ]:


#Looked at in another way, we can view the numbers behind both the survival and the demise of both
# genders.  The amount of men who did not survive relative to the total amount of men is very high.

_ = sns.factorplot("Survived", data=train, hue="Sex", kind="count")


# In[ ]:


#We can look at the passenger's class in a similar way.  Clearly, the upper class (represented with the number 1)
# was far more likely to survive, most likely due to their status.  

_ = sns.barplot(x="Pclass", y="Survived", data=train)


# In[ ]:


#However, seen in a different perspective, we can note that the amount of people deemed high class is much
# less than the lowest class.  Nevertheless, this would be a good feature to include due to the distinct
# trend.

_ = sns.factorplot("Survived", data=train, hue="Pclass", kind="count")


# In[ ]:


#We also wanted to look at how relatives would affect the survival rate.  We assumed that the more family
# a person had on the ship, the higher their chances would be of survival.  Our thinking came from the fact
# that a person would be less likely to be left behind if they had a family keeping them all together, as
# opposed to a single person with no relatives to look after them.

#First we took a look at survival rates of those with siblings and spouses.  This surprised us in that
# it was more favorable to have 2 or less siblings or spouses to survive.

_ = sns.barplot(x="SibSp", y="Survived", data=train)


# In[ ]:


# Conversely, we also took a look at parents and children and found that there was more of an even distribution
# and that survival favored those with 2 or more parents and children.  

_ = sns.barplot(x="Parch", y="Survived", data=train)


# In[ ]:


#Finally we wanted to take a look at age.  We believe that age would play a huge factor in making accurate
# predictions.  We believe that those who are more elderly would not survive at a very high rate due to 
# deteriorating health.  We also think that those who are young will be more likely to survive since they
# will be looked after more.

#Unfortunately for us, most of the passengers fall in the young-to-mid adult range.  Therefore, the 
# effects of the children and senior citizens will be minimal.  Also, we are missing Age values, so we will
# impute data and take another look at this graph again later.

fig = sns.FacetGrid(train, hue="Sex", aspect=5)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = train['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# **3. Cleaning Data & Feature Engineering**
# 
#   This portion is where things get interesting.  After analyzing the data, we have narrowed down the fields that we would like to hone in on.  These fields include:
# 
# * Pclass    
# * Sex           
# * SibSp          
# * Parch          
# * Embarked              
# * Age
# 
#   All of these present interesting patterns relative to the Survived data.  They have strong correlations that could be used for fairly accurate prediction models.  We believe Pclass is a good pick because of the strong correlation between class and survival, Sex because there are strong trends in females having high survival rates and males with low survival rates, SibSp and Parch because they add an interesting family dynamic where having around ~2 relatives seem to increase your chances of survival, Embarked because of its interesting and somewhat inexplicable relationship in survival rates, and Age because there is a broad distribution from child to senior citizen.
#   
#   We will look to categorize and group the values in a few of these columns.  For instance, instead of having a wide range of ages, we could set age categories that indicate if you are a child, an adult, or even a senior.  Similarly, we will look into names since they have titles to their names.  Titles can be used to predict behavior.  For instance, if the passenger has a military title, he may be more selfless and make sure civilians are safe before thinking of his or herself.  This might mean the military titles are less likely to survive.  Conversely, if the title indicates royalty, one would think they would be saved by any means possible due to their status, giving them a higher survival rate.
# 
#   In cleaning the data, our first objective is to eliminate all null values.  We will impute the most statistically likely data in its place so as to maintain the integrity of the data.  Not only that, but many categorical columns will be re-formatted to numeric variables so that they can be used in statistics to make great predictions.  And finally, since we find some columns to be useless and do not hold any significant patterns or correlation, we will drop out everything irrelevant to cleanse the dataset so it can be run in different models with sound accuracy.

# In[ ]:


#To clean this data, we are going to want to start off by filling in all the null values for a more complete dataset.
# We decided that the Cabin column provided no useful information and had far too many missing values
# to really accurately impute realistic data.  Therefore, we found it to be noise and dropped it from the
# train and test data entirely
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


# In[ ]:


#From here we wanted to complete the other two columns Age and Embarked.  Embarked was easier to fill in
# as it was only missing two values.  We first took a look at the count for each embarked location

_ = sns.factorplot("Embarked", data=train, kind="count")


# In[ ]:


#Since an overwhelming majority of passengers came from location "S", we decided to just fill the last
# two values with "S" since it was most likely they embarked from "S" from a statistical standpoint.

train = train.fillna({"Embarked": "S"})


# In[ ]:


#Next was to impute the ages.  Since I was not sure how to impute data that had so many different values
# I decided to use the approach found in the following kernel:

# https://www.kaggle.com/startupsci/titanic-data-science-solutions

#First I needed to combine everything to work with the dataset as a whole.
combine = [train, test]


# In[ ]:


#The approach made use of the titles in the name to group them all.  The idea is that similar titles can
# help determine a rough estimate of the missing values' age.

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


#To make the grouping easier, the approach replaced uncommon titles with general titles that would be easier
# to group.  The original approach placed 'Master' in its own category, however I found that it would be
# closer to the 'Royal' title category, so I grouped it to reduce noise.
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Officer')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir', 'Master'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#Now we can see the new groups and their respective survival rates.  This will be useful not only for
# imputing age, but also as an added feature in the prediction models.
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Royal": 4, "Officer": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()


# In[ ]:


#Not only will title be useful, but categorizing the ages would reduce noise and make it easier to make 
# predictions based off of only a few categories rather than a wide range of ages.
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)


# In[ ]:


#This code just maps the age groups and assigns them accordingly in the new column AgeGroup
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() 
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() 
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() 
royal_age = train[train["Title"] == 4]["AgeGroup"].mode() 
officer_age = train[train["Title"] == 5]["AgeGroup"].mode() 

age_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_mapping[train["Title"][x]]
        
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_mapping[test["Title"][x]]


# In[ ]:


age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()


# In[ ]:


#Drop Age, Name, Ticket, and Fare features since they either contain no more useful information
# or they have been reformatted into other columns to fit into the models.
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)

train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)

train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[ ]:


#Sex and Embarked need to be mapped to show numeric values since they were previously categorical
# data and would not be able to be used in a model.  One final view of the first five rows
# should be enough to tell us if we have all the right values and data types to run.
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()


# In[ ]:


#Test data should be good with no nulls and all have the appropriate data types.
print(test.info())


# **4. Running Models**
# 
# *List of models we will be running:*
# * Support Vector Machines	
# * Gradient Boosting Classifier	
# * Random Forest	
# * Logistic Regression	
# * Linear SVC	
# * Decision Tree	
# * Naive Bayes	
# * Stochastic Gradient Descent	
# * KNN	
# * Perceptron
# ___________________________________________________________________________________________________________________________
# 
# *Data included in our dataset:*
# * PassengerId    
# * Pclass         
# * Sex           
# * SibSp          
# * Parch         
# * Embarked       
# * Title          
# * AgeGroup
# 
# We picked these features because they all had significant correlation with the Survived data which, to us, proved to be excellent factors in making accurate predictions.  All the data was scrubbed and formatted to fit into the models, so all that is left to be done is run them all and see the results.
# 

# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[ ]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[ ]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[ ]:


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[ ]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[ ]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# **5. Submission**
# 
# As shown above with our results, all but three models passed the given 0.75598 score on courseweb.  We decided to submit our highest score, Support Vector Machines with a score of .8426, in the file 'titanic.csv'.  See below for a sample of the file's first five rows.

# In[ ]:


submission = pd.DataFrame({
    "PassengerId" : test['PassengerId'],
    "Survived" : svc.predict(test.drop('PassengerId', axis=1))
})


# In[ ]:


submission.to_csv('titanic.csv', index=False)
submission.head()


# **SettingWithCopyWarning**
# SettingWithCopyWarning is a warning generated from the panda library.  It comes about when it detects a chained assignment.  To avoid the warning, depending on the scenario one could combine chained operatings into one with "loc" so that the original dataframe is set, or to explicitly create a new copy when creating a new dataframe.
