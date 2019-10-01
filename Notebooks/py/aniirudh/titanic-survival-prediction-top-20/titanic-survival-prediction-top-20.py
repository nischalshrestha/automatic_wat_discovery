#!/usr/bin/env python
# coding: utf-8

# # Import Python Libraries
# 
# Let's start by import libraries such as pandas, numpy and matplotlib (for visualization)

# In[3]:


# import pandas, and numpy
import pandas as pd
import numpy as np
from pandas import Series,DataFrame

# matplotlib for visualization
import matplotlib.pyplot as plot
import seaborn as sns

# ignore warnings for now
import warnings
warnings.filterwarnings('ignore')


# # Load the Data
# 
# After importing the required libraries, we load the training and test data using the `read_csv` function, and print some sample of the training data.

# In[4]:


# load the training set
train = pd.read_csv('../input/train.csv')

# load the test set
test = pd.read_csv('../input/test.csv')

# print some samples from the training set
train.sample(5)


# # Describe the Data
# 
# Using the `describe` function to check out all the columns, and values of the training data.

# In[5]:


# describe training data
train.describe(include='all')


# There are 12 columns with 891 values:
# 
# * PassengerId (int): Self explanatory
# 
# 
# * Survived (int): Did the passenger survive or not
# 
# 
# * PClass (int): Ticket Class
# 
# 
# * Name (string)
# 
# 
# * Sex (string)
# 
# 
# * Age (float): If you see the Age column in the table above, you can see that only 714 out of 891 values have been recorded. In the data preprocessing step, we will fill in the missing values.
# 
# 
# * SibSp(int): No. of siblings/spouses aboard
# 
# 
# * Parch(int): No. of parents/children aboard
# 
# 
# * Ticket(int)
# 
# 
# * Fare(float)
# 
# 
# * Cabin(string): Only 204 values have been recorded for cabin. There is a huge gap that we might need to fill (891 - 204), hence the best way would be that we drop this column as we move forward towards prediction.
# 
# 
# * Embarked(string): Boarding point, we will fill in the 2 missing values later in the preprocessing step.

# # Data Visualization
# 
# ### Embarked Feature:

# In[6]:


sns.barplot(x="Embarked", y="Survived", data=train)

print("Number of passengers who embarked at S: ", (train["Embarked"] == "S").value_counts(normalize = True)[1] *100)
print("Number of passengers who embarked at C: ", (train["Embarked"] == "C").value_counts(normalize = True)[1] *100)
print("Number of passengers who embarked at Q: ", (train["Embarked"] == "Q").value_counts(normalize = True)[1] *100)


# #####  Observations:
# 
# 'S' is the most occurred value.
# 
# Passengers who boarded at C had a higher change of survival, and those who boarded at S had the lowest (33%) survival rate.

# ### Pclass Feature:

# In[7]:


sns.barplot(x="Pclass", y="Survived", data=train)


# ##### Observation:
# 
# People with higher class (social status) had a higher survial rate (~62%)

# ### Sex Feature:

# In[8]:


sns.barplot(x="Sex", y="Survived", data=train)


# ##### Observation:
# Females had the higher survival rate as compared to male. This is one of most important feature in our training data

# ### Sibsp Feature:

# In[9]:


sns.barplot(x="SibSp", y="Survived", data=train)


# ##### Observation:
# 
# By looking at the graph, it is evident that people with 1 sibling/spouses had the higher survival rate as compared to people with 0 or more than 1 siblings/spouses on board.

# ### Parch Feature:

# In[10]:


sns.barplot(x="Parch", y="Survived", data=train)


# ##### Observation:
# 
# People who are travelling alone are less likely to survive than the people who are travelling with 1-3 parents/children. 

# ### Name/Title Feature:
# 
# Extracting out title from the passenger name, since Title (Mr, Mrs. etc) will give a substantial amount of information.

# In[11]:


train['Title'] = train.Name.str.extract('([A-Za-z]+)\.', expand=False)
test['Title'] = test.Name.str.extract('([A-Za-z]+)\.', expand=False)

sns.barplot(x="Title", y="Survived", data=train)


# # Preprocessing and Cleaning Data
# 

# ## 1. First, let's start by dropping columns which may not be useful in prediction.
# 
# Let's start by dropping columns which may not be useful in prediction. For now, we are dropping Ticket and PassengerId column (Ticket does not seem that helpful to me, but if possible we can extract out some piece of information, which might help in improving accuracy)

# In[12]:


train = train.drop(['PassengerId', 'Ticket'], axis=1)
test = test.drop('Ticket', axis=1)

test.describe(include='all')


# ## 2. Fill in missing values
# 
# There are some columns which have missing value. For eg Embarked. In the training set, out of 891 rows, only 889 have values filled in for Embarked feature. Similarly for age as well.
# 
# ### Embarked: 
# 
# (Only for train set, since test set has no missing values for Embarked)
# 
# There are 2 missing values for the `Embarked` column. In the data visualization step, we observed that `S` was the most occured value for this feature, hence we will fill the 2 missing values with `S`

# In[13]:


# fill NaN values with S, since S is the most occured value
train["Embarked"] = train["Embarked"].fillna("S")
train.describe(include='all')


# ### Fare:
# 
# (Only required for test set, training set has no missing values for Fare)

# In[14]:


# fill in the missing fare values in test set
# use the median value to fill up missing rows
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test.describe(include='all')


# ### Age:
# 
# (Required for both training and test set)
# 
# Reference: (https://www.kaggle.com/omarelgabry/a-journey-through-titanic?scriptVersionId=447794)

# In[15]:


# get average, std and missing values for age in training set
training_age_avg = train["Age"].mean()
training_age_std = train["Age"].std()
training_age_missing = train["Age"].isnull().sum()

print("Avg Training Age:", training_age_avg, "Std Training Age:", training_age_std, 
      "Missing Age:",training_age_missing)

# get average, std and missing values for age in test set
test_age_avg = test["Age"].mean()
test_age_std = test["Age"].std()
test_age_missing = test["Age"].isnull().sum()

print("Avg Test Age:", test_age_avg, "Std Test Age:", test_age_std, "Missing Age:",test_age_missing)

# generate random number between (mean - std) & (mean + std)
random_1 = np.random.randint(training_age_avg - training_age_std, training_age_avg + training_age_std,
                            size = training_age_missing)

random_2 = np.random.randint(test_age_avg - test_age_std, test_age_avg + test_age_std, size = test_age_missing)

train['Age'].dropna()
train["Age"][np.isnan(train["Age"])] = random_1

test['Age'].dropna()
test["Age"][np.isnan(test["Age"])] = random_2


# ## 3. Cleaning Data
# 
# ### Sex Feature: 
# 
# Map male and female to integers

# In[16]:


combine = [train, test]

# integer mapping for male and female vars
sex_map = {"male": 0, "female": 1}

# map male and female to integers
for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map(sex_map)

# extract out male and female as separate features in the dataset
for dataset in combine:
    dataset['Male'] = dataset['Sex'].map(lambda s: 1 if s == 0 else 0)
    dataset['Female'] = dataset['Sex'].map(lambda s: 1 if  s == 1  else 0)

# remove Sex feature, as we already have male and female feature
train = train.drop('Sex', axis=1)
test = test.drop('Sex', axis=1)

train.head()


# ## Embarked Feature:
# 
# Map S, C and Q to integer values
# 

# In[17]:


combine = [train, test]

# integer mapping for embarked feature
embark_map ={"S": 0, "C": 1, "Q": 2}

# map S, C and Q to integers
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].map(embark_map)

# extract out S, C and Q as separate features in the dataset
for dataset in combine:
    dataset['Embarked_S'] = dataset['Embarked'].map(lambda s: 1 if s == 0 else 0)
    dataset['Embarked_C'] = dataset['Embarked'].map(lambda s: 1 if  s == 1  else 0)
    dataset['Embarked_Q'] = dataset['Embarked'].map(lambda s: 1 if  s == 2  else 0)

# remove Embarked feature, as we already have S, C and Q feature
train = train.drop('Embarked', axis=1)
test = test.drop('Embarked', axis=1)

train.head()


# ## Title Feature:
# 
# Map title values to integer

# In[18]:


combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

# extract out different titles as features
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title_Mr'] = dataset['Title'].map(lambda s: 1 if s == 1 else 0)
    dataset['Title_Miss'] = dataset['Title'].map(lambda s: 1 if  s == 2  else 0)
    dataset['Title_Mrs'] = dataset['Title'].map(lambda s: 1 if  s == 3  else 0)
    dataset['Title_Master'] = dataset['Title'].map(lambda s: 1 if  s == 4  else 0)
    dataset['Title_Royal'] = dataset['Title'].map(lambda s: 1 if  s == 5  else 0)
    dataset['Title_Rare'] = dataset['Title'].map(lambda s: 1 if  s == 6  else 0)

# remove Title feature, as we already have different titles as feature
train = train.drop(['Title', 'Name'], axis=1)
test = test.drop(['Title', 'Name'], axis=1)

train.head()


# ### Family Feature:
# 
# Create a new feature based on the family size

# In[19]:


combine = [train, test]

# create a Fsize feature
for dataset in combine:
    dataset["Fsize"] = dataset["SibSp"] + train["Parch"] + 1

# extract out Fsize into 4 different features
for dataset in combine:
    dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
    dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

# remove Fsize, SibSp and Parch feature, as we already have features on family size
train = train.drop(['Fsize', 'SibSp', 'Parch'], axis=1)
test = test.drop(['Fsize', 'SibSp', 'Parch'], axis=1)

train.head()


# ### PClass Feature:
# 
# Extract out PClass into separate features

# In[20]:


combine = [train, test]

# extract out Pclass into 3 different features
for dataset in combine:
    dataset['Upper_Class'] = dataset['Pclass'].map(lambda s: 1 if s == 1 else 0)
    dataset['Middle_Class'] = dataset['Pclass'].map(lambda s: 1 if  s == 2  else 0)
    dataset['Lower_Class'] = dataset['Pclass'].map(lambda s: 1 if s == 3 else 0)

# remove Pclass, as we already have features on classes
train = train.drop('Pclass', axis=1)
test = test.drop('Pclass', axis=1)

train.head()


# ### Fare Feature:

# In[21]:


combine = [train, test]

for dataset in combine:
    dataset['FareBand'] = pd.qcut(dataset['Fare'], 4, labels = [1, 2, 3, 4])

train = train.drop('Fare', axis=1)
test = test.drop('Fare', axis=1)

train.head()


# ### Cabin Feature:

# In[22]:


combine = [train, test]

# fill missing cabin values with U (undefined)
for dataset in combine:
    dataset["Cabin"] = dataset["Cabin"].fillna('U')

# integer mapping for cabins
cabin_map = {"U": 0, "C": 1, "E": 2, "G": 3, "D": 4, "A": 5, "B": 6, "F": 7, "T": 8}

# map integers to cabin
for dataset in combine:
    dataset["Cabin"] = dataset["Cabin"].map(cabin_map)

for dataset in combine:
    dataset['Cabin_U'] = dataset['Cabin'].map(lambda s: 1 if s == 0 else 0)
    dataset['Cabin_C'] = dataset['Cabin'].map(lambda s: 1 if  s == 1  else 0)
    dataset['Cabin_E'] = dataset['Cabin'].map(lambda s: 1 if s == 2 else 0)
    dataset['Cabin_G'] = dataset['Cabin'].map(lambda s: 1 if s == 3 else 0)
    dataset['Cabin_D'] = dataset['Cabin'].map(lambda s: 1 if s == 4 else 0)
    dataset['Cabin_A'] = dataset['Cabin'].map(lambda s: 1 if s == 5 else 0)
    dataset['Cabin_B'] = dataset['Cabin'].map(lambda s: 1 if s == 6 else 0)
    dataset['Cabin_F'] = dataset['Cabin'].map(lambda s: 1 if s == 7 else 0)
    dataset['Cabin_T'] = dataset['Cabin'].map(lambda s: 1 if s == 8 else 0)

train = train.drop(['Cabin'], axis=1)
test = test.drop('Cabin', axis=1)


# # Choosing Model
# 
# Let's run the training set using different types of model:
# 
# * Logistic Regression
# 
# 
# * Support Vector Machine
# 
# 
# * Random Forest Classifier
# 
# 
# * Decision Tree Classifier
# 
# 
# * Gradient Boost Classifier

# In[23]:


X_train = train
Y_train = train["Survived"]
X_test = test.copy()

X_test.head()


# ## Logistic Regression:

# In[24]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
lr.score(X_train, Y_train)


# ## Support Vector Machine:

# In[30]:


from sklearn import svm

svm = svm.SVC()
svm.fit(X_train, Y_train)
Y_pred = svm.predict(X_test)
svm.score(X_train, Y_train)


# ## Random Forest:

# In[39]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)


# ## Decision Tree:
# 

# In[43]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

kfold = StratifiedKFold(n_splits=10)

DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_

gsadaDTC.best_score_

ids = test["PassengerId"]

predictions = gsadaDTC.predict(X_test)

ids = test["PassengerId"]
int_id = []
for i in ids:
    int_id.append(int(i))
    
int_pred = []
for y in predictions:
    int_pred.append(int(y))

output = pd.DataFrame({"PassengerId": int_id, "Survived": int_pred})
output.to_csv("submission.csv", index=False)

gsadaDTC.best_score_


# ## Gradient Boost:

# In[27]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train, Y_train)
Y_pred = gbc.predict(X_test)
gbc.score(X_train, Y_train)


# ### Among all the models used, Random Forest Classifier gave the highest accuracy of 0.986.

# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
kfold = StratifiedKFold(n_splits=10)

RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [n for n in range(9, 14)],
              "max_features": [1, 3, 10],
              "min_samples_split": [n for n in range(4, 11)],
              "min_samples_leaf": [n for n in range(2, 5)],
              "bootstrap": [False],
              "n_estimators" :[n for n in range(10, 60, 10)],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_

# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
predictions = gsRFC.predict(X_test)
gsRFC.score(X_train, Y_train)

ids = test["PassengerId"]
int_id = []
for i in ids:
    int_id.append(int(i))
    
int_pred = []
for y in predictions:
    int_pred.append(int(y))

output = pd.DataFrame({"PassengerId": int_id, "Survived": int_pred})
output.to_csv("submission.csv", index=False)


gsRFC.best_score_

# print ("Starting 1")
# forrest_params = dict(     
#     max_depth = [n for n in range(9, 14)],     
#     min_samples_split = [n for n in range(4, 11)], 
#     min_samples_leaf = [n for n in range(2, 5)],     
#     n_estimators = [n for n in range(10, 60, 10)],
# )
# print ("Starting 2")

# forrest = RandomForestClassifier()
# print ("Starting 3")

# forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5) 
# print ("Starting 4")

# forest_cv.fit(X_train, Y_train)
# print ("Starting 5")

# print("Best score: {}".format(forest_cv.best_score_))
# print("Optimal params: {}".format(forest_cv.best_estimator_))

# # random forrest prediction on test set
# predictions = forest_cv.predict(X_test)
# print ("Starting 6")

# output = pd.DataFrame({"PassengerId": ids, "Survived": predictions})
# output.to_csv("submission.csv", index=False)

# print ("Starting 7")


# print("Best score: {}".format(forest_cv.best_score_))
# print("Optimal params: {}".format(forest_cv.best_estimator_))

