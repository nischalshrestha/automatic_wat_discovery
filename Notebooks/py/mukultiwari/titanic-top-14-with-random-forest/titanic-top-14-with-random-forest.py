#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction
# 
# This is my first kernel which is inspired by
# 
# Jerry Tseng
# https://www.kaggle.com/jerrytseng/titanic-random-forest-82-78
# Sina
# https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
# 
# This Kernel covers - 
# 
# Data Analysis,
# Data Visualization,
# Feature Engineering,
# Model Comparision

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
sns.set_palette('cool')


# # Importing and Studying the Dataset

# ---

# In[ ]:


training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


# Viewing the data


# In[ ]:


training_data.head()


# In[ ]:


test_data.head()


# In[ ]:


# Information on the dataset


# In[ ]:


training_data.info()


# In[ ]:


test_data.info()


# In[ ]:


# We have 7 Numeric (PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare)
# and 5 categorical Features (Name, Sex, Ticket, Cabin, Embarked)


# In[ ]:


training_data.isnull().sum()


# In[ ]:


# Training Data contains 866 null values ( Age-177, Cabin-687, Emarked-2)


# In[ ]:


test_data.isnull().sum()


# In[ ]:


# Test Data contains 414 null values ( Age-86, Fare-1, Cabin-327)


# ---

# # Exploration and Feature Engineering of the dataset

# ---

# # Passenger Id

# In[ ]:


training_data.PassengerId.nunique()

# Saving the passengerId of test data for later use.
passengerId = test_data['PassengerId']


# In[ ]:


# Since passengerId does not have significant contribution to survival directly therefore we will Drop it.


# In[ ]:


training_data.drop(labels='PassengerId', axis=1, inplace=True)
test_data.drop(labels='PassengerId', axis=1, inplace=True)


# In[ ]:


training_data.head()


# # Pclass

# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].set_title("Pclass vs Frequency")
axes[1].set_title("Pclass vise Survival rate")
fig1_pclass = sns.countplot(data=training_data, x='Pclass', ax=axes[0])
fig2_pclass = sns.barplot(data=training_data, x='Pclass',y='Survived', ax=axes[1])


# In[ ]:


print(training_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


# In[ ]:


# As seen the population of passengers as per Pclass is ( 3 > 1 > 2) 
# Survival percentage as per classes is ( 1 > 2 > 3)
# Inference: 1st class passengers have higher survival rate 


# # Name

# In[ ]:


# Making a new feature Title having only the title extracted from the first name
# Making a new feature nameLength telling the length of the name


# In[ ]:


training_data.Name.nunique()


# In[ ]:


# Title Feature

training_data['Title'] = training_data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
test_data['Title'] = test_data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

# Name Leangth
training_data['Name_Len'] = training_data['Name'].apply(lambda x: len(x))
test_data['Name_Len'] = test_data['Name'].apply(lambda x: len(x))

# Dropping the name feature 

training_data.drop(labels='Name', axis=1, inplace=True)
test_data.drop(labels='Name', axis=1, inplace=True)


# In[ ]:


# Categorizing the name length by simply dividing it with 10.

test_data.Name_Len = (test_data.Name_Len/10).astype(np.int64)+1
training_data.Name_Len = (training_data.Name_Len/10).astype(np.int64)+1


# In[ ]:


print (training_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# In[ ]:


print (training_data[['Name_Len', 'Survived']].groupby(['Name_Len'], as_index=False).mean())


# In[ ]:


fx, axes = plt.subplots(2, 1, figsize=(15, 10))
axes[0].set_title("Title vs Frequency")
axes[1].set_title("Title vise Survival rate")
fig1_title = sns.countplot(data=training_data, x='Title', ax=axes[0])
fig2_title = sns.barplot(data=training_data, x='Title',y='Survived', ax=axes[1])


# In[ ]:


# Observations
# Name length as seen the longer the name the higher is the survival.
# Titles like Mrs. Ms. the lady or any royalty have high survival rates.


# # Gender

# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].set_title("Gender vs Frequency")
axes[1].set_title("Gender vise Survival rate")
fig1_gen = sns.countplot(data=training_data, x='Sex', ax=axes[0])
fig2_gen = sns.barplot(data=training_data, x='Sex', y='Survived', ax=axes[1])


# In[ ]:


print(training_data[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean())


# In[ ]:


# As can be seen that (number of males > number of females) but Survival ratio is inverse
# More females survived as compared to males


# # Age

# In[ ]:


training_data.Age.isnull().sum()

# Creating a list of age values without null values
training_age_n = training_data.Age.dropna(axis=0)


# In[ ]:


# Age contains 177 null values in training set and 86 in test set


# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].set_title("Age vs frequency")
axes[1].set_title("Age vise Survival rate")
fig1_age = sns.distplot(a=training_age_n, bins=15, ax=axes[0], hist_kws={'rwidth':0.7})

# Creating a new list of survived and dead

pass_survived_age = training_data[training_data.Survived == 1].Age
pass_dead_age = training_data[training_data.Survived == 0].Age

axes[1].hist([training_data.Age, pass_survived_age, pass_dead_age], bins=5, range=(0, 100), label=['Total', 'Survived', 'Dead'])
axes[1].legend()
plt.show()


# In[ ]:


# Taking care of null values in Age 
full_data = pd.concat([training_data, test_data])


# In[ ]:


# Null Ages in Training set (177 null values)
train_age_mean = full_data.Age.mean()
train_age_std = full_data.Age.std()
train_age_null = training_data.Age.isnull().sum()
rand_tr_age = np.random.randint(train_age_mean - train_age_std, train_age_mean + train_age_std, size=train_age_null)
training_data['Age'][np.isnan(training_data['Age'])] = rand_tr_age
training_data['Age'] = training_data['Age'].astype(int) + 1

# Null Ages in Test set (86 null values)
test_age_mean = full_data.Age.mean()
test_age_std = full_data.Age.std()
test_age_null = test_data.Age.isnull().sum()
rand_ts_age = np.random.randint(test_age_mean - test_age_std, test_age_mean + test_age_std, size=test_age_null)
test_data['Age'][np.isnan(test_data['Age'])] = rand_ts_age
test_data['Age'] = test_data['Age'].astype(int)

training_data.Age = (training_data.Age/15).astype(np.int64)
test_data.Age = (test_data.Age/15).astype(np.int64) + 1


# In[ ]:


print(training_data[['Age', 'Survived']].groupby(['Age'], as_index = False).mean())


# In[ ]:


# Observations:
# Maximum passengers have age between 20-40 years
# Survival rate is maximum for childrens and elderly


# # SibSp and Parch 

# In[ ]:


# We will create a new feature of family size = SibSp + Parch + 1


# In[ ]:


training_data['FamilySize'] = training_data['SibSp'] + training_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
fx, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].set_title('Family Size counts')
axes[1].set_title('Survival Rate vs Family Size')
fig1_family = sns.countplot(x=training_data.FamilySize, ax=axes[0], palette='cool')
fig2_family = sns.barplot(x=training_data.FamilySize, y=training_data.Survived, ax=axes[1], palette='cool')


# In[ ]:


print(training_data[['FamilySize', 'Survived']].groupby(training_data['FamilySize'], as_index=False).mean())


# In[ ]:


# As observed maximum passengers are alone but the survival is maximum for the family of 4


# # isAlone 

# In[ ]:


# wheather or not the passenger was alone ?

training_data['isAlone'] = training_data['FamilySize'].map(lambda x: 1 if x == 1 else 0)
test_data['isAlone'] = test_data['FamilySize'].map(lambda x: 1 if x == 1 else 0)


# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15, 6))
fig1_alone = sns.countplot(data=training_data, x='isAlone', ax=axes[0])
fig2_alone = sns.barplot(data=training_data, x='isAlone', y='Survived', ax=axes[1])


# In[ ]:


training_data.drop(labels=['SibSp', 'Parch'], axis=1, inplace=True)
test_data.drop(labels=['SibSp', 'Parch'], axis=1, inplace=True)
training_data.head()


# In[ ]:


# Observations:
#The maximum passengers are alone but survival rate is highest for the family of 3-4


# # Ticket

# In[ ]:


# Making a new feature ticket length

training_data['Ticket_Len'] = training_data['Ticket'].apply(lambda x: len(x))
test_data['Ticket_Len'] = test_data['Ticket'].apply(lambda x: len(x))


# In[ ]:


fx, axes = plt.subplots(2, 1, figsize=(15, 10))
axes[0].set_title("Ticket Length vs Frequency")
axes[1].set_title("Length vise Survival rate")
fig1_tlen = sns.countplot(data=training_data, x='Ticket_Len', ax=axes[0])
fig2_tlen = sns.barplot(data=training_data, x='Ticket_Len',y='Survived', ax=axes[1])


# In[ ]:


print(training_data[['Ticket_Len', 'Survived']].groupby(training_data['Ticket_Len'], as_index=False).mean())


# In[ ]:


training_data.drop(labels='Ticket', axis=1, inplace=True)
test_data.drop(labels='Ticket', axis=1, inplace=True)
training_data.head()


# In[ ]:


# Having ticket length may or may not increase acuracy as its not significant, in my case it did increase accuracy.


# # Fare

# In[ ]:


# Fare has 0 null values in training data but 1 null values in test data

test_data.Fare.describe()

# mean of fare in test data is 35 we will replace nul value with mean
test_data['Fare'][np.isnan(test_data['Fare'])] = test_data.Fare.mean()


# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15,5))
fig1_fare = sns.distplot(a=training_data.Fare, bins=15, ax=axes[0], hist_kws={'rwidth':0.7})
fig1_fare.set_title('Fare vise Frequency')

# Creating a new list of survived and dead

pass_survived_fare = training_data[training_data.Survived == 1].Fare
pass_dead_fare = training_data[training_data.Survived == 0].Fare

axes[1].hist(x=[training_data.Fare, pass_survived_fare, pass_dead_fare], bins=5, label=['Total', 'Survived', 'Dead'],         log=True)
axes[1].legend()
axes[1].set_title('Fare vise Survival')
plt.show()


# In[ ]:


# Categorizing the fare value by dividing it with 20 simply

training_data.Fare = (training_data.Fare /20).astype(np.int64) + 1 
test_data.Fare = (test_data.Fare /20).astype(np.int64) + 1 


# In[ ]:


print(training_data[['Fare','Survived']].groupby(['Fare'], as_index = False).mean())


# In[ ]:


training_data.head()


# In[ ]:


# Observations:
# The most frequent fare is between 0-100
# The survival rate is directly praportional to rate i.e. higher the rate higher the survival chances.


# # Cabin

# In[ ]:


# Null values in test data
cabin_null = float(test_data.Cabin.isnull().sum())
print(cabin_null/len(test_data) *100)


# In[ ]:


# Null values in training data
cabin_null = float(training_data.Cabin.isnull().sum())
print(cabin_null/len(training_data) *100)


# In[ ]:


# Making a new feature hasCabin which is 1 if cabin is available else 0
training_data['hasCabin'] = training_data.Cabin.notnull().astype(int)
test_data['hasCabin'] = test_data.Cabin.notnull().astype(int)


# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15, 6))
fig1_hascabin = sns.countplot(data=training_data, x='hasCabin', ax=axes[0])
fig2_hascabin = sns.barplot(data=training_data, x='hasCabin', y='Survived', ax=axes[1])


# In[ ]:


training_data.drop(labels='Cabin', axis=1, inplace=True)
training_data.head()


# In[ ]:


test_data.drop(labels='Cabin', axis=1, inplace=True)
test_data.head()


# In[ ]:


# As observed maximum population on titanic dataset does not have cabin but survival for having cabin is more.


# # Embarked

# In[ ]:


# Embarked has 2 null values in the training data


# In[ ]:


training_data.Embarked.describe()


# In[ ]:


# Since "S" is the most frequent class constituting 72% of the total therefore we will replace null values with "S"


# In[ ]:


training_data['Embarked'] = training_data['Embarked'].fillna('S')


# In[ ]:


fx, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].set_title('Embarked Counts')
axes[1].set_title('Survival Rate vs Embarked')
fig1_embarked = sns.countplot(x=training_data.Embarked, ax=axes[0])
fig2_embarked = sns.barplot(x=training_data.Embarked, y=training_data.Survived, ax=axes[1])


# In[ ]:


print(training_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean())


# In[ ]:


# Observations:
# The maximum passengers are from Southampton
# The maximum survival rate is of the passengers who boarded from Cherbourg


# In[ ]:


print(training_data[['Embarked', 'Fare']].groupby(['Embarked'], as_index = False).mean())


# In[ ]:


# If we observe the fare as grouped by boarding ststion 
# we observe that the most premium customers boarded from Cherbourg therefore maximum survival rate .


# ---

# # Cleaning the data for Classification

# In[ ]:


# Splitting the dataset into dependent and independent features
training_data.head()


# In[ ]:


X = training_data.iloc[:, 1:12].values
y = training_data.iloc[:, 0].values


# In[ ]:


# Resolving the categorical data for training set


# In[ ]:


label_encoder_sex_tr = LabelEncoder()
label_encoder_title_tr = LabelEncoder()
label_encoder_embarked_tr = LabelEncoder()
X[:, 1] = label_encoder_sex_tr.fit_transform(X[:, 1])
X[:, 5] = label_encoder_title_tr.fit_transform(X[:, 5])
X[:, 4] = label_encoder_embarked_tr.fit_transform(X[:, 4])


# In[ ]:


# Splitting the dataset into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17)


# In[ ]:


# Feature Scaling

scaler_x = MinMaxScaler((-1,1))
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)


# # Classifier Model Comparision

# In[ ]:


# Making a list of accuracies
accuracies = []


# # Logistic Regression

# In[ ]:


classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[ ]:


lr_score = classifier.score(X_test, y_test)
accuracies.append(lr_score)
print(lr_score)


# # SVM

# In[ ]:


svm = SVC(kernel='linear')
svm.fit(X_train, y_train)


# In[ ]:


svm_score = svm.score(X_test, y_test)
accuracies.append(svm_score)
print(svm_score)


# # Kernel SVM

# In[ ]:


k_svm = SVC(kernel='rbf')
k_svm.fit(X_train, y_train)


# In[ ]:


k_svm_score = k_svm.score(X_test, y_test)
accuracies.append(k_svm_score)
print(k_svm_score)


# # K- Nearest Neighbours KNN

# In[ ]:


knn = KNeighborsClassifier(p=2, n_neighbors=10)
knn.fit(X_train, y_train)


# In[ ]:


knn_score = knn.score(X_test, y_test)
accuracies.append(knn_score)
print(knn_score)


# # Random Forest

# In[ ]:


rdmf = RandomForestClassifier(n_estimators=20, criterion='entropy')
rdmf.fit(X_train, y_train)


# In[ ]:


rdmf_score = rdmf.score(X_test, y_test)
rdmf_score_tr = rdmf.score(X_train, y_train)
accuracies.append(rdmf_score)
print(rdmf_score)
print(rdmf_score_tr)


# # XgBoost

# In[ ]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)


# In[ ]:


xgb_score = xgb.score(X_test, y_test)
accuracies.append(xgb_score)
print(xgb_score)


# In[ ]:


myLabels = ['Logistic Regression', 'SVM', 'Kernel SVM', 'KNN', 'Random Forest', 'Xgboost']


# In[ ]:


fig1_accu= sns.barplot(x=accuracies, y=myLabels)


# In[ ]:


# As observed Xgboost performs best.
# We will be making three submissions
# Random Forest
# K-Svm
# Xgboost
# Since Random Forest scores best after submission we will apply Grid Search CV on RF


# # Applying Grid search cv for Parameter Tuning of Random Forest
# 
# from sklearn.model_selection import GridSearchCV
# 
# rf = RandomForestClassifier(max_features='auto')
# 
# param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12], "n_estimators": [50, 100, 400, 700]}
# 
# gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
# 
# gs = gs.fit(X, y)
# 
# print(gs.best_score_)
# print(gs.best_params_)

# # Making the Prediction

# In[ ]:


# Preparing test data 
test_data['Title'] = test_data['Title'].replace('Dona.', 'Mrs.')
test_data.head()


# In[ ]:


titanic_test = test_data.iloc[:, 0:11].values


# In[ ]:


# Taking care of categorical data

titanic_test[:, 1] = label_encoder_sex_tr.transform(titanic_test[:, 1])
titanic_test[:, 5] = label_encoder_title_tr.transform(titanic_test[:, 5])
titanic_test[:, 4] = label_encoder_embarked_tr.transform(titanic_test[:, 4])


# In[ ]:


# Feature Scaling

titanic_test = scaler_x.transform(titanic_test)


# In[ ]:


y_pred = rdmf.predict(titanic_test)


# In[ ]:


len(y_pred)


# In[ ]:


titanic_submission = pd.DataFrame({'PassengerId':passengerId, 'Survived':y_pred})


# In[ ]:


titanic_submission.to_csv('rdmf_Titanic.csv', index=False)


# # Thanks for Viewing this kernel feel free to ask or suggest.
