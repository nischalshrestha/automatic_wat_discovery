#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Like most people, the Titanic Challenge was my first (1 of 2) Kaggle competitions and my first machine learning project outside of econometrics courses. There isn't anything fancy in this notebook; no automated feature engineering nor any hyperparameter tuning (I am working on getting better on these things on my own time, but applying them to Kaggle competitions as a full-time student and part-time worker is difficult time-wise). I hope this serves as a decent guide to those starting on their first model, so they can take what they learn here and improve on it. Good luck on your machine learning journey.

# # Setting the Environment

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
get_ipython().magic(u'matplotlib inline')
import seaborn as sns # data visualization
sns.set()


# # Importing the data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # Data Exploration
# 
# This section looks at the shape of the data (how many observations and variables) and looks at basic desciptions of the values for each variable.

# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.describe()


# In[ ]:


train.describe(include = ['O'])


# In[ ]:


train.info


# In[ ]:


train.isnull().sum()


# In[ ]:


test.shape


# In[ ]:


test.describe()


# In[ ]:


test.describe(include = ['O'])


# In[ ]:


test.info


# In[ ]:


test.isnull().sum()


# # Finding Relationships between Independent Variables and Survival

# Quick note, PassengerId and Name are indentifiers and should not affect survival rating.

# In[ ]:


survived = train[train.Survived == 1]
not_survived = train[train.Survived == 0]
print('Survived: %i (%.1f%%)' %(len(survived), float(100*len(survived)/(len(survived)+len(not_survived)))))
print('Did not Survive: %i (%.1f%%)' %(len(not_survived), float(100*len(not_survived)/(len(survived)+len(not_survived)))))


# ## Pclass vs. Survival

# In[ ]:


print(train.Pclass.value_counts())
# Gives the number of people in each class

print(train.groupby('Pclass').Survived.value_counts())
# Counts the number of people in each class who survived and did not survive

print(train[['Pclass', 'Survived']].groupby('Pclass', as_index = False).mean())
# Counts the proportion of people in each class that survived (1 = Survived and 0 = Did Not Survive, so mean = proportion who survived)


# In[ ]:


#train.groupby('Pclass').Survived.mean().plot(kind = 'bar')
sns.barplot(x = 'Pclass', y = 'Survived', data = train)


# The results suggest that people in 1st class were most likely to survive, followed by those in 2nd class. Those in 3rd class were the least likely to survive.
# 
# Perhaps this is a result of 1st class being the closest to the deck, followed by 2nd class, while 3rd class is near the bottom of the ship.

# ## Sex vs. Survival

# In[ ]:


print(train.Sex.value_counts())
# Displays the number of each sex

print(train.groupby('Sex').Survived.value_counts())
# Displays the number of each sex that survived and did not survive

print(train.groupby('Sex').Survived.mean())
# Displays the proportion of each sex that survived or did not survive


# In[ ]:


#train.groupby('Sex').Survived.mean().plot(kind = 'bar')
sns.barplot(x = 'Sex', y = 'Survived', data = train)


# Females are far more likely than males to have survived the Titanic. I hypothesize this is due to the "women and children" first policy.

# ## Age vs. Survival

# In[ ]:


train.Age.describe()


# In[ ]:


age = train.Age.dropna()
sns.distplot(age, bins = 25, kde = False)


# The passengers on the Titanic tended to be younger adults in their 20s and 30s. Among the children, the age skews to very young.

# In[ ]:


train['AgeBand'] = np.where(train.Age <= 16, 1, 
                            np.where((train.Age > 16) & (train.Age <= 32), 2, 
                                     np.where((train.Age > 32) & (train.Age <= 48), 3, 
                                              np.where((train.Age > 48) & (train.Age <= 64), 4, 
                                                      np.where((train.Age > 64) & (train.Age <= 80), 5, False)))))


# In[ ]:


print(train.AgeBand.value_counts())
# Displays the number of each sex

print(train.groupby('AgeBand').Survived.value_counts())
# Displays the number of each sex that survived and did not survive

print(train.groupby('AgeBand').Survived.mean())
# Displays the proportion of each sex that survived or did not survive


# In[ ]:


train.groupby('AgeBand').Age.describe()


# In[ ]:


#train.groupby('AgeBand').Survived.mean().plot(kind = 'bar')
sns.barplot(x = 'AgeBand', y = 'Survived', data = train)


# People under 16 were the most likely group to survive, supporting the "women and children" hypothesis. The AgeBand variable is used only for this analysis and will not be used in the final model.

# ## Sibsp vs. Survival

# In[ ]:


train.SibSp.describe()


# Most people came without a spouse or sibling.

# In[ ]:


print(train.SibSp.value_counts())
# Displays the number of each sex

print(train.groupby('SibSp').Survived.value_counts())
# Displays the number of each sex that survived and did not survive

print(train.groupby('SibSp').Survived.mean())
# Displays the proportion of each sex that survived or did not survive


# People with 2 or fewer siblings or spouses on the Titanic (assuming a big chunk of the people with 1 SibSp are spouses) were more likely than those with many people. I hypothesize this is due to it being difficult to round up a big group in a crisis.

# In[ ]:


#train.groupby('SibSp').Survived.mean().plot(kind = 'bar')
sns.barplot(x = 'SibSp', y = 'Survived', data = train)


# ## Parch vs. Survival

# In[ ]:


train.Parch.describe()


# Most people came without their parents and children.

# In[ ]:


print(train.Parch.value_counts())
# Displays the number of each sex

print(train.groupby('Parch').Survived.value_counts())
# Displays the number of each sex that survived and did not survive

print(train.groupby('Parch').Survived.mean())
# Displays the proportion of each sex that survived or did not survive


# Having 3 or more parents or children with you is associated with high survival rates, but some number combinations have small sample sizes so there is more variation in those groups.

# In[ ]:


#train.groupby('Parch').Survived.mean().plot(kind = 'bar')
sns.barplot(x = 'Parch', y = 'Survived', data = train)


# ## Embarked Point vs. Survival

# In[ ]:


train.Embarked.describe(include = ['O'])


# In[ ]:


print(train.Embarked.value_counts())
# Displays the number of each sex

print(train.groupby('Embarked').Survived.value_counts())
# Displays the number of each sex that survived and did not survive

print(train.groupby('Embarked').Survived.mean())
# Displays the proportion of each sex that survived or did not survive


# Most people embarked at South Hampton and those who embarked at Cherbourg were the most likely to survive.

# In[ ]:


#train.groupby('Embarked').Survived.mean().plot(kind = 'bar')
sns.barplot(x = 'Embarked', y = 'Survived', data = train)


# ## FamilySize vs. Survival

# In[ ]:


train['FamilySize'] = train.SibSp + train.Parch


# In[ ]:


train.FamilySize.describe()


# Most people on the Titanic were alone, at least in this sample.

# In[ ]:


print(train.FamilySize.value_counts())
# Displays the number of each sex

print(train.groupby('FamilySize').Survived.value_counts())
# Displays the number of each sex that survived and did not survive

print(train.groupby('FamilySize').Survived.mean())
# Displays the proportion of each sex that survived or did not survive


# Having 3 or fewer family members is associated with higher likelihood of survival, and being alone is more dangerous than having another person there.

# In[ ]:


#train.groupby('FamilySize').Survived.mean().plot(kind = 'bar')
sns.barplot(x = 'FamilySize', y = 'Survived', data = train)


# ## Fare

# In[ ]:


train.Fare.describe()


# In[ ]:


sns.distplot(train.Fare, bins = 15, kde = False)


# Because Fare is so heavily skewed, I took the natural log of it to try and smooth it out.

# In[ ]:


train['logFare'] = np.where(train.Fare != 0, np.log(train.Fare), train.Fare)


# In[ ]:


sns.distplot(train.logFare, bins = 15)


# # Creating a Model

# ## Data Pre-Processing

# In[ ]:


from sklearn.impute import SimpleImputer

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['FamilySize'] = train.SibSp + train.Parch
train['logFare'] = np.where(train.Fare != 0, np.log(train.Fare), train.Fare)

cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']

train = train.drop(cols_to_drop, axis=1)
X_test = test.drop(cols_to_drop, axis=1)

train_data = pd.get_dummies(train)
X_test = pd.get_dummies(X_test)

X_train = train_data.drop('Survived', axis=1)
y_train = train_data.Survived

my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)


# ## Importing Model Modules

# In[ ]:


from xgboost import XGBClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# ### Support Vector Machine

# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.3, random_state = 0)

my_svm_model = svm.SVC(kernel='linear')
my_svm_model.fit(X_train, y_train)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_svm_model, X_train, y_train, cv=kfold)
print("SVM Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ### Random Forest

# In[ ]:


my_forest_model = RandomForestClassifier(n_estimators=50)
my_forest_model.fit(X_train, y_train)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_forest_model, X_train, y_train, cv=kfold)
print("Random Forest Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ### K-Nearest Neighbors Classifier

# In[ ]:


my_knn_model = KNeighborsClassifier(n_neighbors=4)
my_knn_model.fit(X_train, y_train)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_knn_model, X_train, y_train, cv=kfold)
print("Knn Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ### Gaussian Naive Bayes Classifier

# In[ ]:


my_gnb_model = GaussianNB()
my_gnb_model.fit(X_train, y_train)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_gnb_model, X_train, y_train, cv=kfold)
print("GNB Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ### Logistic Regression

# In[ ]:


my_logit_model = LogisticRegression()
my_logit_model.fit(X_train, y_train)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_logit_model, X_train, y_train, cv=kfold)
print("Logistic Regression Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ### XTREME GRADIENT BOOSTED TREE

# In[ ]:


my_xgb_model = XGBClassifier(n_estimators = 1000, learning_rate = 0.65)
my_xgb_model.fit(X_train, y_train, early_stopping_rounds = 5, eval_set = [(test_X, test_y)], verbose = False)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_xgb_model, X_train, y_train, cv=kfold)
print("XGBTree Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Final Model (Going with a Gradient Boosted Tree)

# In[ ]:


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['FamilySize'] = train.SibSp + train.Parch
train['logFare'] = np.where(train.Fare != 0, np.log(train.Fare), train.Fare)

cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']

train.Pclass = train.Pclass.astype(str)
train = train.drop(cols_to_drop, axis=1)
test.Pclass = test.Pclass.astype(str)
X_test = test.drop(cols_to_drop, axis=1).copy()

X_test['FamilySize'] = X_test.SibSp + X_test.Parch
X_test['logFare'] = np.where(X_test.Fare != 0, np.log(X_test.Fare), X_test.Fare)

train_data = pd.get_dummies(train)
X_test = pd.get_dummies(X_test)

X_train = train_data.drop('Survived', axis=1)
y_train = train_data.Survived

my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)

train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.25, random_state = 0)

my_xgb_model = XGBClassifier(n_estimators = 150, 
                             learning_rate = 0.05, 
                             max_depth = 3)
my_xgb_model.fit(X_train, y_train, early_stopping_rounds = 5, eval_set = [(test_X, test_y)], verbose = False)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(my_xgb_model, X_train, y_train, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

my_predictions = my_xgb_model.predict(X_test)

jcleme_submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": my_predictions})

jcleme_submission.to_csv('new_jcleme_xgb_submission.csv', index = False)

