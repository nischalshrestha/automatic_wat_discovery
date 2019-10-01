#!/usr/bin/env python
# coding: utf-8

# For this kernel, I will try different machine learning classification models to predict the survival in the Titanic tragety.
# 
# 1. The simple logistic regression has prediction score 0.74641, rank 9565, bad
# 1. The Support Vector Classification with GridSearchCV  has scored 0.79, ranked up 2373
# 1. The Support Vector Classification with with normalized numeric feature,   has scored 0.78947, ranked up 4448
# 1. Random forest classifier

# * I am still working on improving this kernel. I will keep updating my tries and whether they work or not. 
# * If you think my kernel is helpful, please give me a voteup. This is very important for new people like me. Thank you in advance.
# * If you have any question, please feel free to leave me a message, I will check every day. Thank you so much.
# 

# ## Part 1: Import the libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats


# ## Part 2: Load the train and test data and check data

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


# Check number of observations in the train and test data
print(train.shape)
print(test.shape)


# In[ ]:


# check the columns in train and test
print(train.columns)
print(test.columns)
print('\n')
print('Variables in test but not in train is: ', set(train.columns)-set(test.columns))


# In[ ]:


# Check the first and last five observations of train and test data
train.head()


# In[ ]:


test.head()


# In[ ]:


# Check data type and NAN value
print(train.info())
print(test.info())


# In[ ]:


# Check basic descriptive information for numeric features
train.describe()


# ## Part 3: Data Visualization and feature selection

# ### 3.1 Count plot to check how many people died
# 
# * Survival 0 = No, 1 = Yes
# * We can see that about 62% died and 38% alive

# In[ ]:


sns.countplot(x=train['Survived'])
train['Survived'].value_counts(normalize=True)


# ### 3.2 Check whether the survival has any relationship with Gender
# * From the count plot and two way contigency table, 74% of female survive and only 19% of male survive.
# * Female are more likely to survive in the Titanic tragedy

# In[ ]:


sns.countplot(train['Survived'], hue=train['Sex'])
# Add contigency table for Sex by Survived
pd.crosstab(train['Sex'],train['Survived'], normalize='index')


# ### 3.3 Check the relationship between Survival and Economic class
# 
# pclass: A proxy for socio-economic status (SES)
# * 1st = Upper
# * 2nd = Middle
# * 3rd = Lower
# 
# From the Count plot and contigency table, 
# * Upper class with Pclass=1 has survival rate 63%
# * Middle class with Pclass=2 has survival rate 47%
# * Lower class with Pclass=3 has survival rate of only 24%
# 
# So Higher Economic class people are more likely to survive than people in lower Economic class 

# In[ ]:


sns.countplot(train['Survived'], hue=train['Pclass'])
# Add contigency table for Pclass by Survived
pd.crosstab(train['Pclass'],train['Survived'], normalize='index')


# ### 3.4 Check age information and Survival by Age
# 
# * We can see that there are some children and very senior people on the boat
# * Children under 10 has more than 50% of survival rate
# * The survival rate decreases as age grow. Child under 5% has survival rate of 67.5% and senior people >60 has only survival rate of 27%

# In[ ]:


plt.figure(figsize=(7,7))
plt.hist(train['Age'].dropna(), bins=30)
# We can see that there are some children and very senior people on the boat


# * Create age group and check the relationship between Survived and Age

# In[ ]:


print('The minimum age is: ', train['Age'].min())
print('The maximum age is: ', train['Age'].max())

def age_group(Age):
    if Age<5:
        return "Group 1: < 5 Years old"
    elif Age<10:
        return "Group 2: 5-10 Years old"
    elif Age<20:
        return "Group 3: 10-20 Years old"
    elif Age<40:
        return "Group 4: 20-40 Years old"
    elif Age<60:
        return "Group 5: 40-60 Years old"
    else:
        return "Group 6: >= 60 Years old"
train['Age_Group']=train['Age'].apply(age_group)


# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(train['Age_Group'], hue=train['Survived'])
pd.crosstab(train['Age_Group'],train['Survived'], normalize='index' )


# ### 3.5 Survived and number of Sibling and spouse
# 
# sibsp: The dataset defines family relations in this way...
# * Sibling = brother, sister, stepbrother, stepsister
# * Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# (1) Most people do not have siblings or spouses on boat
# 
# (2) People with 1 sibling or spouse has highest survival rate of 53.6%

# In[ ]:


# Check the number of siblings and spouse
sns.countplot(train['SibSp'])
# Most people do not have siblings or spouses on boat


# In[ ]:


sns.countplot(train['SibSp'], hue=train['Survived'])
pd.crosstab(train['SibSp'], train['Survived'], normalize='index')


# ### 3.6 Survived and number of Parent/Child
# 
# parch: The dataset defines family relations in this way...
# * Parent = mother, father
# * Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.
# 
# (1) Most people do not bring parents or children

# In[ ]:


# Check number of parents or children, 
print(train['Parch'].value_counts())
sns.countplot(train['Parch'])
# Most people do not bring parents or children


# In[ ]:


sns.countplot(train['Parch'], hue=train['Survived'])
pd.crosstab(train['Parch'], train['Survived'], normalize='index')


# ### 3.7 Survived and Fare
# 
# * Most of the Fare price are under 50
# * People with Fare price >50 are more likelily to survive than people with Fare price <50. This is consistent with the Pclass.

# In[ ]:


sns.distplot(train['Fare'], kde=False)


# In[ ]:


# Create categorical variables for Fare to check whether Survived has relationship with Fare Price
print(train['Fare'].min())
print(train['Fare'].max())
def fare_cat(Fare):
    if Fare<50:
        return "C1: <50"
    elif Fare<100:
        return "C2: <100"
    elif Fare<200:
        return "C3: <200"
    elif Fare<300:
        return "C4: <300"
    else:
        return "C5: >=300"
train['Fare_Cat']=train['Fare'].apply(fare_cat)
print(pd.crosstab(train['Fare_Cat'], train['Survived']))
print(pd.crosstab(train['Fare_Cat'], train['Survived'], normalize='index'))


# ### 3.8 Survived and Embarked
# 
# Port of Embarkation 
# * C = Cherbourg, 
# * Q = Queenstown, 
# * S = Southampton 
# 
# People embarked at Cherbourg has survival rate of 55% which is much higher than the other two ports. Possibly because people at Cherbourg are richer.

# In[ ]:


sns.countplot(train['Embarked'], hue=train['Survived'])
pd.crosstab(train['Embarked'],train['Survived'], normalize='index')


# ### 3.9 Drop 'PassengerId', 'Name','Ticket' which seems not correlated with survival
# 
# * Name, Ticket Number and PassengerId seems not correlated with Survived, we will drop them for now

# In[ ]:


# Drop variables not correlated
train.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True)

TestId=test['PassengerId']
test.drop(['PassengerId', 'Name','Ticket'], axis=1, inplace=True)


# In[ ]:


train.drop(['Age_Group', 'Fare_Cat'], axis=1, inplace=True)


# 1. First we will use the numeric Age and Fare in the model
# 2. Next we will use the Age_Group and Fare_Cat

#  ## Part 4: Check and impute missing values

# In[ ]:


# use seaborn.heatmap to check NAN values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# From the heatmap, Age has some NAN values and most of the Cabin are missing, Embarked has few missing values
# Check missing percentage
train_nan_pct=((train.isnull().sum())/(train.isnull().count())).sort_values(ascending=False)
train_nan_pct[train_nan_pct>0]


# In[ ]:


# Drop the Cabin columns since too many NAN
train.drop(['Cabin'], axis=1, inplace=True)

# Since Age is skewed, impute with median
train['Age'].fillna(train['Age'].median(), inplace=True)
# Fill the numeric Embarked with mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)


# In[ ]:


# use seaborn.heatmap to check NAN values for test data
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# From the heatmap, Age has some NAN values and most of the Cabin are missing, Fare has few missing values
# Check missing percentage
test_nan_pct=((test.isnull().sum())/(test.isnull().count())).sort_values(ascending=False)
test_nan_pct[test_nan_pct>0]


# In[ ]:


test.drop(['Cabin'], axis=1, inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Check whether impute Age by Pclass will imporve prediction performance


# ## Part 5: Create dummy variables for categorical variables
# 
# scipy.sklearn could not deal with categorical variables directly. So we need to first create dummy variables. We use drop_first=True to avoid multicolinearity

# In[ ]:


train=pd.get_dummies(train, drop_first=True)
test=pd.get_dummies(test, drop_first=True)


# ## Part 6: Build Logistic Regression to predict survival
# 
# Logistic regression is a classifical statistical model for classification and is good for intepretation. For logistic regression, we build the hypothesis function log((p(y=1|X))/(1-p(y=1|X)))=theta*X. So the probability of y=1 given X is p(y=1|X)=(exp(theta*X))/(1+exp(theta*X)). 

# ### 6.1 Train/validation split the train data set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val=train_test_split(train.drop(['Survived'], axis=1), train['Survived'], test_size=0.3, random_state=100)


# ### 6.2 Logistic Regression
# 
# 9565/score=0.74641

# In[ ]:


from sklearn.linear_model import LogisticRegression
cs=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
score=[]
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
for c in cs:
    lr=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=c, fit_intercept=True, intercept_scaling=1, class_weight=None)
    lr.fit(X_train, y_train)
    predicted=lr.predict(X_val)
    score.append(accuracy_score(predicted, y_val))
plt.scatter(x=cs, y=score)
score=pd.Series(score, index=cs)
print(score.argmax())
print(score.max())


# In[ ]:


from sklearn.linear_model import LogisticRegression
cs=np.arange(6,12, 0.2)
score=[]
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
for c in cs:
    lr=LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=c, fit_intercept=True, intercept_scaling=1, class_weight=None)
    lr.fit(X_train, y_train)
    predicted=lr.predict(X_val)
    score.append(accuracy_score(predicted, y_val))
plt.scatter(x=cs, y=score)
score=pd.Series(score, index=cs)
print(score.argmax())
print("The best accuracy score is: ", score.max())


# In[ ]:


lr=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=6, fit_intercept=True, intercept_scaling=1, class_weight=None)
lr.fit(train.drop(['Survived'], axis=1), train['Survived'])
test_predicted=lr.predict(test)


# In[ ]:


submission=pd.DataFrame()
submission['PassengerId']=TestId
submission['Survived']=test_predicted
submission.to_csv('submission.csv', index=False)


# ## Part 7. SVM classifier

# ### 7.1 Normalize the numeric features

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler(copy=True, with_mean=True, with_std=True)
sub_features=train[['Age','Fare']]
sub_features
scaler.fit(sub_features)
scaler.fit_transform(sub_features)


# In[ ]:


train[['Age','Fare']]=scaler.fit_transform(sub_features)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val=train_test_split(train.drop(['Survived'], axis=1), train['Survived'], test_size=0.3, random_state=100)


# ### 7.2 Build simple SVC model
# 
# Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
# 
# In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.

# In[ ]:


from sklearn.svm import SVC
svc=SVC(C=1, kernel='rbf', tol=0.001)
svc.fit(X_train, y_train)


# ### 7.2 Model evaluation

# In[ ]:


predicted=svc.predict(X_val)
print(confusion_matrix(y_val, predicted))
print('\n')
print(classification_report(y_val, predicted))
print('\n')
print('Accuracy score is: ', accuracy_score(y_val, predicted))
# The accuracy score is not good by using the above parameters. We will tune the hyperparameters using GridSearchCV


# ### 7.3 GridSearchcv to tune hyperparameters
# 
# * GridSearchCV exhaustive search over specified parameter values for an estimator.

# In[ ]:


# Create a dictionary called param_grid and fill out some parameters for C and gamma.
# 'gamma':auto has  aaccuracy score of 0.8260
#param_grid = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300], 'kernel': ['rbf'], 'gamma':['auto']}
# 'gamma':[0.1] has better aaccuracy of 0.8271
param_grid = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300], 'kernel': ['rbf'], 'gamma': [10,1,0.1,0.01,0.001,0.0001]}

from sklearn.model_selection import GridSearchCV
# we can specify scoring='accuracy' (default), 'precision', 'f1', 'recall' to choose parameters
grid=GridSearchCV(SVC(), param_grid, verbose=1,  scoring='accuracy', refit=True)
grid.fit(train.drop(['Survived'], axis=1), train['Survived'])


# In[ ]:


# The best hyperparameters chosen is
print(grid.best_params_)
print(grid.best_estimator_)
print('Mean cross-validated score of the best_estimator: ', grid.best_score_)
print('The number of cross-validation splits (folds/iterations): ', grid.n_splits_)


# In[ ]:


# Re-tune the hyperparameters based on previous results C=9, gamma=0.05, 0.8316498316498316
param_grid = {'C': np.arange(1,20), 'kernel': ['rbf'], 'gamma': [0.01,0.03,0.04, 0.05, 0.06, 0.07, 0.1,0.13,0.15,0.17,0.2,0.23,0.25,0.27,0.3]}

from sklearn.model_selection import GridSearchCV
# we can specify scoring='accuracy' (default), 'precision', 'f1', 'recall' to choose parameters
grid=GridSearchCV(SVC(), param_grid, verbose=1,  scoring='accuracy', refit=True)
grid.fit(train.drop(['Survived'], axis=1), train['Survived'])

# The best hyperparameters chosen is
print(grid.best_params_)
print(grid.best_estimator_)
print('Mean cross-validated score of the best_estimator: ', grid.best_score_)
print('The number of cross-validation splits (folds/iterations): ', grid.n_splits_)


# ### Use all training data to fit model with the best parameters
# {'C': 9, 'gamma': 0.05, 'kernel': 'rbf'}
# 

# ### Normalize the test features

# In[ ]:


test_sub_features=test[['Age','Fare']]
test_sub_features
scaler.fit(test_sub_features)
scaler.fit_transform(test_sub_features)
test[['Age','Fare']]=scaler.fit_transform(test_sub_features)


# In[ ]:


test.head()


# In[ ]:


svc=SVC(C=9, gamma=0.05, kernel='rbf', tol=0.001)
svc.fit(train.drop(['Survived'], axis=1), train['Survived'])
test_predicted=svc.predict(test)
submission=pd.DataFrame()
submission['PassengerId']=TestId
submission['Survived']=test_predicted
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




