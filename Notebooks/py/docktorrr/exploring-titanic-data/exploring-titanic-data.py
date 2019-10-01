#!/usr/bin/env python
# coding: utf-8

# # Titanic dataset exploration and survival prediction

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing, grid_search, metrics, linear_model, neighbors, svm, ensemble
import xgboost as xgb


# ## Reading data

# In[ ]:


data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# In[ ]:


data_train.head()


# In[ ]:


print(data_train.info())
print(data_test.info())


# ## Analyzing dataset features

# The dataset has features that identifies a passenger, these features (PassengerId, Name, Ticket) do not influence the output result. So we can remove them from dataset

# In[ ]:


# save PassengerId from test dataset to put in the results later
pass_ids = data_test[['PassengerId']]
# remove PassengerId field that unique for every passenger
data_train = data_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
data_test = data_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)


# The dataset has two numerical features (Age and Fare), the rest are categorial features. Let's analyze each feature and it's impact on the result value (Survival).

# ### Survived (data label)

# In[ ]:


data_train.Survived.value_counts()


# ### Pclass (passenger class)

# In[ ]:


# as class is categorical feature we can make its type as string
data_train['Pclass'] = data_train['Pclass'].astype(str)


# In[ ]:


# possible class values
print(data_train['Pclass'].value_counts().sort_index())


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))

subplt = data_train[data_train.Pclass=='1']['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='blue')
subplt.set_title('Class 1')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])

subplt = data_train[data_train.Pclass=='2']['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='green')
subplt.set_title('Class 2')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])

subplt = data_train[data_train.Pclass=='3']['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[2], color='pink')
subplt.set_title('Class 3')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])


# We can see that there's a survival dependency on the passenger class. Passengers of higher class had a higher chance of survival.

# ### Sex

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

subplt = data_train[data_train.Sex=='male']['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='blue')
subplt.set_title('Male')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])

subplt = data_train[data_train.Sex=='female']['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='green')
subplt.set_title('Female')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])


# Females had a much bigger chance of survival

# ### Age

# In[ ]:


axis = data_train['Age'].dropna().astype(int).plot(kind='hist', bins=80)
axis.set_title('Age distribution')


# Let's fill NaN age with random values (normally distributed with mean and standard deviation of the known age distribution) 

# In[ ]:


# train data
mean_age = data_train['Age'].mean()
std_age = data_train['Age'].std()
print('Mean: {}, standard deviation: {}'.format(mean_age, std_age))
tofillwith = np.floor(np.random.normal(mean_age, std_age, data_train['Age'].isnull().sum()))
tofillwith[tofillwith < 0] = 0 # random value can be negative
data_train['Age'][pd.isnull(data_train["Age"])] = tofillwith


# In[ ]:


# check new distribution
axis = data_train['Age'].astype(int).plot(kind='hist', bins=80)
axis.set_title('Age distribution')


# In[ ]:


# test data
mean_age = data_test['Age'].mean()
std_age = data_test['Age'].std()
print('Mean: {}, standard deviation: {}'.format(mean_age, std_age))
tofillwith = np.floor(np.random.normal(mean_age, std_age, data_test['Age'].isnull().sum()))
tofillwith[tofillwith < 0] = 0 # random value can be negative
data_test['Age'][pd.isnull(data_test["Age"])] = tofillwith


# In[ ]:


plt.figure(figsize=(9,3))
data_train[data_train.Survived==1]['Age'].plot(kind='density', label='Survived')
data_train[data_train.Survived==0]['Age'].plot(kind='density', label='Died')
plt.xlabel('Age')
plt.legend()
plt.title('Age distribution')


# ### SibSp

# In[ ]:


data_train.SibSp.value_counts()


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
ax1= sns.countplot(x='SibSp', data=data_train[data_train.Survived==1], ax=axes[0])
ax2 = sns.countplot(x='SibSp', data=data_train[data_train.Survived==0], ax=axes[1])
ax1.set_title('Survived')
ax2.set_title('Died')


# ### Parch

# In[ ]:


data_train.Parch.value_counts()


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
ax1= sns.countplot(x='Parch', data=data_train[data_train.Survived==1], ax=axes[0])
ax2 = sns.countplot(x='Parch', data=data_train[data_train.Survived==0], ax=axes[1])
ax1.set_title('Survived')
ax2.set_title('Died')


# In[ ]:


# Check dependency of SibSp and Parch features
plt.scatter(data_train['SibSp'], data_train['Parch'])


# It seems that SibSp and Parch features are quite independent

# ### Fare

# In[ ]:


plt.figure(figsize=(9,3))
data_train[data_train.Survived==1]['Fare'].plot(kind='density', label='Survived')
data_train[data_train.Survived==0]['Fare'].plot(kind='density', label='Died')
plt.xlabel('Fare')
plt.legend()
plt.title('Fare distribution')


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))

subplt = data_train[data_train.Fare<=10.]['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0])
subplt.set_title('Fare < 10')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])

subplt = data_train[(data_train.Fare>10.0) & (data_train.Fare<=50.0)]['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1])
subplt.set_title('Fare from 10 to 50')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])

subplt = data_train[(data_train.Fare>50.0) & (data_train.Fare<=100.0)]['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0])
subplt.set_title('Fare from 50 to 100')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])

subplt = data_train[data_train.Fare>100.0]['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1])
subplt.set_title('Fare > 100')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])


# People with low fare had less chances of survival

# We have one NaN value in test data so we can set it to mean (from both train and test datasets)

# In[ ]:


mean_fare = (data_train['Fare'].sum() + data_test['Fare'].sum()) / (data_train['Fare'].count() + data_test['Fare'].count())
data_test['Fare'] = data_test['Fare'].fillna(mean_fare)


# ### Cabin

# In[ ]:


print(data_train.Cabin.isnull().sum())


# We can remove this column as about querter of objects have NaN values, so it has no influence on result.

# In[ ]:


data_train.drop(['Cabin'], axis=1, inplace=True)
data_test.drop(['Cabin'], axis=1, inplace=True)


# ### Embarked

# In[ ]:


data_train['Embarked'].value_counts()


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 2.5))

subplt = data_train[data_train.Embarked=='S']['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='blue')
subplt.set_title('S')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])

subplt = data_train[data_train.Embarked=='C']['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='green')
subplt.set_title('C')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])

subplt = data_train[data_train.Embarked=='Q']['Survived'].value_counts().sort_index().plot(kind='bar', ax=axes[2], color='pink')
subplt.set_title('Q')
subplt.set_ylabel('count')
subplt.set_xticklabels(['Died', 'Survived'])


# There are only two NaN values, so leave them as it is. Later they will disappeared after binary encoding 

# ## Preprocess data

# ### Binary encoding of categorial features

# Let's make binary encoding of categorial features: Sex, Pclass, Embarked

# In[ ]:


# encode Sex feature
data_train.Sex = np.where(data_train.Sex=='male', 1, 0)
data_test.Sex = np.where(data_test.Sex=='male', 1, 0)


# In[ ]:


# encode Pclass & Embarked feature
data_train = pd.get_dummies(data=data_train, columns=['Pclass', 'Embarked'])
data_test = pd.get_dummies(data=data_test, columns=['Pclass', 'Embarked'])


# In[ ]:


print(data_train.info())
print(data_test.info())


# ### Scaling real features

# In[ ]:


# Check before scaling
pd.tools.plotting.scatter_matrix(data_train[['Age', 'Fare']], alpha=0.5, figsize=(7, 7))
plt.suptitle('Age and Fare before scaling')
plt.show()


# We can see that real features, Age and Fare, have different scale.

# In[ ]:


scaler = preprocessing.StandardScaler()
data_train[['Age', 'Fare']] = scaler.fit_transform(data_train[['Age', 'Fare']])
data_test[['Age', 'Fare']] = scaler.transform(data_test[['Age', 'Fare']])


# In[ ]:


# Check after scaling
pd.tools.plotting.scatter_matrix(data_train[['Age', 'Fare']], alpha=0.5, figsize=(7, 7))
plt.suptitle('Age and Fare after scaling')
plt.show()


# ## Chosing classifier 

# In[ ]:


# Extract features (X) and labels (y) from data
X = data_train[data_train.columns[1:]]
y = data_train[data_train.columns[0]]


# ### Logistic regression

# In[ ]:


# Finding optimal parameters of the classifier
param_grid = {
              'C': [0.01, 0.05, 0.1, 0.5, 1],
              'penalty': ['l1', 'l2']
             }
estimator = linear_model.LogisticRegression()
lr_gs = grid_search.GridSearchCV(estimator, param_grid, cv=4)
lr_gs.fit(X, y)


# In[ ]:


lr_estimator = lr_gs.best_estimator_
print(lr_gs.best_params_)
print(lr_gs.best_score_)


# ### K Nearest Neighbors

# In[ ]:


# Finding optimal parameters of the classifier
param_grid = {
              'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
              'weights': ['uniform', 'distance'],
              'p': [1, 2]
             }
estimator = neighbors.KNeighborsClassifier()
knn_gs = grid_search.GridSearchCV(estimator, param_grid, cv=4)
knn_gs.fit(X, y)


# In[ ]:


knn_estimator = knn_gs.best_estimator_
print(knn_gs.best_params_)
print(knn_gs.best_score_)


# ### Support Vector Machine

# In[ ]:


# Finding optimal parameters of the classifier
param_grid = {
              'C': [0.5, 1, 2, 4, 10, 20],
              'kernel': ['linear', 'poly', 'rbf'],
             }
estimator = svm.SVC()
svc_gs = grid_search.GridSearchCV(estimator, param_grid, cv=4)
svc_gs.fit(X, y)


# In[ ]:


svc_estimator = svc_gs.best_estimator_
print(svc_gs.best_params_)
print(svc_gs.best_score_)


# ### Random Forest

# In[ ]:


# Finding optimal parameters of the classifier
param_grid = {
              'n_estimators': [50, 100, 500],
              'min_samples_leaf': [1, 3, 5]
             }
estimator = ensemble.RandomForestClassifier()
rf_gs = grid_search.GridSearchCV(estimator, param_grid, cv=4)
rf_gs.fit(X, y)


# In[ ]:


rf_estimator = rf_gs.best_estimator_
print(rf_gs.best_params_)
print(rf_gs.best_score_)


# In[ ]:


# display feature importance
plt.figure(figsize=(8,5))
plt.barh(np.arange(X.shape[1]), rf_estimator.feature_importances_, align='center')
plt.yticks(np.arange(X.shape[1]), X.columns)
plt.title('Feature importance')


# ### Gradient Boosting (XGBoost)

# In[ ]:


# Finding optimal parameters of the classifier
param_grid = {
              'n_estimators': [10, 50, 100],
              'max_depth': [3, 5, 10],
              'learning_rate': [0.1, 0.5]
             }
estimator = xgb.XGBClassifier()
xgb_gs = grid_search.GridSearchCV(estimator, param_grid, cv=4)
xgb_gs.fit(X, y)


# In[ ]:


xgb_estimator = xgb_gs.best_estimator_
print(xgb_gs.best_params_)
print(xgb_gs.best_score_)


# ### Comparing models

# In[ ]:


estimators = [lr_gs, knn_gs, svc_gs, rf_gs, xgb_gs]
labels = ['Logistic regression', 'KNN', 'SVC', 'Random Forest', 'XGB']

plt.figure(figsize=(8,5))
plt.barh(np.arange(5.), list(map(lambda e: e.best_score_, estimators)), align='center')
plt.yticks(np.arange(5.), labels)
plt.xlim(0.7, 0.9)
plt.title('Models comparison')


# The best classifier is SVC with radial kernel, C = 4.

# In[ ]:


estimator = svc_estimator


# ## Predict and submit

# In[ ]:


y_test = estimator.predict(data_test)
result_df = pd.DataFrame(columns=['PassengerID', 'Survived'])
result_df.PassengerID = pass_ids
result_df.Survived = y_test
result_df.head()


# In[ ]:


result_df.to_csv('titanic_results.csv', index=False)

