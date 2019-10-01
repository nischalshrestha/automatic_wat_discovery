#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from scipy import stats
from scipy.stats import norm, skew


# In[ ]:


# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# making copies of original datasets for rest of this kernel
df_train = train.copy()
df_test = test.copy()


# In[ ]:


target = df_train['Survived']  #target variable
df_train = df_train.drop('Survived', axis=1) #drop target variable from training dataset
df_train['training_set'] = True #assing an extra variable to training and testing dataset before joining them
df_test['training_set'] = False
df_full = pd.concat([df_train, df_test]) #concatenate both dataframes prior to EDA


# In[ ]:


df_full.head()


# In[ ]:


# Feature selection
df_full = df_full.drop('PassengerId', axis=1)
df_full = df_full.drop('Name', axis=1)
df_full = df_full.drop('Ticket', axis=1)


# In[ ]:


df_full.head()


# In[ ]:


# Feature Engineering
# Replace Cabin with a feature that tells whether a passenger had a cabin on the Titanic
df_full['InCabin'] = df_full['Cabin'].apply(lambda x: 0 if type(x) == float else 1) 
df_full = df_full.drop('Cabin', axis=1)
# New feature noting if a passenger was travelling alone
df_full['IsAlone'] = 0
df_full['FamilySize'] = df_full.SibSp + df_full.Parch + 1
df_full.loc[df_full['FamilySize'] == 1, 'IsAlone'] = 1


# In[ ]:


df_full.head()


# In[ ]:


# The previous cell output seems to indicate that are missing values. Let's verify that.
df_full.isnull().sum()


# In[ ]:


# Remove NaNs
df_full['Embarked'] = df_full['Embarked'].fillna("U")
df_full['Fare'] = df_full['Fare'].fillna(df_full['Fare'].median())
df_full['Age'] = df_full['Age'].fillna(df_full['Age'].median())


# In[ ]:


# checking that we no longer have missing values
df_full.isnull().sum()


# In[ ]:


df_full.head()


# In[ ]:


# New categorial feature for age which may provide better classification
df_full['Categ_Age'] = 0
df_full.loc[df_full['Age'] < 10, 'Categ_Age'] = 0
df_full.loc[df_full['Age'] >= 10, 'Categ_Age'] = 1
df_full.loc[df_full['Age'] >= 18, 'Categ_Age'] = 2
df_full.loc[df_full['Age'] >= 25, 'Categ_Age'] = 3
df_full.loc[df_full['Age'] >= 35, 'Categ_Age'] = 4
df_full.loc[df_full['Age'] >= 45, 'Categ_Age'] = 5
df_full.loc[df_full['Age'] >= 55, 'Categ_Age'] = 6
df_full.loc[df_full['Age'] >= 65, 'Categ_Age'] = 7


# In[ ]:


df_full = df_full.drop('Age', axis=1)
df_full.head()


# In[ ]:


#convert categorical variable into dummy
df_full = pd.get_dummies(df_full)


# In[ ]:


df_full.head()


# ### Correlation Map

# In[ ]:


#Correlation map to see how features are correlated with SalePrice
corrmat = df_full.corr()
plt.subplots(figsize=(10,10))
sns.heatmap(corrmat,square=True, cmap="YlGnBu");


# ## Building Machine Learning Model(s)

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # import 'train_test_split'
from sklearn.ensemble import RandomForestClassifier # import RandomForestRegressor
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer # import metrics from sklearn
from time import time
from sklearn.model_selection import GridSearchCV # Search over specified parameter values for an estimator.
from sklearn.model_selection import RandomizedSearchCV # Search over specified parameter values for an estimator.
from sklearn.model_selection import ShuffleSplit # Random permutation cross-validator


# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 5))
df_full[['Fare']] = scaler.fit_transform(df_full[['Fare']])
#df_full[['Age']] = scaler.fit_transform(df_full[['Age']])


# In[ ]:


df_full.head()


# In[ ]:


df_train = df_full[df_full['training_set']==True]
df_train = df_train.drop('training_set', axis=1)
df_test = df_full[df_full['training_set']==False]
df_test = df_test.drop('training_set', axis=1)


# In[ ]:


(df_train.shape, df_test.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_train, target, random_state=42)


# ### AdaBoostClassifier

# In[ ]:


from sklearn.ensemble  import AdaBoostClassifier


# In[ ]:


ada_classifier = AdaBoostClassifier(random_state=42)
#cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
#parameters = {'n_estimators':[500, 1000, 1500, 2000], 
#              'learning_rate':[0.05, 0.1, 0.15, 0.2]}
#scorer = make_scorer(f1_score)
#ada_obj = RandomizedSearchCV(ada_classifier, 
#                              parameters, 
#                              scoring = scorer, 
#                              cv = cv_sets,
#                              random_state= 99)
#ada_fit = ada_obj.fit(X_train, y_train)
#ada_opt = ada_fit.best_estimator_


# In[ ]:


#ada_fit.best_params_


# In[ ]:


ada_obj = AdaBoostClassifier(learning_rate = 0.1,
                             n_estimators = 2000,
                             random_state=42)
ada_opt = ada_obj.fit(X_train, y_train)


# ### GradientBoostingClassifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
GBC_classifier = GradientBoostingClassifier(random_state=42)
#cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
#parameters = {'n_estimators':[500, 1000, 1500], 
#              'learning_rate':[0.01, 0.03, 0.05],
#              'min_samples_split':[2,4,6],
#              'min_samples_leaf':[3,5,7]}
#scorer = make_scorer(f1_score)
#GBC_obj = RandomizedSearchCV(GBC_classifier, 
#                             parameters, 
#                             scoring = scorer, 
#                             cv = cv_sets,
#                             random_state= 99)
#GBC_fit = GBC_obj.fit(X_train, y_train)
#GBC_opt = ada_fit.best_estimator_


# In[ ]:


#GBC_fit.best_params_


# In[ ]:


GBC_obj = GradientBoostingClassifier(learning_rate = 0.05,
                                     max_depth = 3,
                                     min_samples_leaf = 5,
                                     min_samples_split = 4,
                                     n_estimators = 500,
                                     random_state=42)
GBC_opt = GBC_obj.fit(X_train, y_train)


# ### Submission

# In[ ]:


# Get the predictions for df_test f
y_pred_GBC = GBC_opt.predict(df_test)
y_pred_ada = ada_opt.predict(df_test)


# In[ ]:


y_pred_final = 0.5*y_pred_GBC + 0.5*y_pred_ada
y_pred_final = y_pred_final.astype(int)


# In[ ]:


# Final submission
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred_final})
my_submission.to_csv('submission-160518.csv', index=False)

