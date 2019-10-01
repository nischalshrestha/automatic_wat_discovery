#!/usr/bin/env python
# coding: utf-8

# Load in our Titanic data:

# In[ ]:


import numpy as np
import pandas as pd
import os

titanic_full_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')


# Start with looking at the data.

# In[ ]:


titanic_full_train.head(10)


# In[ ]:


titanic_full_train.describe(include='all')


# Items to note regarding the data:
# - We are missing values for Age, Cabin, and Embarked
#     - Embarked is only missing two values
#     - Age is missing 177 values
#     - Cabin only has 204 of the 891 values
# - Pclass only has values of 1, 2, or 3
# - Sex only has values of 'male' or 'female'
# - SibSp and Parch are the counts of siblings/spouses and parents/children on board, respectively, and are predominantly zero
# - Embarked only has values of C, Q, and S
#     

# Now let's look at some plots of the data:

# In[ ]:


fields = ['Survived', 'Age', 'Fare', 'Parch', 'SibSp', 'Pclass']
titanic_full_train[fields].hist(bins=25, figsize=(20,15))


# Items to note from the plots:
# - regarding age, most people are in their twenties, but there are a number of small children as well
# - the majority of fares are very low, with few larger fares
# - most had no parents/children on board, but there were some
# - there were more in third class than first and second
# - most had no siblings or spouses, but there were some (decent number of 1 - probably spouses?)
# - the survival rate was not very good

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.impute import SimpleImputer

titanic_y = titanic_full_train.Survived
clf = GradientBoostingClassifier()
titanic_X_colns = ['PassengerId', 'Age', 'Fare',]
titanic_X = titanic_full_train[titanic_X_colns]
my_imputer = SimpleImputer()
imputed_titanic_X = my_imputer.fit_transform(titanic_X)

clf.fit(imputed_titanic_X, titanic_y)
titanic_plots = plot_partial_dependence(clf, features=[1,2], X=imputed_titanic_X, 
                                        feature_names=titanic_X_colns, grid_resolution=10)


# From the above plots, survival chances increased for the young (women and children got off the boat first?) and those with higher fares also had increased survival chances (closer to lifeboats?).

# In[ ]:


y = titanic_full_train.Survived
X = titanic_full_train.drop(['Survived'], axis=1)


# Clean the data and prepare for modeling.  Remove the fields we aren't using.
# 
# I will remove Name and Ticket, as these won't be included in the model.  I will also remove Cabin, because there are so many missing values.

# In[ ]:


X = X.drop(['Name', 'Ticket', 'Cabin'], axis=1)


# In[ ]:


#use one-hot encoding for categoricals (Sex, Embarked) using get_dummies
OHE_X = pd.get_dummies(X)


# In[ ]:


#review the data we now have
OHE_X.describe(include='all')


# We now have indicator variables for each value of Sex and Embarked, rather than text.
# And all of our data is numerical now.

# For Age, there are a number of missing values, but this variable appears to be important, so we will impute the missing values so we can use this element in the model.

# In[ ]:


#keep track of what was imputed
X_plus = OHE_X.copy()

cols_with_missing = (col for col in OHE_X.columns 
                                 if OHE_X[col].isnull().any())
for col in cols_with_missing:
    X_plus[col + '_was_missing'] = X_plus[col].isnull()

X_plus = my_imputer.fit_transform(X_plus)


# In[ ]:


#split the data into training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_plus,
                                                    y, 
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify = y)


# In[ ]:


#try random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)


# In[ ]:


#try XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)


# In[ ]:


#try logistic regression

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(X_train, y_train)
lg.score(X_test, y_test)


# In[ ]:


#try randomized search cv

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': range(8, 20),
    'max_depth': range(6, 10),
    'learning_rate': [.4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1]
}

gbm = XGBClassifier(n_estimators=10)

xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = gbm, scoring = "accuracy", 
                                    verbose = 1, n_iter = 50, cv = 4)

X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)
xgb_random.fit(X, y)

print("Best parameters found: ", xgb_random.best_params_)
print("Best accuracy found: ", xgb_random.best_score_)


# In[ ]:


# Treat the test data in the same way as training data:
# use OHE, then impute and track imputed values
OHE_X_submit = pd.get_dummies(titanic_test)
OHE_X, OHE_X_submit = OHE_X.align(OHE_X_submit,
                                  join='inner', 
                                  axis=1)

OHE_X_submit_plus = OHE_X_submit.copy()
cols_with_missing = (col for col in OHE_X.columns 
                                 if OHE_X[col].isnull().any())
for col in cols_with_missing:
    OHE_X_submit_plus[col + '_was_missing'] = OHE_X_submit_plus[col].isnull()
OHE_X_submit_plus = my_imputer.transform(OHE_X_submit_plus)


# In[ ]:


# Use the model to make predictions
xgb_pred = xgb_random.predict(OHE_X_submit_plus)
submission = pd.concat([titanic_test.PassengerId, pd.DataFrame(xgb_pred)], axis = 'columns')
submission.columns = ["PassengerId", "Survived"]
submission.to_csv('titanic_submit.csv', header=True, index=False)

