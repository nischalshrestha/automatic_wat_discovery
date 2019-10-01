#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.columns


# In[ ]:


X_full = pd.concat([train.drop('Survived', axis = 1), test], axis = 0)


# In[ ]:


X_full.shape


# Clean X_full. Afterwards, we will split it back up into training and test sets.

# In[ ]:


X_full.drop('PassengerId', axis = 1, inplace=True)


# In[ ]:


X_full.isnull().sum()


# In[ ]:


(X_full.Age.isnull() & X_full.Cabin.isnull()).sum()


# I would guess that these people died, so we couldn't collect their information. 

# In[ ]:


train.Survived.mean()


# In[ ]:


train.Cabin.notnull().mean()


# Coincidence? Maybe not.

# In[ ]:


(train.Cabin.isnull() & (train.Survived == 0)).mean()


# In[ ]:


selector = (train.Cabin.isnull() & train.Age.isnull())

train[selector].Survived.mean()


# In[ ]:


train.Survived.mean()


# In[ ]:


selector = (train.Cabin.isnull())

train[selector].Survived.mean()


# We can conclude that not cabin_null is a good indicator of not_survived, but cabin_null and age_null is even better.

# In[ ]:


X_full['Nulls'] = X_full.Cabin.isnull().astype('int') + X_full.Age.isnull().astype('int')


# We can further divide the cabin category by simply extracting the first letter.

# In[ ]:


X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0] # this captures the letter

# this transforms the letters into numbers
cabin_dict = {k:i for i, k in enumerate(X_full.Cabin_mapped.unique())} 
X_full.loc[:, 'Cabin_mapped'] = X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)


# In[ ]:


cabin_dict


# In[ ]:


X_full.columns


# In[ ]:


X_full.drop(['Age', 'Cabin'], inplace = True, axis = 1)


# The guy with the missing fare was from thee lower class. Assume he paid the average price.

# In[ ]:


fare_mean = X_full[X_full.Pclass == 3].Fare.mean()

X_full['Fare'].fillna(fare_mean, inplace = True)


# In[ ]:


X_full.isnull().sum()


# In[ ]:


X_full[X_full.Embarked.isnull()]


# In[ ]:


X_full[X_full['Pclass'] == 1].Embarked.value_counts()


# In[ ]:


X_full['Embarked'].fillna('S', inplace = True)


# In[ ]:


X_full.isnull().sum()


# In[ ]:


X_full.drop(['Name', 'Ticket'], axis = 1, inplace = True)


# In[ ]:


X_full.dtypes.


# In[ ]:


X_dummies = pd.get_dummies(X_full, columns = ['Sex', 'Nulls', 'Cabin_mapped', 'Embarked'], drop_first= True)


# In[ ]:


X_dummies.dtypes


# In[ ]:





# Now let's train.

# In[ ]:


X = X_dummies[:len(train)]; new_X = X_dummies[len(train):]
y = train.Survived


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = .3,
                                                    random_state = 5,
                                                   stratify = y)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf.score(X_test, y_test)


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb = XGBClassifier()


# In[ ]:


xgb.fit(X_train, y_train)


# In[ ]:


xgb.score(X_test, y_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)
lg.score(X_test, y_test)


# In[ ]:


import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': range(8, 20),
    'max_depth': range(6, 10),
    'learning_rate': [.4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1]
}

# Instantiate the regressor: gbm
gbm = XGBClassifier(n_estimators=10)

# Perform random search: grid_mse
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = gbm, scoring = "accuracy", 
                                    verbose = 1, n_iter = 50, cv = 4)


# Fit randomized_mse to the data
xgb_random.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", xgb_random.best_params_)
print("Best accuracy found: ", xgb_random.best_score_)


# In[ ]:


xgb_pred = xgb_random.predict(new_X)


# In[ ]:


submission = pd.concat([test.PassengerId, pd.DataFrame(xgb_pred)], axis = 'columns')


# In[ ]:


submission.columns = ["PassengerId", "Survived"]


# In[ ]:


submission.to_csv('titanic_submission.csv', header = True, index = False)


# In[ ]:




