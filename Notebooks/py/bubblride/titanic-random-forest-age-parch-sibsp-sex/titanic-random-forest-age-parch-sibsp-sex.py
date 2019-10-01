#!/usr/bin/env python
# coding: utf-8

# ## Load Modules

# In[ ]:


import pandas as pd
from sklearn import tree
from sklearn import ensemble
import graphviz 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score


# ## Summary
# * sklearn's Random Forest classifier
# * some hyperparameter optimization with `RandomizedSearchCV`, `k=30` fold CV

# ## Prep Data
# * Age is binned in 4 age groups plus 1 for missing data

# In[ ]:


df = pd.read_csv("../input/train.csv")
df2 = df.copy()

df2['Sex'] = df['Sex'].apply(lambda e : 1 if e=='male' else (2 if e=='female' else None))
df2['Age'] = df['Age'].apply(lambda e : 5 if pd.isnull(e) else (1 if e<7 else (2 if e<20 else (3 if e<37 else 4))))

y = df2['Survived'].values #.reshape(-1,1)
x = df2[['Age', 'Parch', 'SibSp', 'Sex']].values


# Find the best model

# In[ ]:


hyperparam = {
    'max_depth': [2,3,5,7,11,13,15,20,30],
    'min_samples_leaf': [10,15,20,25,30,40,50],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt'],
    'n_estimators': [2,3,4],
    #'bootstrap': [True, False]
}

opti = RandomizedSearchCV(
    estimator = ensemble.RandomForestClassifier(), 
    param_distributions = hyperparam, 
    cv = 30,
    n_jobs = -1,
    n_iter = 100,
    random_state = 42,
    return_train_score = True)

opti.fit(X=x, y=y)

print(opti.best_params_)

print(opti.best_score_)


# Display the best model

# In[ ]:


bestmodel = opti.best_estimator_
bestmodel.fit(X=x, y=y)

print( accuracy_score(y, bestmodel.predict(x)) )


# In[ ]:


bestmodel.feature_importances_


# ## Predict and Submit
# 

# In[ ]:


df = pd.read_csv('../input/test.csv')

df['Sex'] = df['Sex'].apply(lambda e : 1 if e=='male' else (2 if e=='female' else None))
df['Age'] = df['Age'].apply(lambda e : 5 if pd.isnull(e) else (1 if e<7 else (2 if e<20 else (3 if e<37 else 4))))

x = df[['Age', 'Parch', 'SibSp', 'Sex']].values


# In[ ]:


yhat = bestmodel.predict(x)


# In[ ]:


result = pd.DataFrame(columns=['PassengerId', 'Survived'], index=df.index)
result['PassengerId'] = df['PassengerId']
result['Survived'] = yhat
result.to_csv('randomforest-age-parch-sibsp-sex.csv', index=False)

