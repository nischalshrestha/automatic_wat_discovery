#!/usr/bin/env python
# coding: utf-8

# ## Load Modules

# In[ ]:


import pandas as pd
from sklearn import tree
from sklearn import ensemble
import graphviz 
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# ## Summary
# * sklearn's Random Forest classifier
# * some hyperparameter optimization with `RandomizedSearchCV`, `k=30` fold CV
# * Add more variables

# ## Load Data

# In[ ]:


df = pd.read_csv("../input/train.csv")


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# ## Prep Data
# * Pclass - 3 labels
# * Name - Look for `mr.` (1), `mrs.` (2), `miss.` (3), or missing (4)
# * Sex - `male` (1), `female` (2), or missing (3)
# * Age - binned in 4 age groups plus 1 for missing data
# * SibSp - Look for 0, 1, 2 or more
# * Parch - Yes or no
# * Ticket - Is numbered ticket (True) or contain strings (False)
# * Fare 
# * Cabin - Yes or No
# * Embarked - S, C, Q, else

# In[ ]:


#pd.unique(df2.Ticket)
#[str.isnumeric(e) for e in df2.Ticket]
#pd.notnull(df2.Cabin)
#pd.unique(df2.Embarked)

#import re
#pat1 = re.compile(r'mr\.', flags=re.I)  #re.I=Ignore Case
#pat2 = re.compile(r'mrs\.', flags=re.I)  
#pat3 = re.compile(r'miss\.', flags=re.I)
#df.Name.apply(lambda s: 1 if bool(pat1.search(s)) else (2 if bool(pat2.search(s)) else (3 if bool(pat3.search(s)) else -1))  )


# In[ ]:


def label_titles(s):
    import re
    pat1 = re.compile(r'mr\.', flags=re.I)  #re.I=Ignore Case
    if bool(pat1.search(s)):
        return 1
    
    pat2 = re.compile(r'mrs\.', flags=re.I) 
    if bool(pat2.search(s)):
        return 2
    
    pat3 = re.compile(r'miss\.', flags=re.I)
    if bool(pat3.search(s)):
        return 3
    
    return -1


# In[ ]:


def transform(df):
    #df['Pclass']
    df['Title'] = df['Name'].apply(label_titles)
    df['Sex'] = df['Sex'].apply(lambda e : 1 if e=='male' else (2 if e=='female' else -1))
    #df['Age'] = df['Age'].apply(lambda e : -1 if pd.isnull(e) else (1 if e<7 else (2 if e<20 else (3 if e<37 else 4))))
    df['Age'] = df['Age'].apply(lambda e: -1 if pd.isnull(e) else e)
    df['SibSp'] = df['SibSp'].apply(lambda e: 2 if e>2 else e)
    df['Parch'] = df['Parch'].apply(lambda e: 1 if e>0 else 0)
    df['Ticket'] = [str.isnumeric(e) for e in df.Ticket]
    #df['Fare']
    df['Cabin'] = pd.notnull(df.Cabin)
    df['Embarked'] = df['Embarked'].apply(lambda e: -1 if pd.isnull(e) else (1 if e=='S' else (2 if e=='C' else 3)))
    return df


# In[ ]:


df2 = df.copy()
df2 = transform(df2)
df2.head()


# In[ ]:


y = df2['Survived'].values #.reshape(-1,1)
x = df2[['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']].values
len(x[0])


# Find the best model

# In[ ]:


hyperparam = {
    'n_estimators': range(50, 121, 10),
    'max_depth': range(3, 2*9+1, 3),
    'min_samples_leaf': range(10, 21, 5),
    'min_samples_split': range(20, 41, 10),
    #'max_features': ['sqrt', 'auto'],
    #'bootstrap': [True, False] 
}

opti = GridSearchCV(
    estimator = ensemble.RandomForestClassifier(max_features='sqrt', bootstrap=True), 
    param_grid = hyperparam, 
    cv = 30,
    n_jobs = -1,
    return_train_score = True)

"""
opti = RandomizedSearchCV(
    estimator = ensemble.RandomForestClassifier(max_features='sqrt', bootstrap=True), 
    param_distributions = hyperparam, 
    cv = 30,
    n_jobs = -1,
    n_iter = 100,
    random_state = 42,
    return_train_score = True)
"""

opti.fit(X=x, y=y)

print(opti.best_params_)

print(opti.best_score_)


# Display the best model

# In[ ]:


bestmodel = opti.best_estimator_
#bestmodel.fit(X=x, y=y)

print( accuracy_score(y, bestmodel.predict(x)) )


# In[ ]:


bestmodel.feature_importances_


# ## Predict and Submit
# 

# In[ ]:


df = pd.read_csv('../input/test.csv')
df = transform(df)
x = df[['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']].values


# In[ ]:


yhat = bestmodel.predict(x)


# In[ ]:


result = pd.DataFrame(columns=['PassengerId', 'Survived'], index=df.index)
result['PassengerId'] = df['PassengerId']
result['Survived'] = yhat
result.to_csv('randomforest-many-variable2.csv', index=False)

