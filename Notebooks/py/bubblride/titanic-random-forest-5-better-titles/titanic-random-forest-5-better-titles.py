#!/usr/bin/env python
# coding: utf-8

# ## Load Modules

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import ensemble
import graphviz 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# ## Summary
# * sklearn's Random Forest classifier
# * some hyperparameter optimization with `RandomizedSearchCV`, `k=30` fold CV
# * Add more variables
# * add more titles, see [Chapter 3.1. by user vincentlugat](https://www.kaggle.com/vincentlugat/titanic-data-analysis-rf-prediction-0-81818)

# ## Load Data

# In[ ]:


df = pd.read_csv("../input/train.csv")


# ## Prep Data
# 

# In[ ]:


def label_titles(s):
    import re

    for rgx in [r'dona\.', r'lady\.', r'Countess\.', r'sir\.', r'jonkheer\.']:
        pat = re.compile(rgx, flags=re.I)
        if bool(pat.search(s)): 
            return 6

    for rgx in [r'capt\.', r'col\.', r'don\.', r'dr\.', r'major\.', r'rev\.']:
        pat = re.compile(rgx, flags=re.I)
        if bool(pat.search(s)): 
            return 5

    pat4 = re.compile(r'master\.', flags=re.I) 
    if bool(pat4.search(s)):
        return 4

    for rgx in [r'miss\.', r'ms\.', r'mlle\.']:
        pat = re.compile(rgx, flags=re.I)
        if bool(pat.search(s)): 
            return 3

    pat2a = re.compile(r'mrs\.', flags=re.I) 
    pat2b = re.compile(r'mme\.', flags=re.I) 
    if bool(pat2a.search(s)): return 2
    if bool(pat2b.search(s)): return 2

    pat1 = re.compile(r'mr\.', flags=re.I)  #re.I=Ignore Case
    if bool(pat1.search(s)):
        return 1

    return 999


# In[ ]:


def transform(df0):
    df = pd.DataFrame(index=df0.index)
    
    # Pclass - use the 3 labels
    df['Pclass'] = df0['Pclass'].apply(lambda e: 999 if pd.isnull(e) else e)
    
    # Title - 1:"Mr", 2:"Mrs", 3:"Miss", 4:"Master", 5:"Officer", 6:"Royalty"
    df['Title'] = df0['Name'].apply(label_titles)
    
    # Gender - male, female, missing
    df['Sex'] = 999
    df.loc[df0['Sex']=='male', 'Sex'] = 2 
    df.loc[df0['Sex']=='female', 'Sex'] = 1
    
    # Age - number or missing
    df['Age'] = df0['Age'].apply(lambda e: 999 if pd.isnull(e) else e)
    
    # SibSp - number or missing
    df['SibSp'] = df0['SibSp'].apply(lambda e: 999 if pd.isnull(e) else e)

    # Parch - number or missing
    df['Parch'] = df0['Parch'].apply(lambda e: 999 if pd.isnull(e) else e)
    
    # Ticket - Is numbered ticket (True) or contain strings (False)
    #df['Ticket'] = [int(str.isnumeric(e)) for e in df0.Ticket]
    
    # Fare - keep the numeric values
    df['Fare'] = df0['Fare'].apply(lambda e: 999 if pd.isnull(e) else e)
    
    # Cabin - 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T', missing (-1)
    tmp = pd.DataFrame(df0['Cabin'])
    tmp.loc[df0['Cabin'].isnull(), 'Cabin'] = '?9999'
    tmp['Deck'] = [e[0] for e in tmp['Cabin']]
    #tmp['Section'] = [e[1:].split(' ')[0] for e in tmp['Cabin']]
    
    df['Deck'] = 999
    df.loc[tmp['Deck']=='A', 'Deck'] = 1 
    df.loc[tmp['Deck']=='B', 'Deck'] = 2 
    df.loc[tmp['Deck']=='C', 'Deck'] = 3 
    df.loc[tmp['Deck']=='D', 'Deck'] = 4 
    df.loc[tmp['Deck']=='E', 'Deck'] = 5 
    df.loc[tmp['Deck']=='F', 'Deck'] = 6 
    df.loc[tmp['Deck']=='G', 'Deck'] = 7 
    df.loc[tmp['Deck']=='T', 'Deck'] = 8
    
    #df['Section'] = np.trunc(pd.to_numeric(tmp['Section']).values / 10)
    #df['Section'] = pd.to_numeric(tmp['Section']).apply(lambda n: n % 2 != 0)
    #df.loc[df['Section'].isnull(), 'Section'] = 999
    
    # Embarked - S, C, Q, missing (-1)
    df['Embarked'] = 999
    df.loc[df0['Embarked']=='S', 'Embarked'] = 1 
    df.loc[df0['Embarked']=='C', 'Embarked'] = 2
    df.loc[df0['Embarked']=='Q', 'Embarked'] = 3
    
    # done
    return df


# In[ ]:


y = df['Survived'].values 
df2 = transform(df)
x = df2.values


# In[ ]:


#df[df2['Title']==5]


# 
# Find the best model

# In[ ]:


hyperparam = {
    #'criterion': ['gini', 'entropy']
    #'min_impurity_decrease': [0, .01, .02, .03], 
    #'max_features': ['sqrt', 'log2'],
    #'n_estimators': range(30,91,10),
    #'max_depth': range(6,30), 
    #'min_samples_leaf': range(2,6),  
    #'min_samples_split': range(4,13,2), 
    #'max_leaf_nodes': [4, 8, 16, 32],
    #'oob_score': [True, False],
    #'bootstrap': [True, False] 
    #'warm_start': [True, False]
}

opti = GridSearchCV(
    estimator = ensemble.RandomForestClassifier(
        criterion='gini',
        min_impurity_decrease=0.01, #split dep on impurity (0.0)
        max_features='sqrt',
        n_estimators=150,  #number of sub-tres
        max_depth=9,  #max depth of all trees
        min_samples_leaf=2, #smallest leaf size
        min_samples_split=4, #smallest size at split
        max_leaf_nodes=None,  #max number of leafs in a tree
        oob_score=True, #Uses n-1 obs for validation
        bootstrap=True, 
        random_state=23, #for bootstrap=True
        warm_start=False, #reuse previous tree fits
        class_weight=None, #weight for multi-output Ys 
        verbose=0 #logging level
    ), 
    param_grid = hyperparam, 
    cv = 10,
    n_jobs = -1,
    return_train_score = True)

opti.fit(X=x, y=y)

print(opti.best_estimator_)

print(opti.best_params_)

print(opti.best_score_)


# Display the best model

# In[ ]:


bestmodel = opti.best_estimator_
bestmodel.fit(X=x, y=y)

print( accuracy_score(y, bestmodel.predict(x)) )


# In[ ]:


pd.DataFrame((100*bestmodel.feature_importances_).round(1), 
             index=df2.columns, 
             columns=['Importance']).sort_values(by='Importance', ascending=False)


# ## Predict and Submit
# 

# In[ ]:


df = pd.read_csv('../input/test.csv')
x = transform(df).values


# In[ ]:


yhat = bestmodel.predict(x)


# In[ ]:


result = pd.DataFrame(columns=['PassengerId', 'Survived'], index=df.index)
result['PassengerId'] = df['PassengerId']
result['Survived'] = yhat
result.to_csv('randomforest-5-titles.csv', index=False)

