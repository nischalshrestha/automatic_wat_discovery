#!/usr/bin/env python
# coding: utf-8

# ## Summary
# *  Like [this](https://www.kaggle.com/bubblride/titanic-random-forest-5-better-titles) but with XGBoost
# 
# 
# ## Load Modules

# In[ ]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


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
    
    # Pclass - use the 3 labels and missing (999)
    df['Pclass'] = df0['Pclass'].apply(lambda e: 999 if pd.isnull(e) else e)
    
    # Title - 1:"Mr", 2:"Mrs", 3:"Miss", 4:"Master", 5:"Officer", 6:"Royalty"
    df['Title'] = df0['Name'].apply(label_titles)
    
    # Gender - male, female, missing (99)
    df['Sex'] = 999
    df.loc[df0['Sex']=='male', 'Sex'] = 2
    df.loc[df0['Sex']=='female', 'Sex'] = 1
    
    # Age - number or missing (999)
    df['Age'] = df0['Age'].apply(lambda e: 999 if pd.isnull(e) else e)
    
    # SibSp - number or missing (999)
    df['SibSp'] = df0['SibSp'].apply(lambda e: 999 if pd.isnull(e) else e)

    # Parch - number or missing (999)
    df['Parch'] = df0['Parch'].apply(lambda e: 999 if pd.isnull(e) else e)
    
    # Ticket - Is numbered ticket (True) or contain strings (False)
    #df['Ticket'] = [int(str.isnumeric(e)) for e in df0.Ticket]
    
    # Fare - keep the numeric values or missing (-999)
    df['Fare'] = df0['Fare'].apply(lambda e: -999 if pd.isnull(e) else e)
    
    # Cabin - 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T', missing (999)
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
    
    #df['Section1'] = np.trunc(pd.to_numeric(tmp['Section']).values / 10)
    #df.loc[df['Section1'].isnull(), 'Section1'] = 999
    #df['Section2'] = pd.to_numeric(tmp['Section']).apply(lambda n: n % 2 != 0)
    #df.loc[df['Section2'].isnull(), 'Section2'] = 999
    
    # Embarked - S, C, Q, missing (999)
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


# Find the best model

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


hyperparam = {
    #'learning_rate': (np.exp(0.05 * np.arange(1, 7)) - 1).round(2),
    #'max_depth': np.arange(3, 29, 4),
    #'n_estimators': [100,150,200],
    #'objective': ['binary:logistic'],
    #'booster': ['gbtree', 'dart'],
    #'gamma': np.arange(0, .2, .02),
    #'min_child_weight': np.arange(0.8, 1.5, .1),
    #'max_delta_step': [.01, .02],
    #'subsample': np.arange(.5, .91, .1),
    #'colsample_bytree': np.arange(.5, .91, .1),
    #'reg_alpha': np.arange(0, 1.11, .1),
    #'reg_lambda': np.arange(0, 1.11, .1)
}

opti = GridSearchCV(
    estimator = XGBClassifier(
        learning_rate=0.11, #eta learning rate (0.3)
        max_depth=9, #max num of levels (9)
        n_estimators=50,  #number of trees
        objective='binary:logistic',  #type of target func
        booster='gbtree', #type of model
        gamma=0.12,  #minimum loss reduction on a leaf (0.0)
        min_child_weight=1.25, #min sum of wgt per child (1.0), set >1 to underfit
        max_delta_step=0.1, #set >0 for more conservative weight updates
        subsample=.5, #pct of obs part of random subsamples (1.0)
        colsample_bytree=.6, #max pct of features used in sub-trees (1.0)
        colsample_bylevel=1.0, #not necessary if you use subsample
        reg_alpha=0.8, #L1 regulization param (0.0)
        reg_lambda=0.1, #L2 regulization param (0.1)
        scale_pos_weight=1, #balance positive and negative weights
        base_score=0.5, #start values
        random_state=24, #for resampling
        n_jobs=-1,
        silent=True
    ), 
    param_grid = hyperparam, 
    cv = 25,
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
df2 = transform(df)
x = df2.values


# In[ ]:


yhat = bestmodel.predict(x)


# In[ ]:


result = pd.DataFrame(columns=['PassengerId', 'Survived'], index=df.index)
result['PassengerId'] = df['PassengerId']
result['Survived'] = yhat
result.to_csv('xgboost-5-titles.csv', index=False)

