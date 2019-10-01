#!/usr/bin/env python
# coding: utf-8

# ## Summary
# * fork of the [Random Forest](https://www.kaggle.com/bubblride/titanic-random-forest-4-phonetic) version
# * insteadt XGBoost is used
# * `Name` data is ignored, see [reasons here](https://www.kaggle.com/bubblride/titanic-feature-from-name)
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


def transform(df0, clu=None, vec=None):
    df = pd.DataFrame(index=df0.index)
    
    # Pclass - use the 3 labels and missing (999)
    df['Pclass'] = df0['Pclass'].apply(lambda e: 999 if pd.isnull(e) else e)
    
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
    df['Ticket'] = [int(str.isnumeric(e)) for e in df0.Ticket]
    
    # Fare - keep the numeric values or missing (-999)
    df['Fare'] = df0['Fare'].apply(lambda e: -999 if pd.isnull(e) else e)
    
    # Cabin - 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T', missing (999)
    tmp = pd.DataFrame(df0['Cabin'])
    tmp.loc[df0['Cabin'].isnull(), 'Cabin'] = '?9999'
    tmp['Deck'] = [e[0] for e in tmp['Cabin']]
    tmp['Section'] = [e[1:].split(' ')[0] for e in tmp['Cabin']]
    
    df['Deck'] = 999
    df.loc[tmp['Deck']=='A', 'Deck'] = 1 
    df.loc[tmp['Deck']=='B', 'Deck'] = 2 
    df.loc[tmp['Deck']=='C', 'Deck'] = 3 
    df.loc[tmp['Deck']=='D', 'Deck'] = 4 
    df.loc[tmp['Deck']=='E', 'Deck'] = 5 
    df.loc[tmp['Deck']=='F', 'Deck'] = 6 
    df.loc[tmp['Deck']=='G', 'Deck'] = 7 
    df.loc[tmp['Deck']=='T', 'Deck'] = 8
    
    df['Section'] = np.trunc(pd.to_numeric(tmp['Section']).values / 10)
    df.loc[df['Section'].isnull(), 'Section'] = 999
    
    # Embarked - S, C, Q, missing (999)
    df['Embarked'] = 999
    df.loc[df0['Embarked']=='S', 'Embarked'] = 1 
    df.loc[df0['Embarked']=='C', 'Embarked'] = 2
    df.loc[df0['Embarked']=='Q', 'Embarked'] = 3
    
    # done
    return df, clu, vec


# In[ ]:


y = df['Survived'].values 

df2, clu, vec = transform(df)
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
    #'max_delta_step': [0, .1, .2],
    #'subsample': np.arange(.5, .91, .1),
    #'colsample_bytree': np.arange(.5, .91, .1),
    #'reg_alpha': np.arange(0, 1.11, .1),
    #'reg_lambda': np.arange(0, 1.11, .1)
}

opti = GridSearchCV(
    estimator = XGBClassifier(
        learning_rate=0.11, #eta learning rate (0.3)
        max_depth=7, #max num of levels (9)
        n_estimators=100,  #number of trees
        objective='binary:logistic',  #type of target func
        booster='gbtree', #type of model
        gamma=0.12,  #minimum loss reduction on a leaf (0.0)
        min_child_weight=1.25, #min sum of wgt per child (1.0), set >1 to underfit
        max_delta_step=0, #set >0 for more conservative weight updates
        subsample=.5, #pct of obs part of random subsamples (1.0)
        colsample_bytree=.7, #max pct of features used in sub-trees (1.0)
        colsample_bylevel=1.0, #not necessary if you use subsample
        reg_alpha=0.7, #L1 regulization param (0.0)
        reg_lambda=1.0, #L2 regulization param (0.1)
        scale_pos_weight=1, #balance positive and negative weights
        base_score=0.5, #start values
        random_state=23, #for resampling
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
df2, _, _ = transform(df, clu, vec)
x = df2.values


# In[ ]:


yhat = bestmodel.predict(x)


# In[ ]:


result = pd.DataFrame(columns=['PassengerId', 'Survived'], index=df.index)
result['PassengerId'] = df['PassengerId']
result['Survived'] = yhat
result.to_csv('xgboost-noname.csv', index=False)

