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

# ## Load Data

# In[ ]:


df = pd.read_csv("../input/train.csv")


# In[ ]:


#df.head()
#df.isnull().sum()


# ## Prep Data
# 

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


def transform(df0):
    df = pd.DataFrame(index=df0.index)
    
    # Pclass - use the 3 labels
    df['Pclass'] = df0['Pclass'].apply(lambda e: -1 if pd.isnull(e) else e)
    
    # Title - Look for "mr.", "mrs.", "miss.", missing (-1)
    df['Title'] = df0['Name'].apply(label_titles)
    
    # Gender - male, female, missing (-1)
    df['Sex'] = -1
    df.loc[df0['Sex']=='male', 'Sex'] = 1 
    df.loc[df0['Sex']=='female', 'Sex'] = 2
    
    # Age - number or missing (-1)
    df['Age'] = df0['Age'].apply(lambda e: -1 if pd.isnull(e) else e)
    
    # SibSp - number or missing (-1)
    df['SibSp'] = df0['SibSp'].apply(lambda e: -1 if pd.isnull(e) else e)

    # Parch - number or missing (-1)
    df['Parch'] = df0['Parch'].apply(lambda e: -1 if pd.isnull(e) else e)
    
    # Ticket - Is numbered ticket (True) or contain strings (False)
    df['Ticket'] = [int(str.isnumeric(e)) for e in df0.Ticket]
    
    # Fare - keep the numeric values
    df['Fare'] = df0['Fare'].apply(lambda e: -1 if pd.isnull(e) else e)
    
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
    #df.loc[df['Section'].isnull(), 'Section'] = 999
    
    # Embarked - S, C, Q, missing (-1)
    df['Embarked'] = -1 
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


hyperparam = {
    'n_estimators': [30], #range(50, 121, 10),
    'max_depth': [6,9,12,20,30], #range(3, 2*9+1, 3),
    #'min_samples_leaf': [10], #range(10, 21, 5),
    #'min_samples_split': [20], #range(20, 41, 10),
    #'max_features': ['sqrt', 'auto'],
    #'bootstrap': [True, False] 
}

opti = GridSearchCV(
    estimator = ensemble.RandomForestClassifier(), 
    param_grid = hyperparam, 
    cv = 30,
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
result.to_csv('randomforest-many-variable3-debugged.csv', index=False)

