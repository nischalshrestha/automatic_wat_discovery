#!/usr/bin/env python
# coding: utf-8

# ## Summary
# * Random Forest classification, tinkered with hyperparams till i found somethine
# * `Name` data converted to phonetic codes with [metaphone](https://pypi.org/project/Metaphone/)
# * these phonetic codes are counted and cluster with KMean (8 cluster)
# 
# 
# ## Load Modules

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# ## Load Data

# In[ ]:


df = pd.read_csv("../input/train.csv")


# ## Prep Data
# 

# In[ ]:


#example: word_to_phonetic(['Futrelle, Mrs. Jacques Heath (Lily May Peel)'])
def word_to_phonetic(arr):
    from metaphone import doublemetaphone as dmp
    out = list()
    for e in arr:
        p = list()
        for s in e.split():
            p += dmp(s)
        p = [s for s in p if s]
        out.append(' '.join(p))
    return out

def cluster_fit(arr):
    #from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer 
    from sklearn.cluster import KMeans
    
    # word counter
    arr2 = word_to_phonetic(arr)
    #vec = TfidfVectorizer().fit(arr2)
    vec = CountVectorizer().fit(arr2)
    X = vec.transform(arr2) #bug fixed
    # cluster 
    clu = KMeans(n_clusters=8).fit(X)
    res = clu.labels_
    return res, clu, vec

def cluster_predict(clu, vec, arr):
    arr2 = word_to_phonetic(arr)
    X = vec.transform(arr2)
    return clu.predict(X)


# In[ ]:


def transform(df0, clu=None, vec=None):
    df = pd.DataFrame(index=df0.index)
    
    # Pclass - use the 3 labels and missing (999)
    df['Pclass'] = df0['Pclass'].apply(lambda e: 999 if pd.isnull(e) else e)
    
    # Name Clusters
    if clu:
        df['NameCluster'] = cluster_predict(clu, vec, df0['Name'])
    else:
        tmp, clu, vec = cluster_fit(df0['Name'])
        df['NameCluster'] = tmp
    
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


hyperparam = {
    #'criterion': ['gini', 'entropy']
    #'min_impurity_decrease': [.01, .02, .03], 
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
        min_impurity_decrease=0.0, #split dep on impurity (use with entropy)
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
df2, _, _ = transform(df, clu, vec)
x = df2.values


# In[ ]:


yhat = bestmodel.predict(x)


# In[ ]:


result = pd.DataFrame(columns=['PassengerId', 'Survived'], index=df.index)
result['PassengerId'] = df['PassengerId']
result['Survived'] = yhat
result.to_csv('randomforest-phonetic4.csv', index=False)

