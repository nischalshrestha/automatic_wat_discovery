#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn import tree
import graphviz 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import os
print(os.listdir("../input"))


# ## Summary
# It is EDA and not a submission. Accurancy is rather low but the tree diagram might make things clear.
# 
# #### Methods
# 
# * Decision Tree classification (with CV resampling)
# * Check `Age` and `Parch`  - Does age matter if a person has parents or children?
# 
# #### Results
# 
# * No kids, no parents => Dead
# * Has kids or parents
#     * Under 7 => Survive (Child with parent)
#     * 7 to 19 => Dead
#     * 20 to 36 => Survive (Parent with child)
#     * Older than 36 => Dead
#   
# #### Interpretation
# This very much the **"Women and children first"** drill.

# ## Load Data

# In[ ]:


df = pd.read_csv("../input/train.csv")
#df.head(6)


# ## Age + Parch
# * Drop examples with NA in `Age` or `Parch`
# * `k=30` fold CV
# * `max_depth` and `min_samples_leaf` hyperparameters
# 
# Findings
# 
# * The number of parents or children does not matter
# * If a passenger have at least 1 parent/child, then age matters
# * If a passenger have no parents nor kids on board, the age does not matter (passenger is more likely to be dead)
# 

# ### Data Prep

# In[ ]:


df1 = df.dropna(subset=['Age', 'Parch'])
y = df1['Survived'].values.reshape(-1,1)
x = df1[['Age', 'Parch']].values


# ### Find the Best Tree

# In[ ]:


hyperparam = {
    #'criterion': ['gini', 'entropy'],
    #'splitter': ['best', 'random'],
    #'class_weight': ['balanced', {0: 0.5, 1: 0.5}, {0: 0.8, 1: 0.2}],
    'max_depth': [2,5,10,15], 
    'min_samples_leaf':range(20,60)
}

opti = GridSearchCV(
    estimator=tree.DecisionTreeClassifier(), 
    param_grid=hyperparam, 
    cv=30,
    n_jobs=2,
    return_train_score=True)

opti.fit(X=x, y=y)

print(opti.best_params_)

print(opti.best_score_)


# ### Display the Best Tree

# In[ ]:


bestmodel = opti.best_estimator_
bestmodel.fit(X=x, y=y)

print( accuracy_score(y, bestmodel.predict(x)) )

dot_data = tree.export_graphviz(
    bestmodel, 
    out_file=None,  
    feature_names=['Age', 'Parch'],
    class_names=['Dead', 'Alive'],
    filled=True, 
    rounded=True,
    special_characters=True)
graph = graphviz.Source(dot_data)

graph


# ## Simplified Tree
# * Remove examples with NA `Age` entries
# * Remove examples with passengers without parents nor children

# In[ ]:


df2 = df.dropna(subset=['Age'])
df2 = df2[df2['Parch'] > 0]
y = df2['Survived'].values.reshape(-1,1)
x = df2['Age'].values.reshape(-1,1)


# In[ ]:


hyperparam = {
    'max_depth': range(2,10), 
    'min_samples_leaf':range(20,60)
}

opti = GridSearchCV(
    estimator=tree.DecisionTreeClassifier(), 
    param_grid=hyperparam, 
    cv=30,
    n_jobs=2,
    return_train_score=True)

opti.fit(X=x, y=y)

print(opti.best_params_)

print(opti.best_score_)


# In[ ]:


bestmodel = opti.best_estimator_
bestmodel.fit(X=x, y=y)

print( accuracy_score(y, bestmodel.predict(x)) )

dot_data = tree.export_graphviz(
    bestmodel, 
    out_file=None,  
    feature_names=['Age'],
    class_names=['Dead', 'Alive'],
    filled=True, 
    rounded=True,
    special_characters=True)
graph = graphviz.Source(dot_data)

graph


# ## Submit
# Let's submit the simplified tree prediction

# In[ ]:


df = pd.read_csv('../input/test.csv')

# Flag missing age as False, flag Parch=0 as False
idx_ok = df['Age'].notnull() & (df['Parch'] > 0)
#idx_ok

# Slice the eligible Age values
x = df.loc[idx_ok, ['Age']].values
#x


# In[ ]:


yhat = bestmodel.predict(x)


# In[ ]:


result = pd.DataFrame(columns=['PassengerId', 'Survived'], index=df.index)
result['PassengerId'] = df['PassengerId']
result.loc[:, 'Survived'] = 0
result.loc[idx_ok, 'Survived'] = yhat
result.head()


# In[ ]:


result.to_csv('decisiontree-age-parch.csv', index = False)

