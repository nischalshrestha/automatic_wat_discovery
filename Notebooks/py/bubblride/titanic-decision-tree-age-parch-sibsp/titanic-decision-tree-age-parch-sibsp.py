#!/usr/bin/env python
# coding: utf-8

# ## Load Modules

# In[ ]:


import pandas as pd
from sklearn import tree
import graphviz 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# ## Summary
# * sklearn's Decision Tree classifier
# * some hyperparameter optimization, `k=30` fold CV
# * 3 levels, 1 level for each variable

# ## Age + Parch + SibSp (Family Structure)
# * Check `Age`, `Parch`, `SibSp` 
# * It's kind of the family structure

# In[ ]:


df = pd.read_csv("../input/train.csv")
df2 = df.dropna(subset=['Age', 'Parch', 'SibSp'])
y = df2['Survived'].values.reshape(-1,1)
x = df2[['Age', 'Parch', 'SibSp']].values


# Find the best model

# In[ ]:


hyperparam = {
    'min_samples_leaf':range(20,60)
}

opti = GridSearchCV(
    estimator=tree.DecisionTreeClassifier(max_depth=3), 
    param_grid=hyperparam, 
    cv=30,
    n_jobs=2,
    return_train_score=True)

opti.fit(X=x, y=y)

print(opti.best_params_)

print(opti.best_score_)


# Display the best model

# In[ ]:


bestmodel = opti.best_estimator_
bestmodel.fit(X=x, y=y)

print( accuracy_score(y, bestmodel.predict(x)) )

dot_data = tree.export_graphviz(
    bestmodel, 
    out_file=None,  
    feature_names=['Age', 'Parch', 'SibSp'],
    class_names=['Dead', 'Alive'],
    filled=True, 
    rounded=True,
    special_characters=True,
    proportion=True)
graph = graphviz.Source(dot_data)

graph


# ## Predict and Submit
# 

# In[ ]:


df = pd.read_csv('../input/test.csv')
idx_ok = df[['Age', 'Parch', 'SibSp']].notnull().all(axis=1)
x = df.loc[idx_ok, ['Age', 'Parch', 'SibSp']].values


# In[ ]:


yhat = bestmodel.predict(x)


# In[ ]:


result = pd.DataFrame(columns=['PassengerId', 'Survived'], index=df.index)
result['PassengerId'] = df['PassengerId']
result['Survived'] = 0 #Default prediction is death
result.loc[idx_ok, 'Survived'] = yhat
result.to_csv('decisiontree-age-parch-sibsp.csv', index=False)


# In[ ]:




