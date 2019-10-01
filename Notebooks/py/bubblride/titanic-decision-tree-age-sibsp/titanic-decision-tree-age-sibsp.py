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
# 
# #### Methods
# 
# * Decision Tree classification (with CV resampling)
# * Check `Age` and `SibSp`  
# 

# ## Load Data

# In[ ]:


df = pd.read_csv("../input/train.csv")
#df.head(6)


# ## Age + SibSp
# * Drop examples with NA in `Age` or `SibSp`
# * `k=30` fold CV
# * `max_depth` and `min_samples_leaf` hyperparameters

# ### Data Prep

# In[ ]:


df1 = df.dropna(subset=['Age', 'SibSp'])
y = df1['Survived'].values.reshape(-1,1)
x = df1[['Age', 'SibSp']].values


# ### Find the Best Tree

# In[ ]:


hyperparam = {
    'max_depth': [2,3,4,5,6,10,15], 
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
    feature_names=['Age', 'SibSp'],
    class_names=['Dead', 'Alive'],
    filled=True, 
    rounded=True,
    special_characters=True,
    proportion=True)

graph = graphviz.Source(dot_data)

graph


# ## Findings
# 
# #### Survivors without siblings/spouse
# 
# * **Babys and small children**: Small children under 7 have good chances to survive. Siblings does not matter -- I guess most adults (relatives or not) would just grab a baby or small child and place them in rescue boat.
# * **The single child**: 7 to 16 years old children without siblings -- I guess they were strong enough to reach the rescue boats on their own. They were not obliged to help younger siblings. I would also speculate (bcoz i didn't check the interaction with `Parch`) that the parents put extra effort into rescuing their only child.
# * **Young Singles without Siblings**: Passengers between 17 and 27 without siblings nor spouse. One can assume that this age group is in good physical shape. I speculate  (I didn't check `Parch`) that these people boarded the ship alone and had no obligation to rescue family members.
# 
# #### Survivors with 1 sibling/spouse
# * Under 25 -- Young people who are not alone
# * Between 35 and 39 -- Life experienced with some sort of authority
# * Older than 48  -- Elderly
# 
# #### 2 or more siblings including a spouse? => Dead
# * I guess it is very difficult to move quickly with group 3 or more people during an emergency situation
# * Unclear Leadership: Decision-making (e.g. "turn left or right?") could be complex
# * Loosing time: The whole group need to stop, if one member falls behind
# * Worry about too many people: A passenger might get lost in the hopeless endavour to find all his family member although they are alread lost
# 

# ### Submit
# * Use the fitted Decision Tree 
# * Any example with missing data (e.g. Age is not reported) are assumed to be dead

# In[ ]:


df = pd.read_csv('../input/test.csv')

# Index the complete examples
idx_ok = df[['Age', 'SibSp']].notnull().all(axis=1)
#idx_ok

# Slice only the complete example
x = df.loc[idx_ok, ['Age', 'SibSp']].values
# x


# In[ ]:


df.index


# Predict the label for the complete examples

# In[ ]:


yhat = bestmodel.predict(x)


# Prep the CSV export. 

# In[ ]:


result = pd.DataFrame(columns=['PassengerId', 'Survived'], index=df.index)
result['PassengerId'] = df['PassengerId']
result.loc[:, 'Survived'] = 0
result.loc[idx_ok, 'Survived'] = yhat
result.head()


# In[ ]:


result.to_csv('decisiontree-age-sibsp.csv', index = False)

