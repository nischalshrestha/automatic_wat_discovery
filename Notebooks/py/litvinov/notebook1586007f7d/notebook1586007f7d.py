#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd


# In[ ]:


test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df = pd.read_csv('../input/train.csv')


# In[ ]:


combine = [train_df, test_df]


# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


train_df[["Fare", "Survived"]].groupby(pd.cut(train_df['Fare'], bins=15, labels=False)).mean().sort_values(by='Fare', ascending=False)


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[ ]:


train_df[["Survived"]].groupby(lambda ind: 'NaN' if (pd.isnull (train_df['Cabin'].loc[ind])) else train_df['Cabin'].loc[ind][0], as_index=True, sort=False, group_keys=True).mean()


# In[ ]:


train_df[["PassengerId"]].groupby(lambda ind: 'NaN' 
                                     if (pd.isnull (train_df['Cabin'].loc[ind])) 
                                     else train_df['Cabin'].loc[ind][0]).count()


# In[ ]:


train_df[["Fare"]].groupby(lambda ind: 'NaN' if (pd.isnull (train_df['Cabin'].loc[ind])) else train_df['Cabin'].loc[ind][0]).mean()


# In[ ]:


train_df.loc[train_df['Fare'] > 500]


# In[ ]:


import re

train_df[["Survived"]].groupby(lambda ind: 'STON' 
                               if re.search('STON', train_df['Ticket'].loc[ind]) 
                               else 'PC' if re.search('P.?C', train_df['Ticket'].loc[ind])
                               else 'SC' if re.search('S.?C?', train_df['Ticket'].loc[ind])
                               else 'PP' if re.search('P.?P', train_df['Ticket'].loc[ind]) 
                               else 'CA' if re.search('C.?A?', train_df['Ticket'].loc[ind]) 
                               else 'A/' if re.search('A.?/', train_df['Ticket'].loc[ind]) 
                               else 'letter' if re.search('[a-zA-Z]+', train_df['Ticket'].loc[ind]) 
                               else 'number').mean()


# In[ ]:


train_df[["Fare"]].groupby(lambda ind: 'STON' 
                               if re.search('STON', train_df['Ticket'].loc[ind]) 
                               else 'PC' if re.search('P.?C', train_df['Ticket'].loc[ind])
                               else 'SC' if re.search('S.?C?', train_df['Ticket'].loc[ind])
                               else 'PP' if re.search('P.?P', train_df['Ticket'].loc[ind]) 
                               else 'CA' if re.search('C.?A?', train_df['Ticket'].loc[ind]) 
                               else 'A/' if re.search('A.?/', train_df['Ticket'].loc[ind]) 
                               else 'letter' if re.search('[a-zA-Z]+', train_df['Ticket'].loc[ind]) 
                               else 'number').mean ()


# In[ ]:


train_df[["Fare"]].groupby(lambda ind: 'STON' 
                               if re.search('STON', train_df['Ticket'].loc[ind]) 
                               else 'PC' if re.search('P.?C', train_df['Ticket'].loc[ind])
                               else 'SC' if re.search('S.?C?', train_df['Ticket'].loc[ind])
                               else 'PP' if re.search('P.?P', train_df['Ticket'].loc[ind]) 
                               else 'CA' if re.search('C.?A?', train_df['Ticket'].loc[ind]) 
                               else 'A/' if re.search('A.?/', train_df['Ticket'].loc[ind]) 
                               else 'letter' if re.search('[a-zA-Z]+', train_df['Ticket'].loc[ind]) 
                               else 'number').count ()


# dropping features

# In[ ]:


train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# In[ ]:


for dataset in combine:
    dataset['Ticket'] = dataset['Ticket'].map(lambda x: 'STON' if re.search('STON', x) 
                               else 'PC' if re.search('P.?C', x)
                               else 'SC' if re.search('S.?C?', x)
                               else 'PP' if re.search('P.?P', x) 
                               else 'CA' if re.search('C.?A?', x) 
                               else 'A/' if re.search('A.?/', x) 
                               else 'letter' if re.search('[a-zA-Z]+', x) 
                               else 'number',
               na_action=None)
train_df.head()


# In[ ]:


for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].map(lambda x: 'NaN' if (pd.isnull(x)) else x[0],
               na_action=None)
train_df.head()


# In[ ]:


from scipy.stats.stats import spearmanr, kendalltau, pearsonr
s = train_df['Age'].values
p = train_df['Survived'].values
spearmanr(s, p)


# In[ ]:


for dataset in combine:
    dataset['Age'] = dataset['Age'].map(lambda x: 30 if (pd.isnull(x)) else x,
               na_action=None)
train_df.head()


# In[ ]:


cols_to_transform = [ 'Ticket', 'Cabin', 'Embarked' ]

encoded = pd.get_dummies(pd.concat([train_df,test_df], axis=0), columns = cols_to_transform)
train_rows = train_df.shape[0]
train_encoded = encoded.iloc[:train_rows, :]
test_encoded = encoded.iloc[train_rows:, :] 


# In[ ]:


from sklearn.feature_selection  import SelectKBest
selector = SelectKBest(k=20)

features = list(test_encoded)
selector.fit(train_encoded[features], train_encoded["Survived"])

scores = -np.log10(selector.pvalues_)

plt.bar(range(len(features)), scores)
plt.xticks(range(len(features)), features, rotation='vertical')
plt.show()


# In[ ]:


test_encoded = test_encoded.drop('Survived', 1)


# In[ ]:


print("After", train_encoded.shape, test_encoded.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

import numpy as np
#from sklearn.model_selection import train_test_split

#x_train, x_test = train_test_split(train_encoded, test_size = 0.09)
#y_train = x_train.pop("Survived")
#y_test = x_test.pop("Survived")

#x = train_encoded
#y = x.pop("Survived")


# In[ ]:


train_encoded.isnull().sum()


# In[ ]:


list(train_encoded)


# In[ ]:


from sklearn.feature_selection  import SelectKBest
selector = SelectKBest(k=20)

features = list(test_encoded)
selector.fit(train_encoded[features], train_encoded["Survived"])

scores = -np.log10(selector.pvalues_)

plt.bar(range(len(features)), scores)
plt.xticks(range(len(features)), features, rotation='vertical')
plt.show()


# In[ ]:


features1 = ['Age',
 'Fare',
 'Parch',
 'PassengerId',
 'Pclass',
 'Sex',
 'SibSp',
 'Ticket_A/',
 'Ticket_CA',
 'Ticket_PC',
 'Ticket_PP',
 'Ticket_SC',
 'Ticket_STON',
 'Ticket_letter',
 'Ticket_number',
 'Cabin_A',
 'Cabin_B',
 'Cabin_C',
 'Cabin_D',
 'Cabin_E',
 'Cabin_F',
 'Cabin_G',
 'Cabin_NaN',
 'Cabin_T',
 'Embarked_C',
 'Embarked_Q',
 'Embarked_S']


# In[ ]:


from sklearn.model_selection import GridSearchCV

alg_frst_model = RandomForestClassifier(oob_score = True ,n_jobs = -1,random_state =1)
alg_frst_params = [{
    "n_estimators": [80,90, 100],
    "min_samples_split": [4,6, 7],
    "min_samples_leaf": [2, 3,4]
}]
alg_frst_grid = GridSearchCV(alg_frst_model, alg_frst_params, refit=True, verbose=1, n_jobs=-1)
alg_frst_grid.fit(train_encoded[features1], train_encoded['Survived'])
alg_frst_best = alg_frst_grid.best_estimator_
print("Accuracy (random forest auto): {} with params {}"
      .format(alg_frst_grid.best_score_, alg_frst_grid.best_params_))


# In[ ]:


model = RandomForestClassifier(n_estimators = 100, min_samples_leaf=3, min_samples_split=8, oob_score = True ,n_jobs = -1,random_state =1)
model.fit(train_encoded[features1], train_encoded['Survived'])


# In[ ]:



test_encoded[features1].isnull().sum()


# In[ ]:


test_encoded['Fare'] = dataset['Fare'].map(lambda x: 50 if (pd.isnull(x)) else x)


# In[ ]:


prediction = model.predict (test_encoded[features1])


# In[ ]:


submission = pd.DataFrame({
    "PassengerId": test_encoded["PassengerId"],
    "Survived": prediction
})
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv("titanic-submission.csv", index=False)


# In[ ]:




