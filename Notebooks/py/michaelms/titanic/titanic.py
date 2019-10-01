#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')


# In[ ]:


import seaborn as sns
sns.set()
import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

train.columns


# # Exploratory analysis
# 
# First a look at a age distributions for survival chance, split between male and female passengers

# In[ ]:


g = sns.FacetGrid(train, col='Survived', hue='Sex', hue_order=['male', 'female'])
g.map(sns.plt.hist, 'Age').add_legend();


# This becomes even more interesting, when we differentiate between passenger classes

# In[ ]:


g = sns.FacetGrid(train, col='Pclass', row='Survived', hue='Sex', hue_order=['male', 'female'])
g.map(sns.plt.hist, 'Age').add_legend();


# ## Machine learning

# In[ ]:


train.columns


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

#feature_names = train.columns.drop(['Name', 'PassengerId', 'Survived', 'Age'])
feature_names = ['Pclass', 'Sex']
train_vectorizer = DictVectorizer(sparse=False)
X_train = train_vectorizer.fit_transform(train[feature_names].T.to_dict().values())


# In[ ]:


clf = RandomForestClassifier()
clf.fit(X_train,
        train['Survived'])


# In[ ]:


X_test = train_vectorizer.fit_transform(test[feature_names].T.to_dict().values())
test['Survived'] = clf.predict(X_test)


# In[ ]:


test[['PassengerId', 'Survived']].to_csv('my_solution.csv', index=False)
test.head()


# In[ ]:




