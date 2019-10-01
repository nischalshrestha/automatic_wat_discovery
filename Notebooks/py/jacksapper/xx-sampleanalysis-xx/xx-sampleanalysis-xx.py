#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#setup...
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
get_ipython().magic(u'matplotlib inline')


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# In[ ]:


g = sns.FacetGrid(df, col='Survived', row = 'Sex', hue='Pclass')
g.map(plt.hist, 'Pclass')


# In[ ]:


df['is_male'] = pd.get_dummies(df['Sex'])['male']
df_test = pd.read_csv('../input/test.csv')
df_test['is_male'] = pd.get_dummies(df_test['Sex'])['male']


# In[ ]:


model = LogisticRegression();
model_params = ['is_male', 'Pclass']
target_params = 'Survived'
X = df[model_params]
y = df[target_params]
model.fit(X,y)
model.score(X,y)


# In[ ]:


X_test = df_test[model_params]
df_test['Survived'] = model.predict(X_test)


# In[ ]:


df_test
df_test[['PassengerId','Survived']].to_csv('predict.csv', index=False)

