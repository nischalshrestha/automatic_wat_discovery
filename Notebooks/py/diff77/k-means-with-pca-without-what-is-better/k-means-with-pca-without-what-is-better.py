#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


# In[ ]:


train= pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')


# In[ ]:


# preview the data
train.head()


# In[ ]:


train.drop(['Ticket','Fare','Cabin','Name','PassengerId'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.nunique()


# In[ ]:


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])


# In[ ]:


train.isnull().sum()


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


train[['Pclass','SibSp','Parch','Embarked']]=train[['Pclass','SibSp','Parch','Embarked']].astype(str) 
train= pd.get_dummies(train)


# In[ ]:


train.head()


# In[ ]:


y=train['Survived']
data=train.drop('Survived',axis=1)
data = StandardScaler().fit_transform(data)


# In[ ]:


kmeans = KMeans(n_clusters=2, random_state=123, algorithm='elkan')


# In[ ]:


kmeans.fit(data)
clusters = kmeans.predict(data)
cluster_df = pd.DataFrame()
cluster_df['cluster'] = clusters
cluster_df['class'] = y
sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count')


# In[ ]:


f1_score(y, clusters)


# In[ ]:


mPCA = PCA(n_components=20)
PrincipleComponents = mPCA.fit_transform(data)
variance = mPCA.explained_variance_ratio_
variance_ratio = np.cumsum(np.round(variance, decimals=3)*100)
variance_ratio


# In[ ]:


plt.title("PCA components VS percentage of variance explained")
plt.ylabel("Percentage (%)")
plt.xlabel("# of components")
plt.plot(variance_ratio)


# In[ ]:


PCAdata = PrincipleComponents[:,:15]

kmeans.fit(PCAdata )
clusters = kmeans.predict(PCAdata )

cluster_df = pd.DataFrame()
cluster_df['cluster'] = clusters
cluster_df['class'] = y
sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count')


# In[ ]:


f1_score(y, clusters)


# In[ ]:


f1scores=[]
for i in  range(2,20):
    PCAdata = PrincipleComponents[:,:i]
    kmeans.fit(PCAdata )
    clusters = kmeans.predict(PCAdata )
    f1score=f1_score(y, clusters)
    f1scores.append(f1score)
    print('PCA dimensions: {}, f1 score {}'.format(i, f1score))
    


# In[ ]:


plt.plot(range(2,20),f1scores)

