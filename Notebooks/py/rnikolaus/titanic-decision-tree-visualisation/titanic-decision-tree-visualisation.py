#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier


# In[ ]:


def loaddf(filename):
    df = pd.read_csv('../input/'+filename).drop(['Cabin','Ticket','Name'],axis=1) #drop stuff we don't use
    df = pd.concat([df, pd.get_dummies(df['Sex'])],axis=1).drop(['Sex'],axis=1)
    df = pd.concat([df, pd.get_dummies(df['Embarked'])],axis=1).drop(['Embarked'],axis=1)
    df['Fsize']=df['Parch']+df['SibSp']+1 #calculate family size
    return df

def cleanse(df, dropna=True):
    if dropna:
        df = df.dropna()
    else:
        df['Age']=df['Age'].fillna(28)
        df['Fare']=df['Fare'].fillna(7.71)
        #df['Pclass']=df['Pclass'].fillna(3)
        df=df.fillna(0)
    return df
    
def load_X(df):
    train_X=df[['male','Pclass','Age','Fsize']]
    return train_X
def load_y(df):
    train_y=df['Survived']
    return train_y


# In[ ]:


train = cleanse(loaddf('train.csv'),dropna=False)
train_X, train_y = load_X(train),load_y(train)
model = DecisionTreeClassifier(max_depth=3)
model.fit(train_X, train_y)
pd.DataFrame([model.feature_importances_],columns=train_X.columns )


# In[ ]:


from sklearn.tree import export_graphviz
export_graphviz(model, out_file='tree.dot', feature_names = train_X.columns.tolist(),class_names=['Died','Survived'],
           rounded = True, proportion = False, precision = 0, filled = True)
get_ipython().system(u'dot -Tpng tree.dot -o tree.png ')
from IPython.display import Image
Image(filename = 'tree.png')

