#!/usr/bin/env python
# coding: utf-8

# TP4 : decision trees and random forest

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn import tree

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test    = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


corr = train.corr()
_ , ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })


# In[ ]:


#type(train['Survived'][0])
train['Name']


# In[ ]:


type(train)


# **Data Preparation**
# 

# In[ ]:


target = train['Survived'].values
train = train.drop(['Survived'], axis=1)
id_test = test['PassengerId']
train_size = train.shape[0]


# In[ ]:


titanic = pd.concat((train,test), axis=0, ignore_index=True)


# In[ ]:


titanic = titanic.drop(['Name','PassengerId','Ticket'], axis=1)


# raplacing the Na variables 

# In[ ]:


titanic.Age = titanic.Age.fillna(titanic.Age.mean())
titanic.Fare = titanic.Age.fillna(titanic.Fare.mean())
titanic.Cabin = titanic.Cabin.fillna( 'U' )


# In[ ]:


titanic.Cabin = titanic.Cabin.map( lambda c : c[0] )


# In[ ]:


print ("Nb null dans Age : "+str(titanic.Age.isnull().sum()))
print ("Nb null dans Parch : "+str(titanic.Parch.isnull().sum()))
print ("Nb null dans Pclass : "+str(titanic.Pclass.isnull().sum()))
print ("Nb null dans Fare : "+str(titanic.Fare.isnull().sum()))
print ("Nb null dans Sex : "+str(titanic.Sex.isnull().sum()))
print ("Nb null dans Cabin : "+str(titanic.Cabin.isnull().sum()))


# In[ ]:


features = ['Pclass','Sex','SibSp','Parch','Cabin', 'Embarked']


# One Hot encoding

# In[ ]:


for f in features:
    titanic_dummy = pd.get_dummies(titanic[f], prefix = f)
    titanic = titanic.drop([f], axis = 1)
    titanic = pd.concat((titanic, titanic_dummy), axis = 1)


# In[ ]:


titanic


# **Data Modeling**

# In[ ]:


vals = titanic.values
X = vals[:train_size]
y = target
X_test = vals[train_size:]


# In[ ]:


X


# In[ ]:


model = GradientBoostingClassifier()
model.fit(X,y)
y_pred = model.predict(X_test)


# In[ ]:


from IPython.display import Image

dot_data = tree.export_graphviz(model, out_file='tree.dot', 
                         filled=True, rounded=True,  
                         special_characters=True)  


# In[ ]:


test = pd.DataFrame( { 'PassengerId': id_test , 'Survived': y_pred } )


# In[ ]:


test.to_csv( 'titanic_pred.csv' , index = False )

