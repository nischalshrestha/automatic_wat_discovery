#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from matplotlib import pyplot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


PATH = '../input/'


# In[ ]:


Train = pd.read_csv(f'{PATH}train.csv')
Test = pd.read_csv(f'{PATH}test.csv')


# In[ ]:


Train.head(), Test.head()


# In[ ]:


Train_Y = Train.Survived
WholeDataSet =   pd.concat( [ Train.drop(['Survived'], axis =1 ), Test])


# In[ ]:


Train_Y.head(), WholeDataSet.head()


# In[ ]:


WholeDataSet.isnull().sum(axis=0) # check the value in each column step 1


# In[ ]:


WholeDataSet.describe() # check the value in each column step 2


# In[ ]:


WholeDataSet.drop('Cabin', axis=1, inplace=True) # too many data missing in the dataset


# In[ ]:


# As kaggle don't support this feather, we have to remove the name column as well.
#
#from nameparser import HumanName
#def GetTitleFromHumanName(Name):
#    return(HumanName(Name).title)
#GetTitleFromHumanName('Bjorklund, Mr. Ernst Herbert')


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    #Name = cols[1]
    Pclass = cols[1]
    if pd.isnull(Age):
        return  WholeDataSet[WholeDataSet.Pclass == Pclass]['Age'].median()
    else:
        return Age


# In[ ]:


WholeDataSet['Age'] = WholeDataSet[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


WholeDataSet['Embarked'] = WholeDataSet['Embarked'].fillna('S') # most the data is S and the missing data is not too much.


# In[ ]:


WholeDataSet['Fare'] = WholeDataSet['Fare'].fillna(WholeDataSet.Fare.median()) # use the average fare is good enough


# In[ ]:


WholeDataSet.isnull().sum(axis=0) # check the data again


# In[ ]:


WholeDataSet.drop(['PassengerId','Ticket', 'Name'], axis=1, inplace=True) # remove the data we don't know how to deal yet


# In[ ]:


WholeDataSet = pd.get_dummies(WholeDataSet) # one-hot code to the categorical column


# In[ ]:


WholeDataSet.head()


# In[ ]:


# fine tune the model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed = 22
test_size = 0.2

X_train, X_test, Y_train, Y_test = train_test_split(WholeDataSet[:len(Train_Y)], Train_Y, test_size = test_size, random_state = seed) 


# In[ ]:


# you might try this many time to try find the better parameters as below
md = xgb.XGBClassifier(n_estimators=25, max_depth=9, learning_rate=0.2, subsample=0.6, colsample_bylevel=0.8)
md.fit( X_train, Y_train )


# In[ ]:


predictions = md.predict(X_test)


# In[ ]:


accuracy_score(Y_test, predictions) #looks good, but the submit result only be 0.77, not sure why


# In[ ]:


from matplotlib import pyplot


# In[ ]:


md.feature_importances_ 


# In[ ]:


pyplot.bar(range(len(md.feature_importances_)), md.feature_importances_)
pyplot.show()


# In[ ]:


# submit the result with the full data set trainning
md = xgb.XGBClassifier(n_estimators=25, max_depth=9, learning_rate=0.2, subsample=0.6, colsample_bylevel=0.8)
md.fit( WholeDataSet[:len(Train_Y)], Train_Y )
pred = md.predict(WholeDataSet[len(Train_Y):])
Test['Survived'] = pred
# Test.to_csv(f'{PATH}submission.csv',  columns=['PassengerId', 'Survived'], index=False) # no permission to write, sorry


# In[ ]:




