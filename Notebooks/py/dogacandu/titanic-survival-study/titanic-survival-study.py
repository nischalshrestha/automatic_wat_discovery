#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ****Titanic Survival Study

# read the files into dataframes:

# In[ ]:


import pandas as pd
titanic_traindf=pd.read_csv('../input/train.csv')
titanic_testxdf=pd.read_csv('../input/test.csv')
titanic_testydf=pd.read_csv('../input/gender_submission.csv')


# In[ ]:


titanic_traindf.head(2)


# drop string columns:
# 

# In[ ]:


titanic_traindf=titanic_traindf.drop('Name', axis=1)
titanic_traindf=titanic_traindf.drop('Ticket', axis=1)
titanic_traindf=titanic_traindf.drop('Cabin', axis=1)
titanic_traindf=titanic_traindf.drop('Embarked', axis=1)
titanic_testxdf=titanic_testxdf.drop('Name', axis=1)
titanic_testxdf=titanic_testxdf.drop('Ticket', axis=1)
titanic_testxdf=titanic_testxdf.drop('Cabin', axis=1)
titanic_testxdf=titanic_testxdf.drop('Embarked', axis=1)


# In[ ]:


titanic_traindf.head(2)


# Sex is important variable in predicting survival, therfore I keep it. Survival rates by gender are: 

# In[ ]:


titanic_groupby=titanic_traindf.groupby('Sex')
titanic_groupby.Survived.sum()/titanic_groupby.Survived.count()


# I replace values in 'Sex' column with 0 and 1 for 'female' and 'male' respectively
# 

# In[ ]:


titanic_traindf['Sex'].replace(to_replace=['female','male'], value=[0,1],inplace=True)
titanic_testxdf['Sex'].replace(to_replace=['female','male'], value=[0,1],inplace=True)


# There is also many missing values in Age column. I keep it as it is because xgboost accepts NA values. for scikit-learn I'll need to impute this column. 

# In[ ]:


titanic_traindf.Age.isna().sum()


# Next step is xgboost modelling which has accuracy 79%

# In[ ]:


X=titanic_traindf[['Pclass','Sex','Age', 'SibSp','Parch','Fare']]
y=titanic_traindf['Survived']
test_X=titanic_testxdf[['Pclass','Sex','Age','SibSp', 'Parch','Fare']]
from xgboost import XGBClassifier
model=XGBClassifier(n_estimators=1000,learning_rate=0.05)
model.fit(X,y, early_stopping_rounds=5,eval_set=[(X, y)],verbose=False)
predicted = model.predict(test_X)


# Random Forest with default parameters gives same accuracy 79%

# In[ ]:


from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
imputer = Imputer()
X_imputed=pd.DataFrame(imputer.fit_transform(X))
test_X_imputed=pd.DataFrame(imputer.fit_transform(test_X))
X_imputed.columns=['Pclass','Sex','Age', 'SibSp','Parch','Fare']
test_X_imputed.columns=['Pclass','Sex','Age', 'SibSp','Parch','Fare']
rf = RandomForestClassifier()
rf.fit(X_imputed, y)
predicted = model.predict(test_X_imputed)


# Finally lets try KNN, which gives 60% accuracy
# 

# In[ ]:


from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_imputed, y)
predicted = knn.predict(test_X_imputed)

