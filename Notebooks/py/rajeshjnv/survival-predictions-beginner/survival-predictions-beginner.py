#!/usr/bin/env python
# coding: utf-8

# # In this kernel i develop a model and find the accurecy using the train data and aply the same to predict the test data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')


# In[ ]:


data = pd.read_csv('../input/train.csv')
data.head(3)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.Survived.sum()


# In[ ]:


data.hist(figsize=(12,8))
plt.figure()


# In[ ]:


y=data.pop('Survived')
y.head(3)


# In[ ]:


total=data.isnull().sum()
total


# In[ ]:


numeric_variables=list(data.dtypes[data.dtypes !="object"].index)
data[numeric_variables].head(3)                                   


# In[ ]:


data['Age'].plot.hist()


# In[ ]:


data['Age'].fillna(data.Age.mean(),inplace=True)


# In[ ]:


data[numeric_variables].head(3) 


# In[ ]:


data['Age'].plot.hist()


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model=RandomForestClassifier(n_estimators=100)
model.fit(data[numeric_variables],y)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y,model.predict(data[numeric_variables]))*100


# # in this model i got accurecy of 100% 

# # prediction

# In[ ]:


td=pd.read_csv('../input/test.csv')
td[numeric_variables].head(3)


# In[ ]:


td.info()


# In[ ]:


td.hist(figsize=(12,8))
plt.figure()


# In[ ]:


total=td.isnull().sum()
total


# In[ ]:


td.describe()


# In[ ]:


td['Age'].plot.hist()


# In[ ]:


td['Age'].fillna(td.Age.mean(),inplace=True)


# In[ ]:


td['Age'].plot.hist()


# In[ ]:


td=td[numeric_variables].fillna(td.mean()).copy()


# In[ ]:


td[numeric_variables].head(3)


# In[ ]:


y_pred=model.predict(td[numeric_variables])


# In[ ]:


sub=pd.DataFrame({
          'PassengerId':td['PassengerId'],
           'Survived':y_pred
})


# In[ ]:


sub.head(20)


# # I am a beginner in data science, Your feedback and suggestions is very important to me,upvote will motivates and appreciate me for further work.
# 
# I hope this kernel is helpfull for you for any querry or advice please commment.
