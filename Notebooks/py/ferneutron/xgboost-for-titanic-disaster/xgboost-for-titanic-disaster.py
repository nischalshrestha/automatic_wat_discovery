#!/usr/bin/env python
# coding: utf-8

# # Machine Learning for Disaster

# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# # 1. Introduction
# 
# The idea of this work is to show in a simple and easy way, a solution for the classification problem in the Titanic disaster. In this notebook we use the XGBoost classifier.

# # 2. Let's Start

# In[ ]:


"""Importing libraries and stuff"""
# Author: Fernando-Lopez-Velasco

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import category_encoders as ce
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


"""Loading files as a pandas dataframe"""

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


"""Splitting data"""

Y = train['Survived'].copy() # We extract the target vector
Xtrain = train.drop(['Survived','PassengerId', 'Name'], axis=1) # Drop some columns which are not useful
Xtest = test.drop(['PassengerId','Name'], axis=1)


# In[ ]:


Xtest.head()


# # 3. Handling null values
# 
# In this section we will to solve the problem with missing or null values

# In[ ]:


"""First we split data in categorical and no categorical values"""

train_category = Xtrain.select_dtypes(include=['object']).copy()
test_category = Xtest.select_dtypes(include=['object']).copy()
train_float = Xtrain.select_dtypes(exclude=['object']).copy()
test_float = Xtest.select_dtypes(exclude=['object']).copy()


# ## 3.1 Null values in not categorical data
# 
# First we need to implement some method to adress this problem, in this case we will use the Imputer method provided by scikit-learn.

# In[ ]:


imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train_float)


# In[ ]:


Xtrain_float= imp.transform(train_float)
Xtest_float = imp.transform(test_float)


# ## 3.2 Transformation of categorical data into numerical format
# Now that we have solved the problem of null values in categorical data, we need to transform continuos values into discrete format. To do this we will use the technique "Backward Difference Encoder".

# In[ ]:


"""Declaring the object of BackwardDifferenceEncoder and fitting"""

encoder = ce.BackwardDifferenceEncoder(cols=['Sex', 'Ticket','Cabin','Embarked'])
encoder.fit(train_category)


# In[ ]:


"""Transforming data"""

Xtrain_category = encoder.transform(train_category)
Xtest_category = encoder.transform(test_category)


# In[ ]:


"""We need to drop some columns, this is because the transformation have generated extra columns"""

train_cols = Xtrain_category.columns
test_cols = Xtest_category.columns


# In[ ]:


flag = 0
cols_to_drop = []
for i in train_cols:
    for j in test_cols:
        if i == j:
            flag = 1
    if flag == 0:
        cols_to_drop.append(i)
    else:
        flag = 0


# In[ ]:


"""Dropping columns"""

Xtrain_category = Xtrain_category.drop(cols_to_drop, axis=1)


# In[ ]:


print(Xtrain_category.shape)
print(Xtest_category.shape)


# ## 3.3 Null values in categorical data
# 
# To solve the problem with null values in categorical data we will implement the Imputer function provided by scikit-learn.

# In[ ]:


"""Intialize the object imputer"""

imp.fit(Xtrain_category)


# In[ ]:


"""Transforming data"""

Xtrain_category = pd.DataFrame(imp.transform(Xtrain_category), columns = Xtrain_category.columns)
Xtest_category = pd.DataFrame(imp.transform(Xtest_category), columns = Xtest_category.columns)


# # 4. Scaling data
# 
# To scale data, we will use the function MinMaxScaler provided by scikit-learn.

# In[ ]:


"""Initializing and fiting"""

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(Xtrain_float)


# In[ ]:


"""Scaling"""

Xtrain_float = pd.DataFrame(min_max_scaler.transform(Xtrain_float), columns = train_float.columns)
Xtest_float = pd.DataFrame(min_max_scaler.transform(Xtest_float), columns = test_float.columns)


# In[ ]:


Xtest_float.head()


# # 5. Concatenating categorical and numerical data

# In[ ]:


Xtest_category.head()


# In[ ]:


"""As we have two kinds of datasets which are categorical and not categorical data, we need to concatenate both"""

Xtrain = pd.concat([Xtrain_float,Xtrain_category], axis=1)
Xtest = pd.concat([Xtest_float,Xtest_category], axis=1)


# # 6. XBoost Classifier
# 
# To solve this classification problem we will to apply the XBoost classifier.

# In[ ]:


"""Initializing the XBoost classifier"""

model = xgb.XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.1)


# In[ ]:


"""Fitting"""

model.fit(Xtrain, Y)


# In[ ]:


"""Making a prediction"""

Ypred = model.predict(Xtest)


# In[ ]:


"""Saving data"""
Ypred = pd.DataFrame({'Survived':Ypred})
prediction = pd.concat([test['PassengerId'], Ypred], axis=1)
prediction.to_csv('predictions_xboost.csv', sep=',', index=False)


# In[ ]:


prediction.head()


# In[ ]:




