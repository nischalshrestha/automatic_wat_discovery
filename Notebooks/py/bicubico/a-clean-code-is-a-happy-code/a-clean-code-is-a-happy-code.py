#!/usr/bin/env python
# coding: utf-8

# First: Prepare the environment, import all relevant libraries

# In[51]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# Load the data for train and test sets. Check columns (later we'll need their names)

# In[52]:


# Datasources
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# **Data Analytics**

# Lets take a look at the train data

# In[53]:


train_data.head(20)


# Store the Passenger Id of the test set in an array, so later we can submit survival predictions linked to each Passenger Id

# In[54]:


test_id = test_data.PassengerId


# Store the survival status, so later we can fit the model using the train data (as X) and the target (as y).

# In[55]:


#Prediction Target
#Single column on train data that contains the prediction
train_y = train_data.Survived


# Cleaning: Remove some columns

# In[56]:


#cols_with_missing_values = [col for col in train_data.columns
#                                   if train_data[col].isnull().any()]
cols_with_missing_values=['Age','Cabin']
train_X = train_data.drop(['PassengerId','Survived']+cols_with_missing_values, axis=1)
test_X = test_data.drop(['PassengerId']+cols_with_missing_values, axis=1)


# Handling Categorical Values:  Selecting columns for One Hot Enconding depending on how many different values they have. If they have fewer than 10 different values, then we'll use it for One Hot Encoding

# In[57]:


# Categorical values. 
# Choosing only those columns for on hot encodding where the categorical value for any attribute is not more than 10
low_cardinality_cols = [cname for cname in train_X.columns
                                       if train_X[cname].nunique()< 10 and
                                       train_X[cname].dtype=="object"]
numeric_cols = [cname for cname in train_X.columns
                               if train_X[cname].dtype in ['int64', 'float64']]

useful_cols = low_cardinality_cols + numeric_cols
train_X = train_X[useful_cols]
test_X = test_X[useful_cols]


# In[58]:


def pairplot(X,y):
    X['y'] = y
    sns.pairplot(X,hue='y')
    X=X.drop(['y'],axis=1)
    return 


# In[59]:


#pairplot(train_X,train_y)
#train_X=train_X.drop(['y'],axis=1)


# Lets add some extra info, like family size

# In[60]:


def data_transform(df):
    df['FamilySize'] = df['SibSp']+df['Parch']
    df['CuicoHijoUnico'] = (4-df['Pclass'])/(df['Parch']+1)
    df = pd.get_dummies(df)
    return df

train_X = data_transform(train_X)
test_X = data_transform(test_X)


# Ready for fitting the model.

# In[65]:


my_pipeline=make_pipeline(SimpleImputer(),XGBRegressor())
my_pipeline.fit(train_X, train_y)

#Get Predictions
predictions = np.around(my_pipeline.predict(test_X),0).astype(np.int64)


# In[63]:





# Preparing submission file, just joining PassengerID and Predictions and putting them in a single dataframe.

# In[43]:


#Submit predictions
my_submission = pd.DataFrame({'PassengerId': test_id, 'Survived': predictions})
my_submission.describe()


# Check if the submision data seems to be ok

# In[44]:


my_submission.head(10)


# In[45]:


my_submission.to_csv('submission.csv', index=False)


# 149919 = 0.7942
# 
# 151210 = 0.786

# In[46]:


my_submission['producto']=my_submission['PassengerId']*my_submission['Survived']
my_submission['producto'].values.sum()


# Generate the submission file

# In[15]:





# In[ ]:




