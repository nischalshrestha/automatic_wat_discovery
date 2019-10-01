#!/usr/bin/env python
# coding: utf-8

# First: Prepare the environment, import all relevant libraries

# In[30]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd


# Load the data for train and test sets. Check columns (later we'll need their names)

# In[31]:


# Datasources
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

print("### TRAIN DATA COLUMNS ###")
print(train_data.columns)

print("### TEST DATA COLUMNS ###")
print(test_data.columns)


# **Data Analytics**

# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')



# Lets take a look at the train data

# In[33]:


train_data.head(20)


# sStore the Passenger Id of the test set in an array, so later we can submit survival predictions linked to each Passenger Id

# In[34]:


test_id = test_data.PassengerId


# Store the survival status, so later we can fit the model using the train data (as X) and the target (as y).

# In[35]:


#Prediction Target
#Single column on train data that contains the prediction
target = train_data.Survived


# Cleaning: Remove some columns

# In[36]:


#Identify columns with missing values to erase them from the train set
#finding columns with missing values
cols_with_missing_values = [col for col in train_data.columns
                                   if train_data[col].isnull().any()]
cols_with_missing_values=['Age','Cabin']


# In[37]:


print(cols_with_missing_values)


# In[38]:


# Excluding non valuable columns, like PassengerId, Survived, and columns with missing data
# Predictor Columns
candidate_train_predictors = train_data.drop(['PassengerId','Survived']+cols_with_missing_values, axis=1)
candidate_test_predictors = test_data.drop(['PassengerId']+cols_with_missing_values, axis=1)


# Handling Categorical Values:  Selecting columns for One Hot Enconding depending on how many different values they have. If they have fewer than 10 different values, then we'll use it for One Hot Encoding

# In[39]:


# Categorical values. 
# Choosing only those columns for on hot encodding where the categorical value for any attribute is not more than 10
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns
                                       if candidate_train_predictors[cname].nunique()< 10 and
                                       candidate_train_predictors[cname].dtype=="object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns
                               if candidate_train_predictors[cname].dtype in ['int64', 'float64']]

useful_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[useful_cols]
test_predictors = candidate_test_predictors[useful_cols]


# In[40]:


print(train_predictors.columns)


# In[41]:


tempdf = candidate_train_predictors
df = pd.DataFrame(tempdf)#, columns = useful_cols)
df['y'] = target
sns.pairplot(df,hue='y')


# In[49]:


print(train_predictors.columns)


# Lets add some extra info, like family size

# In[50]:


def feature_engineering(df):
    #df['FamilySize'] = df['SibSp']+df['Parch']
    df['CuicoHijoUnico'] = (4-df['Pclass'])/(df['Parch']+1)
    #df['FarePerPerson'] = df['Fare']/(df['FamilySize']+1)
    #df['AgeClass'] = df['Age']*df['Pclass']
    #df=df.drop(['Parch','SibSp'],axis=1)
    return df

train_predictors = feature_engineering(train_predictors)
test_predictors = feature_engineering(test_predictors)


# In[51]:


train_predictors.head(20)


# Hot Encode both train and test data

# In[16]:


# Adding dummy columns to categorical data
# HotEncoding
one_hot_encoded_train_data = pd.get_dummies(train_predictors)
one_hot_encoded_test_data = pd.get_dummies(test_predictors)

one_hot_encoded_train_data.describe()


# In[17]:


one_hot_encoded_test_data.describe()


# In[18]:


#tempdf = one_hot_encoded_train_data
#df = pd.DataFrame(tempdf)#, columns = useful_cols)
#df['y'] = target
#sns.pairplot(df,hue='y')


# In[19]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
one_hot_encoded_train_data = my_imputer.fit_transform(one_hot_encoded_train_data)
one_hot_encoded_test_data = my_imputer.fit_transform(one_hot_encoded_test_data)


# Lets check on encoded Train Data

# In[20]:


pd.DataFrame(one_hot_encoded_train_data).head()


# In[21]:


pd.DataFrame(one_hot_encoded_test_data).head()


# Ready for fitting the model.

# In[22]:


#Model Selection
#from sklearn.ensemble import RandomForestRegressor
#model = RandomForestRegressor()


from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=1000,learning_rate=0.05)
#Model Fit to Data
model.fit(one_hot_encoded_train_data, target,verbose=False)

#Get Predictions
test_predictions = np.around(model.predict(one_hot_encoded_test_data),0)
test_predictions = test_predictions.astype(np.int64)


# Preparing submission file, just joining PassengerID and Predictions and putting them in a single dataframe.

# In[23]:


#Submit predictions
my_submission = pd.DataFrame({'PassengerId': test_id, 'Survived': test_predictions})
my_submission.describe()


# Check if the submision data seems to be ok

# In[24]:


my_submission.head(10)


# Generate the submission file

# In[25]:


my_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




