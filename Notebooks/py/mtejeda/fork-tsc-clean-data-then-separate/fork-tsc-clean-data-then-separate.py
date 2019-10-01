#!/usr/bin/env python
# coding: utf-8

# Step 1: Import required libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MultiLabelBinarizer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


np.random.seed(0)
# Any results you write to the current directory are saved as output.


# Step 2:  Read in data and have a first look at it. This time we're going to merge the train and test datasets, do all the cleaning, and then separate. Note that since the survived column was added to the end of test, by setting sort to be false in the concatenate, it leaves survived as the last column for the rows from test but keeps it as the second column for the rows from train.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# #Check for Columns with Missing Data
# cols_with_missing = [col for col in train.columns if train[col].isnull().any()]
# cols_with_missing

# #Check for non-integer of float dtype columns
# cols_with_object = [col for col in train.columns if train[col].dtype == object]
# cols_with_object

test['Survived'] = ""
col_test = test.columns.tolist()
col_test.insert(1,col_test[-1])
del col_test[-1]
test = test[col_test]

frames = [train,test]
full = train.append(test,ignore_index=True)


# In[ ]:


test.iloc[-10:]


# In[ ]:


np.shape(test)


# In[ ]:


train.iloc[0:10]


# In[ ]:


train.iloc[-10:]


# In[ ]:


np.shape(train)


# In[ ]:


full.iloc[0:10]


# In[ ]:


full.iloc[-10:]


# Step 3: Create a function to clean the data

# In[ ]:


def data_cleanup(titanic_data):
    sex_dummy = pd.get_dummies(titanic_data.Sex)
    titanic_data['male'] = sex_dummy.male
    titanic_data['female'] = sex_dummy.female
    
    titanic_data.Age = titanic_data.Age.fillna(titanic_data.Age.median())
    titanic_data.Fare = titanic_data.Fare.fillna(titanic_data.Fare.median())
    
    titanic_data['Embarked'].fillna(method = 'bfill', axis = 0)
    embarked_dummy = pd.get_dummies(titanic_data['Embarked'])
    titanic_data['C'] = embarked_dummy.C
    titanic_data['Q'] = embarked_dummy.Q
    titanic_data['S'] = embarked_dummy.S
    
#     titanic_data['Salutation'] = titanic_data['Name'].apply(lambda x: x.split(",")[1:])
#     titanic_data['Salutation'] = titanic_data['Salutation'].apply(lambda x: x[0])
#     titanic_data['Salutation'] = titanic_data['Salutation'].apply(lambda x: x.split(".")[:1])
#     titanic_data['Salutation'] = titanic_data['Salutation'].apply(lambda x: x[0])
    
#     mlb = MultiLabelBinarizer()
#     # tt = pd.DataFrame(mlb.fit_transform(full.pop('Salutation')))
# #     mlb.classes_
                    
#     titanic_data = titanic_data.join(pd.DataFrame(mlb.fit_transform(titanic_data.pop('Salutation')),
#                                 columns = mlb.classes_))

    return titanic_data


# Step 4: Pass the train and test data through the cleaning function

# In[ ]:


full.index


# In[ ]:


full = data_cleanup(full)
full.index


# Step 5: Use one-hot encoding to encode categorical feature Salutation. MultiLabelBinarizer() is not working. After commenting out the last line of the data cleaning funciton where Salutation is converted from 1x1 array into an element, mlb works because it recognizes that inside of the 1x1 list is 1 element not a string of individual elements. I suspect that mlb.fit-transform works by looking at each element of a string based on the output of mlb.classes_ 

# In[ ]:


# mlb = MultiLabelBinarizer()
# # np.shape(mlb.fit_transform(train.pop('Salutation')))
# # other = pd.DataFrame(mlb.fit_transform(train.pop('Salutation')),columns = mlb.classes_,index=train.index)
# # mlb.classes_
# # other['Mr']
# # other.iloc[:,0]
# train = train.join(pd.DataFrame(mlb.fit_transform(train.pop('Salutation')),
#                                   columns = mlb.classes_,
#                                   index=train.index),
#                      on=train.index)
# # train = train.merge(train,pd.DataFrame(mlb.fit_transform(train.pop('Salutation')),
#                                        columns = mlb.classes_,
#                                        index=train.index),
#                     on=train.index)


# Step 5.1: Identify the unique elements of the Salutation column and use those to build a one-hot encoding. Didn't have to do this because I cleaned the cleaning function. Instead check the names of the columns in the dataframe to know which one's we'll be passing to build the model.

# In[ ]:


test.index.size
train.index.size
full.index.size


# Step 6: Update the cleaning data function to include one-hot encoding of Salutation. See Updated function in step 3

# Step 7: Added a few lines before building the model to extract the rows l

# In[ ]:


# df.loc[:, df.columns != 'b']
# train.index
# train.iloc[train.index]
# np.shape(full)
full_test = full.copy()
full_train = full.copy()

#train should be 0 to 890
X_features_train = full_train.drop(full.index[891:],axis=0)
# X_features_train = full.drop(labels=[891:1308],axis=0)
# X_features_train = full.drop(test.index,axis=0)
X_features_train = X_features_train.drop(['PassengerId','Survived','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
np.shape(X_features_train)

#currently this is returning 418 to 1308
X_features_train.iloc[0:891]


#test should be 891 to 1308
X_features_test = full_test.drop(full.index[:891],axis=0)
# X_features_test = full.drop(train.index,axis=0)
# # full.iloc[train.index]
# # full.iloc[test.index]
# # # np.shape(X_features_test)
# # # # np.shape(full)
X_features_test = X_features_test.drop(['PassengerId','Survived','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
# np.shape(X_features_test)
X_features_test.iloc[0:418]

# # # X_features_train = train.iloc[train.index,train.columns != 'PassengerId','Survived','Name','Sex','Ticket','Cabin','Embarked']
# # # X_features_test = test.loc[test.index,train.columns != 'PassengerId','Survived','Name','Sex','Ticket','Cabin','Embarked']
# # # # X_features_train = train[['Age','SibSp','Parch','Fare','Pclass','C','Q','S','male','female']].copy()
# # # # X_features_test  = test[['Age','SibSp','Parch','Fare','Pclass','C','Q','S','male','female']].copy()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

rf = RandomForestClassifier(n_estimators=50,max_depth=20,n_jobs=-1)
rf_model = rf.fit(X_features_train,train['Survived'])
y_pred = rf_model.predict(X_features_test)
# print(np.size(y_pred))
# print(y_pred)
# print(np.shape(X_features_test))
res = pd.DataFrame(y_pred,index=test['PassengerId'])
res.columns = ['Survived']
print(res.info)
res.to_csv('result')


# In[ ]:





# In[ ]:




