#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler # Used for scaling of data
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
get_ipython().system(u'ls')
# Any results you write to the current directory are saved as output.


# In[2]:


raw_train = pd.read_csv('../input/train.csv', index_col=0)
raw_test = pd.read_csv('../input/test.csv', index_col=0)

#Show trining columns
raw_train.columns.values
raw_train.shape


# In[3]:


raw_train.head()


# In[4]:


# Show if any data is null
raw_train.isnull().sum()


#     - Drop unwanted features ['Name', 'Ticket', 'Cabin']
#     - Fill missing data: Age and Fare with the mean, Embarked with most frequent value
#     - Convert categorical features into numeric
#     - Convert Embarked to one-hot

# In[5]:


def prepare_data(data):
    
    data = data.drop(columns=["Name", "Ticket", "Cabin"])
    
    # Fill empty data with mean data instead
    data[['Age']] = data[['Age']].fillna(value=data[['Age']].mean())
    data[['Embarked']] = data[['Embarked']].fillna(value=data['Embarked'].value_counts().idxmax())
    data[['Fare']] = data[['Fare']].fillna(value=data[['Fare']].mean())
    
    # Convert categorical  features into numeric
    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    # Convert Embarked to one-hot
    embarked_one_hot = pd.get_dummies(data['Embarked'], prefix='Embarked')
    data = data.drop('Embarked', axis=1)
    data = data.join(embarked_one_hot)
    
    return data
    
    
    


# In[6]:


train_data = prepare_data(raw_train)
train_data.isnull().sum()


# In[14]:


# Create data and ground truth
# X contains all columns except 'Survived'  
X = train_data.drop(['Survived'], axis=1).values

# It is almost always a good idea to perform some scaling of input values when using neural network models (jb).

scale = StandardScaler()
X = scale.fit_transform(X)

# Y is just the 'Survived' column
Y = train_data['Survived'].values


# **Simple Network using Keras**
#     - Input later with 16 neuron (units/outputs).
#     - Two hidden layers.
#     - Output layer with a single neuron and sigmoid activation function to output a value between 0 and 1.
#     - Optimizer and intit will be searched with GridSearch.

# In[23]:


def create_model(optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(28, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# In[24]:


# Create a classifier with best parameters
model = create_model()

model.fit(X, Y, epochs=400, batch_size=64)


# In[26]:


# Check test data
raw_test.isnull().sum()


# In[27]:


# Prep and clean data
data_test = prepare_data(raw_test)
# Create X_test
X_test = data_test.values.astype(float)
# Scaling
X_test = scale.transform(X_test)
# Check test data
data_test.isnull().sum()


# In[28]:


# Predict 'Survived'
prediction = model.predict_classes(X_test)


# In[29]:


submission = pd.DataFrame()
submission['PassengerId'] = data_test.index
submission['Survived'] = prediction


# In[30]:


submission.shape


# In[31]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




