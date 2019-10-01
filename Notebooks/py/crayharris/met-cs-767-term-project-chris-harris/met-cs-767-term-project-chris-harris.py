#!/usr/bin/env python
# coding: utf-8

# Author: Chris Harris
# Class: BU CS 767 - Machine Learning
# 
# This notebook builds off of Stefan Bergsteins CNN model using Keras. I liked his approach and my goal is to add to it and to compare performance from his CNN to an RNN: https://www.kaggle.com/crayharris/keras-deep-learning-on-titanic-data 
# 
# 

# In[ ]:


# Code Borrowed from Stefan Bergstein
# https://www.kaggle.com/stefanbergstein/keras-deep-learning-on-titanic-data/notebook
# data processing
import numpy as np
import pandas as pd 

# machine learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM #Dropout and LSTM needed for RNN

from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# utils
import time
from datetime import timedelta

# some configuratin flags and variables
verbose=0 # Use in classifier

# Input files
file_train='../input/train.csv'
file_test='../input/test.csv'

# defeine random seed for reproducibility
seed = 23
np.random.seed(seed)

# read training data
train_df = pd.read_csv(file_train,index_col='PassengerId')


# In[ ]:


# Show the columns
train_df.columns.values


# In[ ]:


# Show the data that we've loaded
train_df.head()


# Perform data cleansing
# * Remove Name, Cabin and Ticket (These are beyond the scope of our analysis)
# * Transform categorical features to numeric ones

# In[ ]:


def prep_data(df):
    # Drop unwanted features
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # Fill missing data: Age and Fare with the mean, Embarked with most frequent value
    df[['Age']] = df[['Age']].fillna(value=df[['Age']].mean())
    df[['Fare']] = df[['Fare']].fillna(value=df[['Fare']].mean())
    df[['Embarked']] = df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())
    
    # Convert categorical  features into numeric
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
      
    # Convert Embarked to one-hot
    enbarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = df.drop('Embarked', axis=1)
    df = df.join(enbarked_one_hot)

    return df


# Verify that the null data has been removed

# In[ ]:


train_df = prep_data(train_df)
train_df.isnull().sum()


# Break out training and test data

# In[ ]:


# X contains all columns except 'Survived'  
X = train_df.drop(['Survived'], axis=1).values.astype(float)

# It is almost always a good idea to perform some scaling of input values when using neural network models (jb).

scale = StandardScaler()
X = scale.fit_transform(X)

# Y is just the 'Survived' column
Y = train_df['Survived'].values


# Basic CNN with Keras
# * This net has 16 neurons
# * Two hidden layers

# In[ ]:


def create_model(optimizer='adam', init='uniform'):
    # create model
    if verbose: print("**Create model with optimizer: %s; init: %s" % (optimizer, init) )
    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# In[ ]:


# CHRIS NEW CODE FOR FINAL PROJ
def create_model_5layer(optimizer='adam', init='uniform'):
    # create model
    if verbose: print("**Create model with optimizer: %s; init: %s" % (optimizer, init) )
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(16, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(2, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# In[ ]:


# CHRIS NEW CODE FOR FINAL PROJ
def create_model_10layer(optimizer='adam', init='uniform'):
    # create model
    if verbose: print("**Create model with optimizer: %s; init: %s" % (optimizer, init) )
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(16, kernel_initializer=init, activation='relu'))
    model.add(Dense(16, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(2, kernel_initializer=init, activation='relu'))
    model.add(Dense(2, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# In[ ]:


# CHRIS NEW CODE FOR FINAL PROJ
def create_model_20layer(optimizer='adam', init='uniform'):
    # create model
    if verbose: print("**Create model with optimizer: %s; init: %s" % (optimizer, init) )
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(16, kernel_initializer=init, activation='relu'))
    model.add(Dense(16, kernel_initializer=init, activation='relu'))
    model.add(Dense(16, kernel_initializer=init, activation='relu'))
    model.add(Dense(16, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(2, kernel_initializer=init, activation='relu'))
    model.add(Dense(2, kernel_initializer=init, activation='relu'))
    model.add(Dense(2, kernel_initializer=init, activation='relu'))
    model.add(Dense(2, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Setting static hyperparams from previous runs where grid search was used

# In[ ]:


# pre-selected paramters
best_epochs = 200
best_batch_size = 5
best_init = 'glorot_uniform'
best_optimizer = 'rmsprop'


# Model Build and Perform Prediction

# In[ ]:


# Create a classifier with best parameters
model_pred = KerasClassifier(build_fn=create_model, optimizer=best_optimizer, init=best_init, epochs=best_epochs, batch_size=best_batch_size, verbose=verbose)
model_pred.fit(X, Y)

# Read test data
test_df = pd.read_csv(file_test,index_col='PassengerId')
# Prep and clean data
test_df = prep_data(test_df)
# Create X_test
X_test = test_df.values.astype(float)
# Scaling
X_test = scale.transform(X_test)

# Predict 'Survived'
prediction = model_pred.predict(X_test)


# Create CNN Predictions for 2 layer model

# In[ ]:


submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': prediction[:,0],
})

submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('CNN-submission-simple-cleansing_2Layer.csv', index=False)


# 5 Layer Model creation

# In[ ]:


# CHRIS NEW CODE FOR FINAL PROJ
# Create a classifier with best parameters
model_pred5 = KerasClassifier(build_fn=create_model_5layer, optimizer=best_optimizer, init=best_init, epochs=best_epochs, batch_size=best_batch_size, verbose=verbose)
model_pred5.fit(X, Y)

# Read test data
test_df5 = pd.read_csv(file_test,index_col='PassengerId')
# Prep and clean data
test_df5 = prep_data(test_df5)
# Create X_test
X_test5 = test_df.values.astype(float)
# Scaling
X_test5 = scale.transform(X_test5)

# Predict 'Survived'
prediction = model_pred5.predict(X_test5)


# Create CNN Preditions for 5 layer model

# In[ ]:


# CHRIS NEW CODE FOR FINAL PROJ
submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': prediction[:,0],
})

submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('CNN-submission-simple-cleansing_5Layer.csv', index=False)


# 10 Layer Model creation

# In[ ]:


# CHRIS NEW CODE FOR FINAL PROJ
# Create a classifier with best parameters
model_pred10 = KerasClassifier(build_fn=create_model_10layer, optimizer=best_optimizer, init=best_init, epochs=best_epochs, batch_size=best_batch_size, verbose=verbose)
model_pred10.fit(X, Y)

# Read test data
test_df10 = pd.read_csv(file_test,index_col='PassengerId')
# Prep and clean data
test_df10 = prep_data(test_df10)
# Create X_test
X_test10 = test_df.values.astype(float)
# Scaling
X_test10 = scale.transform(X_test10)

# Predict 'Survived'
prediction = model_pred10.predict(X_test10)


# Create CNN Preditions for 10 layer model

# In[ ]:


# CHRIS NEW CODE FOR FINAL PROJ
submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': prediction[:,0],
})

submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('CNN-submission-simple-cleansing_10Layer.csv', index=False)


# 20 Layer Model creation

# In[ ]:


# CHRIS NEW CODE FOR FINAL PROJ
# Create a classifier with best parameters
model_pred20 = KerasClassifier(build_fn=create_model_20layer, optimizer=best_optimizer, init=best_init, epochs=best_epochs, batch_size=best_batch_size, verbose=verbose)
model_pred20.fit(X, Y)

# Read test data
test_df20 = pd.read_csv(file_test,index_col='PassengerId')
# Prep and clean data
test_df20 = prep_data(test_df20)
# Create X_test
X_test20 = test_df.values.astype(float)
# Scaling
X_test20 = scale.transform(X_test20)

# Predict 'Survived'
prediction = model_pred20.predict(X_test20)


# Create CNN Preditions for 20 layer model

# In[ ]:


# CHRIS NEW CODE FOR FINAL PROJ
submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': prediction[:,0],
})

submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('CNN-submission-simple-cleansing_20Layer.csv', index=False)


# In[ ]:


def create_rnn_model(optimizer='adam', init='uniform'):
    # create model
    if verbose: print("**Create model with optimizer: %s; init: %s" % (optimizer, init) )
    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

