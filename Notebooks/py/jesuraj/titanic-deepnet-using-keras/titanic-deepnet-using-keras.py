#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import required packages
import numpy as np 
import pandas as pd 
import scipy as si
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[ ]:


# Input data files are available in the "../input/" directory.
# To list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


#Read train & test csv files
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


#EDA analysis for train & test data
train.head()


# In[ ]:


test.head()


# In[ ]:


#Numerical variables descriptive analysis
train.describe()


# In[ ]:


#Categorical variable analysis
train.describe(include=['O'])


# In[ ]:


#To get descriptor inforamtion
train.info()


# In[ ]:


# Data preparation for keras model
# To keep only objects
train_obj = train.select_dtypes(include=['object']).copy()
train_obj.head()


# In[ ]:


# To check the frequency
train['Sex'].value_counts()


# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


train=train.replace(["male","female"],[0,1])


# In[ ]:


train.head()


# In[ ]:


train=train.replace(['S','C','Q'],[0,1,2])


# In[ ]:


train= train.fillna(0)
train.head()


# In[ ]:


train.info()


# In[ ]:


x=train[["PassengerId","Pclass", "Sex","Age","SibSp","Parch","Fare","Embarked"]]   
y=train[["Survived"]]
x = x.astype(np.float32).values
y = y.astype(np.float32).values


# In[ ]:


#Test data preparation
test = test.replace(["male", "female"], [0,1])
test = test.replace(["S", "C", "Q"], [0,1,2])
test= test.fillna(0)


# In[ ]:


x_test=test[["PassengerId","Pclass", "Sex","Age","SibSp","Parch","Fare","Embarked"]]  
x_test.head()


# In[ ]:


# Import Keras packages
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
# data split
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold


# In[ ]:


seed = 1234
np.random.seed(seed)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.33)


# In[ ]:


x.shape[:]


# In[ ]:


y.shape[:]


# In[ ]:


x_train.shape[:]


# In[ ]:


x_test.shape[:]


# In[ ]:


y_train.shape[:]


# In[ ]:


y_test.shape[:]


# In[ ]:


# Define Model
model = Sequential()
#input layer
model.add(Dense(8, input_dim=(8)))
model.add(Activation("relu"))
# hidden layers1
model.add(Dense(8))
model.add(Activation("relu"))
# hidden layers2
model.add(Dense(8, input_dim=(8)))
model.add(Activation("relu"))
# output layer
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


#Complie
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Learning
model.fit(x, y, nb_epoch=100, batch_size=10)


# In[ ]:


predict = np.round(model.predict(x_test))


# In[ ]:


predictions = pd.DataFrame(predict)


# In[ ]:


predictions.head()


# In[ ]:


titanic_sub=pd.concat([test[["PassengerId"]], predictions], axis = 1)
titanic_sub=titanic_sub.rename(columns={0:'Survived'})


# In[ ]:


titanic_sub.head()


# In[ ]:


titanic_sub.to_csv("titanic_sub.csv", index=False)

