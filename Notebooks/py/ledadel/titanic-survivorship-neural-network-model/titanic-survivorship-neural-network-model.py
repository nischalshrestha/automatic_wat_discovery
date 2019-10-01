#!/usr/bin/env python
# coding: utf-8

# This is my first Kaggle project, a Titanic survivorship model using a simple Neural Network built with tflearn.
# I tried to explain each step of the process, and I hope it will help someone who is starting out just like I was.
# 
# Along the way I am reusing some code from the following kernels kindly published:
# https://www.kaggle.com/linxinzhe/tensorflow-deep-learning-to-solve-titanic/notebook
# https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
# 
# Features used for input (others were dropped):
# - Sex (one-hot encoded as 0s and 1s)
# - Age (raw values)
# - SibSp (raw values)
# - Parch (raw values)
# - Pclass (one hot encoded and broken into 3 new features one per class)
# - Embarked (one hot encoded and broken into 3 new features one per class)
# 
# My Neural Network architecture (built using tflearn):
# - Two hidden layers with tanh activations (20 and 40 units) and two dropout layers (0.8)
# (though I got pretty good accuracy using a single layer as well)
# - Output layer with two units and softmax activation
# - Training done using learning rate of 0.01, batch size=32 and 300 epochs

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display # Allows the use of display() for DataFrames

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#loading data
train_data= pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")

display(train_data.head())
display(test_data.head())
print ('train data len',len(train_data))
print ('test_data len',len(test_data))


# In[ ]:


#save PassengerId for evaluation
test_passenger_id=test_data["PassengerId"]


# In[ ]:


# Store the 'Survived' feature in a new variable 'outcomes' and remove it from the dataset 
# test dataset doesnt have this feature since we will be using it to test our prediction model
outcomes = train_data['Survived']
train_data = train_data.drop('Survived', axis = 1)


# In[ ]:


#combine all data for preprocessing
full_data = pd.concat([train_data, test_data])
print('full data length is',len(full_data))
print ('train data len is',len(train_data))
print ('test data len is',len(test_data))
display(full_data.head())
display(full_data.tail())


# In[ ]:


#Removing nonessential features that in my opinion do not have predictive power:
#name,ticket, passengerId, Fare
#also removing Cabin because too many NAs (1014 out of 1309), though I do think it is an interesting feature worth considering

cabin_null_count = full_data['Cabin'].isnull().sum()
print ('Cabin NA value count:',cabin_null_count)
print ('out of total',len(full_data['Cabin']))
trim_data = full_data.drop(['Name','Ticket','PassengerId','Cabin','Fare'],axis = 1)
display(trim_data[:10])


# In[ ]:


# Feature processing: sex, convert into binary
trim_data['Sex'] = trim_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
display(trim_data.head())


# In[ ]:


# Feature processing: Age
#there are many NA values in this feature so first we will replace them with a random number
#generated between (mean - std) and (mean + std) 
#approach borrowed from:
#https://www.kaggle.com/sinakhorami/titanic-best-working-classifier

age_avg   = trim_data['Age'].mean()
print (age_avg)
age_std    = trim_data['Age'].std()
print (age_std)
age_null_count = trim_data['Age'].isnull().sum()
print (age_null_count)
    
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
print (age_null_random_list)
trim_data['Age'][np.isnan(trim_data['Age'])] = age_null_random_list
print (trim_data[:20])


# In[ ]:


#checking if there are any more NA values in other features: yes, Embarked has more NAs
for feature in trim_data:
    print ( feature,trim_data[feature].isnull().sum())


# In[ ]:


# Feature processing: Embarked
#Embarked feature has only two NA values. 
#we will fill those with the most occurred value ( 'S' ).
#approach borrowed from:
#https://www.kaggle.com/sinakhorami/titanic-best-working-classifier

trim_data['Embarked'] = trim_data['Embarked'].fillna('S')


# In[ ]:


#Optional feature processing (only if you want to include 'Fare' in your model):
# 'Fare' has one missing value, NA, 
#we can fill in with the median value
#approach borrowed from:
#https://www.kaggle.com/sinakhorami/titanic-best-working-classifier

#trim_data['Fare'] = trim_data['Fare'].fillna(trim_data['Fare'].median())
#print (trim_data[:10])


# In[ ]:


#checking to make sure no more missing values
for feature in trim_data:
    print (feature,trim_data[feature].isnull().sum())


# In[ ]:


#Processing feature: Pclass and Embarked
#splitting each feature into new binary features 

def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


dummy_columns = ["Pclass",'Embarked']
trim_data=dummy_data(trim_data, dummy_columns)
display(trim_data.head())


# In[ ]:


#Optional code (if you want to scale Age data to range[0,1])
#approach borrowed from:
#https://www.kaggle.com/linxinzhe/tensorflow-deep-learning-to-solve-titanic/notebook

#from sklearn.preprocessing import MinMaxScaler
#features=['Age']

#def normalize(data,feature):
    #scaler = MinMaxScaler()
    #data[feature] = scaler.fit_transform(data[feature].values.reshape(-1,1))
    #return data

#for feature in features:
    #trim_data=normalize(trim_data,feature)

#display(trim_data.head())


# In[ ]:


#splitting data back to train and test set
train_data=trim_data[:891]
test_data=trim_data[891:]
print ('train_data len',len(train_data))
print ('test_data len',len(test_data))


# In[ ]:


#Converting 'outcomes' to shape (891,2)
from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
labels=lb.fit_transform(outcomes)
labels=np.hstack((labels,1-labels))
print (labels.shape)
print (labels[:5])


# In[ ]:


print (train_data.shape[1])


# In[ ]:


#convering pd.dataframe into array (this is needed for NN code to work)
print (type(train_data))
train_data=np.array(train_data)
print (type(train_data))
#print (train_data[:5])


# In[ ]:


import tensorflow as tf
import tflearn
# Network building
def build_model():
    # This resets all parameters and variables
    tf.reset_default_graph()
    
    num_hidden_1=20
    num_hidden_2=40
    num_output=2
    
    net=tflearn.input_data([None, train_data.shape[1]])              #input layer
    net=tflearn.fully_connected(net,num_hidden_1,activation='tanh') #hidden layer 1
    net=tflearn.dropout(net,0.8)
    net=tflearn.fully_connected(net,num_hidden_2,activation='tanh') #hidden layer 2
    net=tflearn.dropout(net,0.8)
    net=tflearn.fully_connected(net,num_output,activation='softmax') #output layer
    net=tflearn.regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    return model


# In[ ]:


#Build the model and saving as a variable 'model'
model = build_model()


# In[ ]:


# Training or fitting the model to the data. 
#validation_set=0.1 reserves 10% of the data set as the validation set. 
#You can also set the batch size and number of epochs with the batch_size and n_epoch keywords, respectively. .
model.fit(train_data, labels, validation_set=0.1,show_metric=True,batch_size=32, n_epoch=300)


# In[ ]:


model.predict(test_data)


# In[ ]:


predictions = np.array(model.predict(test_data)).argmax(axis=1)
print (predictions)


# In[ ]:


#getting predictions out into the right format for submission:
#https://www.kaggle.com/linxinzhe/tensorflow-deep-learning-to-solve-titanic/notebook
passenger_id=test_passenger_id.copy()
evaluation=passenger_id.to_frame()
evaluation["Survived"]=predictions
evaluation[:10]


# In[ ]:


# Write the solution to file
evaluation.to_csv("evaluation_submission_nn.csv",index=False)


# In[ ]:




