#!/usr/bin/env python
# coding: utf-8

# Hello everyone. I am doing Logistic Regression hw with Python.
# First of all, I am importing some libraries.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
print(os.listdir("../input"))


# I am including Titanic Data

# In[ ]:


data = pd.read_csv('../input/train.csv')


# Top 5 row

# In[ ]:


data.head()


# **General overview**

# In[ ]:


data.info()


# I'm throwing out unnecessary information from dataset.

# In[ ]:


data.drop(['PassengerId','Cabin','Name','Sex','Ticket','Embarked','Age'],axis=1,inplace=True)
data.head()


# I am defining X and Y values

# In[ ]:


y=data.Survived.values
x_data = data.drop(['Survived'],axis=1)


# Normalization process for  X values

# In[ ]:


x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values


# Train - Test split process for Data 
# 
# %80 Train
# 
# %20 Test

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


# I am creating initialize_weights_and_bias(acronym  => iwab ) and sigmoid functions for Logistic Regression Model
# 
# 
# 

# In[ ]:


def iwab(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head


# Than I will create forward_backward_propagation function(acronym  = > fbp )

# In[ ]:


def fbp(w,b,x_train,y_train):
    #Forward
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    #Backward
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                
    gradients = {"Derivative Weight": derivative_weight, "Derivative Bias": derivative_bias}
    
    return cost,gradients


# I will create a function for updating parameter

# In[ ]:


def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iterarion):
        cost,gradients = fbp(w,b,x_train,y_train)
        cost_list.append(cost)
        w = w - learning_rate * gradients["Derivative Weight"]
        b = b - learning_rate * gradients["Derivative Bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# Now, I am creating prediction function  

# In[ ]:


def predict(w,b,x_test):
    
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_pre = np.zeros((1,x_test.shape[1]))
    
    #   if z value is bigger than 0.5, our prediction is sign one (y_head=1),
    #   if z value is smaller than 0.5, our prediction is sign zero (y_head=0),
    
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            y_pre[0,i] = 0
        else:
            y_pre[0,i] = 1

    return y_pre


# Finally,  I will create Logistic Regression function   ( almost  I come to the end of the road  :)  )
# 

# In[ ]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
  
    dimension =  x_train.shape[0] 
    w,b = iwab(dimension)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    print("Test Accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 50)  


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 2, num_iterations = 200)  


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 500)  


# I tried different learning rate  and num iterations values for finding best test accuracy score. The best one I found test accuracy score: %71.5 for learning_rate = 3, num_iterations = 500.(	mustn't grumble )
# 
# It's could be better.
# 

# Terminally,  I will create Logistic Regression Model with Sklearn Library

# In[ ]:


lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print('Test Accuracy:',lr.score(x_test.T,y_test.T))


# As you can see Test accuracy is %70.94
# 
# It's could be better in the same way
# 
# I will be waiting for your comment 

# In[ ]:




