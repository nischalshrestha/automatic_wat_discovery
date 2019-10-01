#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# # Acquire data
# 

# In[ ]:


train_df=pd.read_csv('../input/train.csv',index_col=0)
test_df=pd.read_csv('../input/test.csv',index_col=0)


# In[ ]:


train_df.describe()


# In[ ]:


test_df.head()


# # Combine data

# In[ ]:


y_train=train_df.pop('Survived')
all_df=pd.concat((train_df,test_df),axis=0)


# In[ ]:


y_train.head()


# In[ ]:


all_df.drop(['Name'],axis=1,inplace=True)


# In[ ]:


all_df.drop(['Ticket'],axis=1,inplace=True)
all_df.drop(['Cabin'],axis=1,inplace=True)


# In[ ]:


all_df.head()


# # Feature Engineering

# In[ ]:


all_df['Pclass'].dtype


# In[ ]:


all_df['Pclass']=all_df['Pclass'].astype(str)


# In[ ]:


all_df['Pclass'].value_counts()


# In[ ]:


pd.get_dummies(all_df['Pclass'],prefix='Pclass').head()


# In[ ]:


all_dummy_df=pd.get_dummies(all_df)
all_dummy_df.head()


# In[ ]:


all_dummy_df['SibSp'].value_counts()
#all_dummy_df['Parch'].value_counts()


# In[ ]:


all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)


# In[ ]:


mean_cols= all_dummy_df.mean()
mean_cols.head(10)


# In[ ]:


all_dummy_df=all_dummy_df.fillna(mean_cols)


# In[ ]:


all_dummy_df['Fare']=np.log1p(all_dummy_df['Fare'])


# In[ ]:


all_dummy_df.head()


# # Model

# In[ ]:


dummy_train_df=all_dummy_df.loc[train_df.index]
dummy_test_df=all_dummy_df.loc[test_df.index]


# In[ ]:


X_train = dummy_train_df.values
X_test = dummy_test_df.values


# In[ ]:


dummy_train_df.shape


# In[ ]:


def tanh(x):  
    return np.tanh(x)

def tanh_deriv(x):  
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):  
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):  
    return logistic(x)*(1-logistic(x))


# In[ ]:


class NeuralNetwork:   
    def __init__(self, layers, learning_rate,activation='tanh'):  
        """  
        :param layers: A list containing the number of units in each layer.
        Should be at least two values  
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"  
        """  
        if activation == 'logistic':  
            self.activation = logistic  
            self.activation_deriv = logistic_derivative  
        elif activation == 'tanh':  
            self.activation = tanh  
            self.activation_deriv = tanh_deriv
        self.learning_rate=learning_rate
        
        self.weights = []  
        for i in range(1, len(layers) - 1):  
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)  
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)
    
    
    def fit(self, X, y,epochs=10000):         
        X = np.atleast_2d(X)         
        temp = np.ones([X.shape[0], X.shape[1]+1])         
        temp[:, 0:-1] = X  # adding the bias unit to the input layer         
        X = temp         
        y = np.array(y)
    
        for k in range(epochs):  
            i = np.random.randint(X.shape[0])  
            a = [X[i]]
    
            for l in range(len(self.weights)):  #going forward network, for each layer
                a.append(self.activation(np.dot(a[l], self.weights[l])))  #Computer the node value for each layer (O_i) using activation function
            error = y[i] - a[-1]  #Computer the error at the top layer
            deltas = [error * self.activation_deriv(a[-1])] #For output layer, Err calculation (delta is updated error)
            
            #Staring backprobagation
            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer 
                #Compute the updated error (i,e, deltas) for each node going from top layer to input layer 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))  
            deltas.reverse()  
            for i in range(len(self.weights)):  
                layer = np.atleast_2d(a[i])  
                delta = np.atleast_2d(deltas[i])  
                self.weights[i] += self.learning_rate * layer.T.dot(delta)
                
                
    def predict(self, x):         
        x = np.array(x)         
        temp = np.ones(x.shape[0]+1)         
        temp[0:-1] = x         
        a = temp         
        for l in range(0, len(self.weights)):             
            a = self.activation(np.dot(a, self.weights[l]))         
        return a
    def predicts(self,X):
        predictions=[]
        for i in X:
            predictions.append(self.predict(i))
        return predictions


# In[ ]:


import math
params=[0.001,0.01,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,0.6]
RMSEs=[]
for param in params:
    nn = NeuralNetwork([12,2,1],param,'tanh')
    nn.fit(X_train,y_train)
    y_predict=nn.predicts(X_train)
    training_root_mean_squared_error=math.sqrt(metrics.mean_squared_error(y_predict,y_train))
    RMSEs.append(training_root_mean_squared_error)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(params, RMSEs)
plt.title("learning_rate vs RMSE");

