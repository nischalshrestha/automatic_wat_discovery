#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(u'pwd')
#%reset
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


# Import TensorFlow and some useful stuff... and the data, of course ;)

# In[2]:


import tensorflow as tf
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset
from keras.layers import Activation, Dense, Dropout, Input
import math
from tensorflow.python.framework import ops
seed = 110

Titanic_train_dataframe = pd.read_csv('../input/train.csv')
Titanic_test_dataframe = pd.read_csv('../input/test.csv')


# **How does the data look like?** below to see

# In[ ]:


Titanic_train_dataframe.describe(include='all')
#Titanic_test_dataframe.describe(include='all')
#Titanic_train_dataframe[Titanic_train_dataframe['Cabin'].str.findall(' ')]


# In[3]:


Last_rep = {}    #to be the common source of ground truth of all people of last name, so we can introduce if survival depends on family size only(not just
                 # the sibblings count).
train = pd.Series(Titanic_train_dataframe.loc[:, 'Name'])
test = pd.Series(Titanic_test_dataframe.loc[:, 'Name'])
full = pd.Series(train).append(test)
a = 0
for i in pd.Series(full.str.split(',').str.get(0)) :
    Last_rep[str(i)] = sum((full.str.split(',').str.get(0)) == i)
    #Titanic_dataframe.loc[a, 'Last_Rep'] = Last_rep[str(i)]
    a +=1
print(Last_rep)


# In[4]:


def process_df(Titanic_dataframe) :
    #modifying the inputs to be numerical:
    # Name processing and counting Cabins, later version can make.
    #Last_rep = { }        #source of truth for last name repetition across all examples, moved to be holistic in the above Jupiter Notebook cell.
    a = 0
    for i in pd.Series(Titanic_dataframe.loc[:, 'Name'].str.split(',').str.get(0)) :
        #Last_rep[str(i)] = sum((pd.Series(Titanic_dataframe.loc[:, 'Name'].str.split(',').str.get(0)) == i)) # not needed since the above cell populated
        #the Last_rep dictionary.
        Titanic_dataframe.loc[a, 'Last_Rep'] = Last_rep[str(i)]
        #print(i)                                                             #Debugging step
        #print(Titanic_dataframe.loc[a, 'Last_Rep'])                          #Debugging step
        if pd.isnull(Titanic_dataframe.loc[a, 'Cabin']) == True :
            Titanic_dataframe.loc[a, 'Cabin Count'] = 0
        else :
            Titanic_dataframe.loc[a, 'Cabin Count'] = Titanic_dataframe.loc[a, 'Cabin'].count(' ') + 1
        a += 1
    #Embarked: S=>0, C=>1, Q=>2, NaN=>3 (NaN is covered in the previous step)
    Titanic_dataframe.loc[:, 'Embarked'] = Titanic_dataframe.loc[:, 'Embarked'].str.replace('S', '0')
    Titanic_dataframe.loc[:, 'Embarked'] = Titanic_dataframe.loc[:, 'Embarked'].str.replace('C', '1')
    Titanic_dataframe.loc[:, 'Embarked'] = Titanic_dataframe.loc[:, 'Embarked'].str.replace('Q', '2')
    Titanic_dataframe.loc[:, 'Embarked'] = Titanic_dataframe.loc[:, 'Embarked'].fillna('3')
    Titanic_dataframe.loc[:, 'Embarked'] = pd.to_numeric(Titanic_dataframe.loc[:, 'Embarked'])
    #Sex: male=0, female = 1
    Titanic_dataframe.loc[:, 'Sex'] = Titanic_dataframe.loc[:, 'Sex'].str.replace('female', '1')
    Titanic_dataframe.loc[:, 'Sex'] = Titanic_dataframe.loc[:, 'Sex'].str.replace('male', '0')
    Titanic_dataframe.loc[:, 'Sex'] = pd.to_numeric(Titanic_dataframe.loc[:, 'Sex'])
    #Age: fill NaN with distinguished value
    Titanic_dataframe.loc[:, 'Age'] = Titanic_dataframe.loc[:, 'Age'].fillna(value=0)
    #Fare: fill NaN with distinguished value
    Titanic_dataframe.loc[:, 'Fare'] = Titanic_dataframe.loc[:, 'Fare'].fillna(value=0)
    return Titanic_dataframe
    #number of cabins

Titanic_train_dataframe_p = process_df(Titanic_train_dataframe)
Titanic_test_dataframe_p = process_df(Titanic_test_dataframe)
#Have a look at the data
Titanic_train_dataframe_p.head()
#return Last_rep


# Now we need to split the training data into Training/Validation/Test sets. Since the test set is given as a sepatarate CSV, then the data in train.csv can use the supplied data in 'train.csv' to be only for Training and Validation. let's set the ratio to be 80:20 (train vs validation). let's split the data now.
# 
# Notes:
# 
# 1- we know from the description (couple of steps above) that we have 891 training/dev samples.
# 
# 2- 80% of 890 will be used for training, the rest will be used for validation

# In[5]:


x = .8*890         # number of training data samples
y = 891 - x        # number of validation data samples
shuffled_Titanic_dataframe = Titanic_train_dataframe_p.iloc[np.random.permutation(len(Titanic_train_dataframe_p))] #to reshuffle the input dataframe. can be reversed by: df_shuffled.reset_index(drop=True)
Training_data = shuffled_Titanic_dataframe.head(int(x))
Validation_data = shuffled_Titanic_dataframe.tail(int(y))
Test_data = Titanic_test_dataframe_p
Training_data.describe; Validation_data.describe(include='all')


# In[ ]:


Test_data.describe(include='all')
#Test_data.head(15)


# Getting rid of the text.
# Text is a problem to handle, so we can try to get rid of text. the idea is to check the last name of each passenger, see how many times it is repeated and store that for later, so we can judge if being part of particular families helped. let's create that function.

# In[6]:


X_train_raw = np.array(pd.DataFrame(Training_data.take([0,2,4,5,6,7,9,11,12,13],axis=1)))
Y_train = np.array(Training_data.take([1],axis=1))
X_dev_raw = np.array(Validation_data.take([0,2,4,5,6,7,9,11,12,13],axis=1))
Y_dev = np.array(Validation_data.take([1],axis=1))
X_test_raw = np.array(pd.DataFrame(Test_data.take([0,1,3,4,5,6,8,10,11,12],axis=1)))
print(str(X_train_raw.shape) + ' '+ str(Y_train.shape) +' '+ str(X_dev_raw.shape) +' '+ str(Y_dev.shape)+ ' '+str(X_test_raw.shape))


# In[7]:


#prepare for normalizing
mu_train = np.mean(X_train_raw,axis=0,keepdims=True)
sigma_train = np.std(X_train_raw,axis=0,keepdims=True)
mu_dev = np.mean(X_dev_raw,axis=0,keepdims=True)
sigma_dev = np.std(X_dev_raw,axis=0,keepdims=True)
print(str(mu_train.shape) + ' '+ str(sigma_train.shape) +' '+ str(mu_dev.shape) +' '+ str(sigma_dev.shape))


# In[8]:


#normalizing
X_train_norm = (X_train_raw-mu_train)/sigma_train**2
X_dev_norm = (X_dev_raw-mu_train)/sigma_train**2
X_test_norm =(X_test_raw-mu_train)/sigma_train**2
X_train = X_train_norm.T
X_dev = X_dev_norm.T
X_test = X_test_norm.T
Y_train = Y_train.T
Y_dev = Y_dev.T
#X_dev.shape
print(str(X_train.shape) + ' '+ str(Y_train.shape) +' '+ str(X_dev.shape) +' '+ str(Y_dev.shape)+' '+ str(X_test.shape))


# In[9]:


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, number of features
    n_y -- scalar, number of classes (from 0 to 1)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    We will use None because the number of examples during test/train is different.
    """
    X = tf.placeholder(tf.float32, shape=(n_x,None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y,None), name='Y')
    return X, Y

X, Y = create_placeholders(10, 1)
print ("X = " + str(X))
print ("Y = " + str(Y))


# In[10]:


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [15,10]
                        b1 : [15, 1]
                        W2 : [15, 15]
                        b2 : [15, 1]
                        W3 : [1, 15]
                        b3 : [1, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(seed)                   
    W1 = tf.get_variable("W1", [15,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [15,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [15,15], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [15,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1,15], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("W3 = " + str(parameters["W3"]))
    print("b3 = " + str(parameters["b3"]))


# In[11]:


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1, name='Z1')                                
    A1 = tf.nn.relu(Z1, name='A1')             
    
    Z2 = tf.add(tf.matmul(W2,A1), b2, name='Z2')                                
    A2 = tf.nn.relu(Z2,name='A2')                                              
    Z3 = tf.add(tf.matmul(W3,A2), b3, name='Z3')      
    Y_hat = tf.round(tf.sigmoid(Z3))
    return Y_hat,Z3

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(10, 1)
    parameters = initialize_parameters()
    Y_hat, Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))
    print('Y_hat = ' + str(Y_hat))


# In[12]:


def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(10,1)
    parameters = initialize_parameters()
    Y_hat,Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))


# In[13]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

mimi = random_mini_batches(X_train, Y_train, mini_batch_size = 32, seed = 0)
len(mimi)
(mim,mom) = mimi[0]
mom.shape



# In[14]:


def model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    # Initialize parameters
    parameters = initialize_parameters()
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Y_hat, Z3 = forward_propagation(X, parameters)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    # Initialize all the variables
    init = tf.global_variables_initializer()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        #print(Y_hat.eval({X:X_train, parameters:parameters}))
        # Calculate the correct predictions
        #correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        correct_prediction = tf.equal(Y_hat, Y)
        #print(sess.run(correct_prediction, feed_dict={X: X_train, Y: Y_train}))
        #print(Z3.eval(feed_dict={X: X_train, Y: Y_train}))
        #print(Y)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #s accuracy = tf.cast(correct_prediction, "float")
        #print(sess.run(tf.cast(correct_prediction, "float"), feed_dict={X: X_train, Y: Y_train}))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_dev, Y: Y_dev}))
        Y_hat_f = Y_hat.eval({X: X_test})
        return parameters,Y_hat_f


# In[ ]:


parameters,pred = model(X_train, Y_train, X_dev, Y_dev,learning_rate = 0.0001, num_epochs = 100000, minibatch_size = 32, print_cost = True)


# In[ ]:


print(pred)


# In[ ]:


#parameters,_ = model(X_train, Y_train, X_dev, Y_dev,learning_rate = 0.00003, num_epochs = 1000, minibatch_size = 32, print_cost = True)


# In[ ]:


parameters


# In[ ]:


def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [10, None])
    Za1 = tf.add(tf.matmul(W1, x), b1, name='Za1')                                
    Aa1 = tf.nn.relu(Za1, name='Aa1')                                              
    Za2 = tf.add(tf.matmul(W2,Aa1), b2, name='Za2')                                
    Aa2 = tf.nn.relu(Za2,name='Aa2')                                              
    Za3 = tf.add(tf.matmul(W3,Aa2), b3, name='Za3')      
    Y_hat = tf.round(tf.sigmoid(Za3))
        
    sess = tf.Session()
    prediction = sess.run(Y_hat, feed_dict = {x: X})
        
    return prediction


# In[ ]:


X_test.shape


# In[ ]:


test_prediciton = predict(X_test, parameters)


# In[ ]:


print(test_prediciton[0])


# In[ ]:





# In[ ]:


#len(list(Test_data.index))
#pd.Series(test_prediciton[0], index=list(Test_data.index))
submission = pd.DataFrame(Test_data.take([0],axis=1))
submission.loc[:,'Survived'] = pd.Series(test_prediciton[0], index=list(submission.index))
submission[['PassengerId', 'Survived']] = submission[['PassengerId', 'Survived']].astype('int')
#print(submission)
submission.describe()
submission.to_csv('submission_201804062319.csv', index=False)
                        
                        


# In[ ]:



tst= pd.read_csv('submission.csv')
#tst.head()

