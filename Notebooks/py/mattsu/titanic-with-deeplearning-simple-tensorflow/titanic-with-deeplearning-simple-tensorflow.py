#!/usr/bin/env python
# coding: utf-8

# ## Introduction: 
# This kernel use the raw tensorflow code, to learn the basics building blocks.
# From the result, we can see tensorflow neural networks can't achieve higher accuracy than 80%.
# Why? My conclusion is the data size is too small that DNN can't work well at all.
# 
# However, here I represent a typical flow for tensorflow.
# 
# As the traning dataset is **too small**, there're obvious **overfitting**
# have to try to avoid overfitting:
# - decrease features
# - add more data
# - regularization
# 
# **My found is that, feature engineering is still the most important if dataset is so small.**

# ## Here's the main structure
# 1. feature engineering: throw away useless feature, and transforming usable feature
# 2. data cleaning and normalization (fill N/A, and separate categories)
# 3. train/test separation
# 4. neural network design
# 5. train and predict

# In[51]:


import numpy as np 
import pandas as pd 
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


# In[52]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# start with some EDA
train.head()
print(train['Embarked'].unique(), train['Pclass'].unique())
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[53]:


# remove useless feature
for df in [train, test]:
    df.drop(labels=["PassengerId", "Cabin", "Name", "Ticket"], axis=1, inplace=True)


# In[54]:


# fill missing value
# another way is to use sklearn imputation to fill na
for df in [train, test]:
    for col in ["Age", "Fare"]:
        df[col] = df[col].fillna(np.mean(df[col]))


# In[55]:


# data normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
for df in [train, test]:
    for col in ["Age", "Fare"]:
        x = df[[col]].values.astype(float)
        df[col] = min_max_scaler.fit_transform(x)


# In[56]:


# transform string
for df in [train, test]:
    # remove duplicate features to avoid overfitting
    #df['is_male'] = np.where(df['Sex']=="male", 1, 0)
    df['is_female'] = np.where(df['Sex']=="female", 1, 0)
    df['EmbarkedS'] = np.where(df['Embarked']=="S", 1, 0)
    df['EmbarkedC'] = np.where(df['Embarked']=="C", 1, 0)
    df['EmbarkedQ'] = np.where(df['Embarked']=="Q", 1, 0)
    df['Pclass1'] = np.where(df['Pclass']==1, 1, 0)
    df['Pclass2'] = np.where(df['Pclass']==2, 1, 0)
    df['Pclass3'] = np.where(df['Pclass']==3, 1, 0)
    df['is_single'] = np.where(np.logical_and(df['SibSp']==0, df['Parch']==0), 1, 0)

# then remove transformed columns
for df in [train, test]:
    df.drop(labels=["Sex", "Embarked", 'Pclass'], axis=1, inplace=True)


# In[57]:


# load up train/validation set! 
train_size = int(train.shape[0] * 0.85)

train_dataset = train[:train_size]
val_dataset = train[train_size:]

X_train = train_dataset.drop(labels=["Survived"], axis=1).values
Y_train = train_dataset["Survived"].values

X_val = val_dataset.drop(labels=["Survived"], axis=1).values
Y_val = val_dataset["Survived"].values

input_size = len(train_dataset.columns) - 1  # number of final features 


# In[58]:


X_train = X_train.reshape((X_train.shape[1], X_train.shape[0]))
X_val = X_val.reshape((X_val.shape[1], X_val.shape[0]))

Y_train = Y_train.reshape((1, Y_train.shape[0]))
Y_val = Y_val.reshape((1, Y_val.shape[0]))


# In[59]:


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    return X, Y

def initialize_parameters():
    # when I increase this to 2 times of input, the cost decrease rapidly
    # when I increase this to 1000, boom! train acc get to 97%, but test acc is below 60%
    output_size = 1
    l1_size = 64 #int(input_size * 2) 
    l2_size = 64 #int(input_size * 1)
    l3_size = output_size#int(input_size * 1)
    l4_size = output_size
    
    W1 = tf.get_variable("W1", [l1_size, input_size],
                         initializer=tf.contrib.layers.xavier_initializer()) # seed=1
    b1 = tf.get_variable("b1", [l1_size, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [l2_size, l1_size], 
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [l2_size, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [l3_size, l2_size], 
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [l3_size, 1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [l4_size, l3_size], 
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [l4_size, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4
                 }

    return parameters


# In[60]:


def forward_propagation(X, parameters):
    """
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    keep_prob = 0.7 #0.3  # if dropout too many, the predict result will be always 0
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.dropout(tf.nn.relu(Z1), keep_prob=keep_prob)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.dropout(tf.nn.relu(Z2), keep_prob=keep_prob)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.nn.dropout(tf.nn.relu(Z3), keep_prob=keep_prob)  
    Z4 = tf.add(tf.matmul(W4, A3), b4)  

    return Z3

def forward_propagation_for_predict(X, parameters):
    """
    Returns: Z3 -- the output of the last LINEAR unit
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.nn.relu(Z3)  # A2 = relu(Z2)
    Z4 = tf.add(tf.matmul(W4, A3), b4)  # Z3 = np.dot(W3,Z2) + b3

    return Z3


# In[61]:


def compute_cost(Z3, Y):
    """
    Z3 -- output of forward propagation (output of the last LINEAR unit)
    Y -- "true" labels vector placeholder, same shape as Z3
    Return: cost - Tensor of the cost function
    """
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


# In[62]:


def model(X_train, Y_train, X_val, Y_val, learning_rate=0.0001,
          num_epochs=1500, print_cost=True):
    """
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs, val_losses = [], []  # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    val_cost = compute_cost(forward_propagation_for_predict(X, parameters), Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            _, epoch_cost = sess.run([optimizer, cost],
                                     feed_dict={X: X_train, Y: Y_train})

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                epoch_val_loss = sess.run(val_cost, feed_dict={X: X_val, Y: Y_val})
                if epoch % 100 == 0:
                    print ("Loss after epoch %i: training[%f] | dev[%f]" % (epoch, epoch_cost, epoch_val_loss))

                val_losses.append(epoch_val_loss)
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.plot(np.squeeze(val_losses))
        plt.legend(["training", "test"])
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        y_hat = tf.cast(tf.greater(tf.sigmoid(Z3),0.5), tf.float32)
        correct_prediction = tf.equal(y_hat, Y)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy: ", "{:3.2f}%".format(100*accuracy.eval({X: X_train, Y: Y_train})))
        print ("Test Accuracy:", "{:3.2f}%".format(100*accuracy.eval({X: X_val, Y: Y_val})))

        return parameters


# In[63]:


parameters = model(X_train, Y_train, X_val, Y_val,learning_rate=0.001, num_epochs=2000)


# In[64]:



def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
             "W4": W4,
              "b4": b4}

    x = tf.placeholder("float", [input_size, None])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.sigmoid(z3) 

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction


# In[65]:


# load up test set
final_test = test.values.T
y_pred = predict(final_test, parameters)

y_final = (y_pred > 0.5).astype(int).ravel()  #.reshape(X_val.shape[0])


# In[66]:


df_test = pd.read_csv("../input/test.csv")
output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})
# output["Survived"].unique()
surv_num = sum(output["Survived"] != 0) / len(output)
print(f"Survive ratio: {surv_num}")


# In[67]:


output.to_csv('prediction-ann.csv', index=False)
output


# In[ ]:




