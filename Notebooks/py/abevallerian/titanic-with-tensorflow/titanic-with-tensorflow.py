#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with Tensorflow
# 
# I know it is quite overkill to use Tensorflow for this task, but I just learned using Tensorflow and I want to apply what I've learned in this task. Basically, I'm going to build Logistic Regression using Tensorflow. So, let's begin!
# 
# First, I start importing the libraries and loading the data.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## Preprocessing the Data
# Let's just take a quick view of the data.

# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


test.head()


# In[ ]:


test.describe()


# The goal of this project is to predict whether a passenger survives. Therefore, I don't think that *Name*, *Ticket*, *Fare*, and *Embarkment* are related to survival. Just delete those columns from the table. Moreover, there are also several *NaN* in the table. Replace those *NaN*s with 0.

# In[ ]:


del train['Name']
del train['Ticket']
del train['Fare']
del train['Embarked']


# In[ ]:


train = train.fillna(value=0.0)


# 1. First, let's preprocess the *Sex*. Just replace it with 0 (Female) or 1 (Male).
# 2. Then, let's handle the *Age*. Since the age is categorical data, I group the age 8 groups: *NaN*, 0-10, 10-20, ..., 70-80. From the desribe above, it's shown that the maximum age is 80.
# 3. *Cabin* is quite interesting. It is stored in string. I think the format is written as *Cabin Section + Cabin Number*. I'm only interested in obtaining the *Cabin Section*.

# In[ ]:


for i in range(train.shape[0]):
    if train.at[i, 'Sex'] == 'male':
        train.at[i, 'Sex'] = 1
    else:
        train.at[i, 'Sex'] = 0


# In[ ]:


train['Age_group'] = 0
for i in range(train.shape[0]):
    for j in range(70, 0, -10):
        if train.at[i, 'Age'] > j:
            train.at[i, 'Age_group'] = int(j/10)
            break
del train['Age'] # it's unnecessary anymore


# In[ ]:


print(list(set(train['Cabin'].values))[:10]) # sample of 'Cabin' values
train['Cabin_section'] = '0'
for i in range(train.shape[0]):
    if train.at[i, 'Cabin'] != 0:
        train.at[i, 'Cabin_section'] = train.at[i, 'Cabin'][0]
CABIN_SECTION = list(set(train['Cabin_section'].values)) # will be reused for test data
print(CABIN_SECTION) # 'Cabin_Section' values
for i in range(train.shape[0]):
    train.at[i, 'Cabin_section'] = CABIN_SECTION.index(train.at[i, 'Cabin_section'])
del train['Cabin'] # it's unnecessary anymore


# I've done with the preprocessing. Here is the result.

# In[ ]:


train.head()


# What's next is preparing the numpy array for the input of Tensorflow. I need to convert the categorical data (*Pclass*, *Age_group*, and *Cabin_section*) into *one hot* array using np.eye. Then, divide the data into training and dev set.

# In[ ]:


pclass = np.eye(train['Pclass'].values.max()+1)[train['Pclass'].values]
age_group = np.eye(train['Age_group'].values.max()+1)[train['Age_group'].values]
cabin_section = np.eye(train['Cabin_section'].values.max()+1)                     [train['Cabin_section'].values.astype(int)] # prevent IndexError


# In[ ]:


X = train[['Sex', 'SibSp', 'Parch']].values
X = np.concatenate([X, age_group], axis=1)
X = np.concatenate([X, pclass], axis=1)
X = np.concatenate([X, cabin_section], axis=1)
X = X.astype(float)

y = train['Survived'].values
y = y.astype(float).reshape(-1, 1)


# In[ ]:


X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1, random_state=0)


# In[ ]:


print(X_train.shape, y_train.shape)


# Repeat the preprocessing for the test data as well.

# In[ ]:


del test['Name']
del test['Ticket']
del test['Fare']
del test['Embarked']

test = test.fillna(value=0.0)

test['Age_group'] = 0
test['Cabin_section'] = '0'
for i in range(test.shape[0]):
    if test.at[i, 'Sex'] == 'male':
        test.at[i, 'Sex'] = 1
    else:
        test.at[i, 'Sex'] = 0

    for j in range(70, 0, -10):
        if test.at[i, 'Age'] > j:
            test.at[i, 'Age_group'] = int(j/10)
            break

    if test.at[i, 'Cabin'] != 0:
        test.at[i, 'Cabin_section'] = test.at[i, 'Cabin'][0]
    test.at[i, 'Cabin_section'] = CABIN_SECTION.index(test.at[i, 'Cabin_section'])

del test['Cabin'] # it's unnecessary anymore
del test['Age'] # it's unnecessary anymore


# In[ ]:


test.head()


# In[ ]:


pclass_test = np.eye(test['Pclass'].values.max()+1)[test['Pclass'].values]
age_group_test = np.eye(test['Age_group'].values.max()+1)[test['Age_group'].values]
cabin_section_test = np.eye(test['Cabin_section'].values.max()+1)                     [test['Cabin_section'].values.astype(int)] # prevent IndexError

X_test = test[['Sex', 'SibSp', 'Parch']].values
X_test = np.concatenate([X_test, age_group_test], axis=1)
X_test = np.concatenate([X_test, pclass_test], axis=1)
X_test = np.concatenate([X_test, cabin_section_test], axis=1)
X_test = X_test.astype(float)

id_test = test['PassengerId'].values
id_test = id_test.reshape(-1, 1)


# In[ ]:


print(X_test.shape, id_test.shape)


# ## Building the Neural Network
# Let's start by defining the hyperparameters

# In[ ]:


seed = 7 # for reproducible purpose
input_size = X_train.shape[1] # number of features
learning_rate = 0.001 # most common value for Adam
epochs = 8500 # I've tested previously that this is the best epochs to avoid overfitting


# The Logistic Regression looks like this: W1\*X + b1 = pred, where \* is the matrix multiplication and sigmoid is used as activation function at the output layer. *Cross Entropy* and *Adam Optimizer* are used as the loss function and optimizer.

# In[ ]:


graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(seed)
    np.random.seed(seed)

    X_input = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X_input')
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')
    
    W1 = tf.Variable(tf.random_normal(shape=[input_size, 1], seed=seed), name='W1')
    b1 = tf.Variable(tf.random_normal(shape=[1], seed=seed), name='b1')
    sigm = tf.nn.sigmoid(tf.add(tf.matmul(X_input, W1), b1), name='pred')
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input,
                                                                  logits=sigm, name='loss'))
    train_steps = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    pred = tf.cast(tf.greater_equal(sigm, 0.5), tf.float32, name='pred') # 1 if >= 0.5
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_input), tf.float32), name='acc')
    
    init_var = tf.global_variables_initializer()


# In[ ]:


train_feed_dict = {X_input: X_train, y_input: y_train}
dev_feed_dict = {X_input: X_dev, y_input: y_dev}
test_feed_dict = {X_input: X_test} # no y_input since the goal is to predict it


# ## Training the Network
# Let's start the training. I initialize the session and variables first and start the training. During training, the loss and accuracy are printed.

# In[ ]:


sess = tf.Session(graph=graph)
sess.run(init_var)


# In[ ]:


cur_loss = sess.run(loss, feed_dict=train_feed_dict)
train_acc = sess.run(acc, feed_dict=train_feed_dict)
test_acc = sess.run(acc, feed_dict=dev_feed_dict)
print('step 0: loss {0:.5f}, train_acc {1:.2f}%, test_acc {2:.2f}%'.format(
                       cur_loss, 100*train_acc, 100*test_acc))
for step in range(1, epochs+1):
    sess.run(train_steps, feed_dict=train_feed_dict)
    cur_loss = sess.run(loss, feed_dict=train_feed_dict)
    train_acc = sess.run(acc, feed_dict=train_feed_dict)
    test_acc = sess.run(acc, feed_dict=dev_feed_dict)
    if step%100 != 0: # print result every 100 steps
        continue
    print('step {3}: loss {0:.5f}, train_acc {1:.2f}%, test_acc {2:.2f}%'.format(
                       cur_loss, 100*train_acc, 100*test_acc, step))


# ## Evaluating the Network
# Actually the network performance is not very good (only around 80%). Finally, I need to prepare the prediction.

# In[ ]:


y_pred = sess.run(pred, feed_dict=test_feed_dict).astype(int)
prediction = pd.DataFrame(np.concatenate([id_test, y_pred], axis=1),
                          columns=['PassengerId', 'Survived'])


# In[ ]:


prediction.head()


# ## Takeaways
# 1. I think I'm not doing enough Exploratory Data Analysis, which I think very crucial in beginning the project.
# 2.  80% accuracy in train and dev set is not very good actually. I think other models such as Random Forest will produce better accuracy.
# 3. Even if Logistic Regression should be used, using Tensorflow is not very efficient. There are many build-in libraries for Logistic Regression (e.g. Scikit-Learn).

# Any feedbacks are very welcomed!

# In[ ]:




