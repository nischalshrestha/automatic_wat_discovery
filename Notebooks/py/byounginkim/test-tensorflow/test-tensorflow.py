#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# load data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_data.head(10)


# In[ ]:


test_data.head(10)


# In[ ]:


# sex to number
sex_code = train_data.Sex.astype('category').cat.codes
sex_code_pd = pd.DataFrame({'sex_code': sex_code})

train_data = train_data.join(sex_code_pd)


# In[ ]:


# make input data
x_data = train_data[['sex_code', 'Age']]
y_data = train_data[['Survived']]


# In[ ]:


nodes = 200
learing_rate = 0.1
epoch_num = 15
batch_size = 40000

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Train
feature_count = x_data.shape[1]

x = tf.placeholder(tf.float32, shape=[None, feature_count], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')

w1 = weight_variable([feature_count, nodes])
b1 = bias_variable([nodes])
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w1 = weight_variable([feature_count, nodes])
b1 = bias_variable([nodes])
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = weight_variable([nodes, 1])
b2 = bias_variable([1])
y = tf.nn.sigmoid(tf.matmul(l1, w2)+ b2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.maximum(0.00001, y)) + (1.0 - y_) * tf.log(tf.maximum(0.00001, 1.0-y)))
reg = 0.01 * (tf.reduce_mean(tf.square(w1)) + tf.reduce_mean(tf.square(w2)))

predict = (y > 0.5)
correct_prediction = tf.equal(predict, (y_ > 0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(learing_rate).minimize(cross_entropy + reg)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(batch_size):
    feed = {x:x_data, y_:y_data}
    sess.run(train_step, feed_dict=feed)
    
    if i % 1000 == 0 or i == batch_size - 1:
         print('{} {} {:.2f}%'.format(i, sess.run(cross_entropy, feed_dict=feed), sess.run(accuracy, feed_dict=feed)*100.0))


# In[ ]:




