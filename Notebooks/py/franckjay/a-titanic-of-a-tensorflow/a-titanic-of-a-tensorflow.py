#!/usr/bin/env python
# coding: utf-8

# I wanted to test my TensorFlow skills on real data sets, and thought that the Titanic Dataset 
# would be a nice start. If you have any questions, please submit them and I will try to answer. If you have any suggestions, I would be very pleased to see those as well. Let us load the data and get our house in order.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

titanic = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) #Fillna fills columns with N/A numbers. Here we use the median values
print('hi')
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0 #Replaces all male values with 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic['Embarked']=titanic['Embarked'].fillna('S') # The most common value!
titanic.loc[titanic['Embarked']=='S', 'Embarked'] = 0
titanic.loc[titanic['Embarked']=='C', 'Embarked'] = 1
titanic.loc[titanic['Embarked']=='Q', 'Embarked'] = 2

#Transform the test data in the same way
test["Age"] = test["Age"].fillna(test["Age"].median()) #Fillna fills columns with N/A numbers. Here we use the median values
test.loc[test["Sex"] == "male", "Sex"] = 0 #Replaces all male values with 0
test.loc[test["Sex"] == "female", "Sex"] = 1
test['Embarked']=test['Embarked'].fillna('S') # The most common value!
test.loc[test['Embarked']=='S', 'Embarked'] = 0
test.loc[test['Embarked']=='C', 'Embarked'] = 1
test.loc[test['Embarked']=='Q', 'Embarked'] = 2


#Just to check our data loaded correctly:
print (titanic.head())

#Just to check our TensorFlow is working.
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
sess.run(hello)


# The above looks like it successfully loaded some data, but we need to start getting the data into a usable format.
# That means filling NaNs in, changing categorical variables, and general mucking about:
# 

# We can do a lot more than this in the future, but this is a barebones version where we can start working on the data to see if TensorFlow works with this data set. I am going to start with some easy examples, because that is just what I do Here is a good Linear Regression intro: http://www.kdnuggets.com/2016/08/gentlest-introduction-tensorflow-part-1.html or  http://www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html. 

# In[ ]:


#from sklearn import metrics
#import skflow
#classifier = skflow.TensorFlowLinearClassifier(n_classes=2)
#classifier.fit(titanic.data, titanic.target)
#score = metrics.accuracy_score(titanic.target, classifier.predict(titanic.data))
#print("Accuracy: %f" % score)
print('skflow is not implemented unfortunately to test the above out.')

#From http://www.kdnuggets.com/2016/09/urban-sound-classification-neural-networks-tensorflow.html/2
nIter = 1000 #How many times should we run this?
nDim = titanic.shape[1] # 12 in this cases
nClasses = 1# How many classes do we have in our model?
nHidden1 = 100 #First hidden layer size
nHidden2 = 80 #Second
SDEV = 1 / np.sqrt(nDim)#Standard Deviation
LR = 0.01 #Learning Rate of our model


# In[ ]:


#Taken from : https://www.kaggle.com/dysonlin/titanic/tensorflow/run/807342

nIter = 40000
LR = 0.1

trainX=titanic[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
trainY=titanic[['Survived']]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
 
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Let's train the model
feature_count = trainX.shape[1]
x = tf.placeholder('float', shape=[None, feature_count], name='x')
y_ = tf.placeholder('float', shape=[None, 1], name='y_')

print(x.get_shape())

nodes = 200

w1 = weight_variable([feature_count, nodes])
b1 = bias_variable([nodes])
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = weight_variable([nodes, 1])
b2 = bias_variable([1])
y = tf.nn.sigmoid(tf.matmul(l1, w2) + b2)

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.maximum(0.01, y)) + (1.0 - y_)*tf.log(tf.maximum(0.00001, 1.0-y)))
reg = 0.01 * (tf.reduce_mean(tf.square(w1)) + tf.reduce_mean(tf.square(w2)))

predict = (y > 0.5)

correct_prediction = tf.equal(predict, (y_ > 0.5))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
                              
                              

train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy + reg)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(nIter):
    feed={x:trainX, y_:trainY}
    sess.run(train_step, feed_dict=feed)
    if i % 1000 == 0 or i == nIter-1:
        print('{} {} {:.2f}%'.format(i, sess.run(cross_entropy, feed_dict=feed), sess.run(accuracy, feed_dict=feed)*100.0))

