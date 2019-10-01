#!/usr/bin/env python
# coding: utf-8

# Try using basic Tensorflow approach. MNIST Basic as reference

# In[ ]:


import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import re
import datetime


# ### Loading Data

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
train_data.head(2)


# In[ ]:


test_data = pd.read_csv('../input/test.csv')
test_data.head(2)


# ### Shaping Data

# In[ ]:


x_train = train_data.drop(['PassengerId','Ticket','Survived'], axis=1)
x_train.head(3)


# In[ ]:


y_train = pd.DataFrame({'Dead':(train_data['Survived']+1)%2,'Survived':train_data['Survived']})
y_train.head(3)


# In[ ]:


x_test = test_data.drop(['PassengerId','Ticket'], axis=1)
x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())
x_test['Age'] = x_test['Age'].fillna(x_test['Age'].mean())
x_test.head(2)


# some functions for data shaping

# In[ ]:


def simplify_ages(df):
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df['Age'], bins, labels=group_names)
    df['Age'] = categories.cat.codes 
    return df

def simplify_cabins(df):
    df['Cabin'] = df['Cabin'].fillna('N')
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])
    df['Cabin'] = pd.Categorical(df['Cabin'])
    df['Cabin'] = df['Cabin'].cat.codes 
    return df

def simplify_fares(df):
    df['Fare'] = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df['Fare'], bins, labels=group_names)
    df['Fare'] = categories.cat.codes 
    return df

def simplify_sex(df):
    df['Sex'] = pd.Categorical(df['Sex'])
    df['Sex'] = df['Sex'].cat.codes 
    return df

def simplify_embarked(df):
    df['Embarked'] = pd.Categorical(df['Embarked'])
    df['Embarked'] = df['Embarked'].cat.codes + 1
    return df

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def simplify_name(df):
    df['Name'] = df['Name'].apply(get_title)
    df['Name'] = df['Name'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Name'] = df['Name'].replace('Mlle', 'Miss')
    df['Name'] = df['Name'].replace('Ms', 'Miss')
    df['Name'] = df['Name'].replace('Mme', 'Mrs')    
    df['Name'] = pd.Categorical(df['Name'])
    df['Name'] = df['Name'].cat.codes + 1
    return df

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = simplify_sex(df)
    df = simplify_embarked(df)
    df = simplify_name(df)
    return df


# In[ ]:


transform_features(x_train)
transform_features(x_test)
x_train.head(2)


# In[ ]:


x_train.count()


# Make simple model with tensorflow's MNIST Basic
# refer: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py

# In[ ]:


x = tf.placeholder("float", [None, 9])
W = tf.Variable(tf.zeros([9, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 2])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)


# Tensorflow session init

# In[ ]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# ### Run train and get results
# ![](http://)Run machine learning.
# Reporting test report for each 50 times, Output csv simulation file for each250 times. Note that I tried 100000 times in my enviroment but set only 1000 for this notebook.

# In[ ]:


for i in range(5000):
    for index, value in x_train.iterrows():
        batch_xs = [value]
        batch_ys = [y_train.loc[index]]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%100 == 0:
        print("loop -> ", i ,", " , end="")
        c=t=0
        for index,value in x_train.iterrows():
            batch_xs = [value]
            dy = sess.run(y, feed_dict={x: batch_xs})
            if train_data.loc[index]['Survived'] == int(np.round(dy[0][1])):
                c = c + 1
            t = t +1
        print(datetime.datetime.today())
        print("try = ", i , ": result = ", c/t)
    if i%500 == 0:
        fn = 'Titanic_' + str(i)  + '_' + str(c/t) + '.csv' 
        f = open(fn,'w')
        f.write('PassengerId,Survived\n')
        for index,value in x_test.iterrows():
            batch_xs = [value]
            dy = sess.run(y, feed_dict={x: batch_xs})
            f.write(str(test_data.loc[index]['PassengerId'])+','+str(int(np.round(dy[0][1])))+'\n') # Survived
        f.close()

