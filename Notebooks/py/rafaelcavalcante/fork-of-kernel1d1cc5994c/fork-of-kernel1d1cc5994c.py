#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.preprocessing import Imputer
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

print(train_df.columns.values)

train_df.head()

train_df.tail()
# Any results you write to the current directory are saved as output.


# In[64]:


train_df.info()


# In[65]:


test_df.info()


# In[66]:


train_df.describe()


# In[67]:


train_df.describe(include=['O'])


# In[ ]:


def drop(data, columns):
    return data.drop(columns, axis=1)

coltodrop = ["PassengerId","Name", "Cabin", "Ticket"]
test_p_id= test_df["PassengerId"]

train_data = drop(train_df, coltodrop)
test_data = drop(test_df, coltodrop)

train_data.head()


# In[69]:


from sklearn.preprocessing import LabelEncoder

def categ_to_int(data, lista, column):
    le = LabelEncoder()
    le.fit(lista)
    data[column]=le.transform(data[column]) 
    return data

train_data = categ_to_int(train_data,["male","female"], "Sex")
test_data = categ_to_int(test_data,["male","female"], "Sex")

train_data.head()


# In[ ]:


train_data["Embarked"]=train_data["Embarked"].fillna('Z')
train_data = categ_to_int(train_data,["Q", "C", "S", "Z"], "Embarked")
test_data = categ_to_int(test_data,["Q", "C", "S", "Z"], "Embarked")
print(train_data["Embarked"].describe())



# In[ ]:


#centralize and normalize data
def nan_rid(data, columns):
    for column in columns:
        imputer=Imputer()
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data

nan_columns = ["Age", "Fare"]


train_data = nan_rid(train_data, nan_columns)
test_data = nan_rid(test_data, nan_columns)
train_data.head()


# In[ ]:


def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


dummy_columns = ["Pclass", "Sex", "Embarked"]
train_data=dummy_data(train_data, dummy_columns)
test_data=dummy_data(test_data, dummy_columns)
train_data.head()

test_data["Embarked_3"]= 0



# In[ ]:


from sklearn.preprocessing import MinMaxScaler

def normalize_data(data, column):
    scaler = MinMaxScaler()
    data[column] = scaler.fit_transform(data[column].values.reshape(-1,1))
    return data

train_data = normalize_data(train_data, "Age")
test_data = normalize_data(test_data, "Age")

train_data = normalize_data(train_data, "Fare")
test_data = normalize_data(test_data, "Fare")

test_data["Embarked_3"]= 0

test_data.head()


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def getDevset(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, dev_x, train_y, dev_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, dev_x, dev_y

train_x, train_y, dev_x, dev_y = getDevset(train_data)


# In[ ]:


from collections import namedtuple

def build_neural_network(hidden_unit1=10,hidden_unit2=10, lmbda=0.1):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training=tf.Variable(True,dtype=tf.bool)
    
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(scale = lmbda)
    fc = tf.layers.dense(inputs, hidden_unit1, activation=None,kernel_initializer=initializer, kernel_regularizer = regularizer)
    fc=  tf.layers.batch_normalization(fc, training=is_training)
    fc= tf.nn.relu(fc)
    fc2 = tf.layers.dense(fc, hidden_unit2, activation=None,kernel_initializer=initializer, kernel_regularizer = regularizer)
    fc2= tf.layers.batch_normalization(fc2, training=is_training)
    fc2=tf.nn.relu(fc)
    
    
    logits = tf.layers.dense(fc2, 1, activation=None)
    cross_entropy =  tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    
    cost = tf.reduce_mean(cross_entropy)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


# In[ ]:


def get_batch(data_x,data_y,batch_size=32):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        
        yield batch_x,batch_y


# In[ ]:


def TrainingParameters(hidden_unit1,hidden_unit2, lmbda, epochs, learning_rate,batch_size):

    model = build_neural_network(hidden_unit1,hidden_unit2, lmbda)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for batch_x,batch_y in get_batch(train_x,train_y,batch_size):
                feed = {model.inputs: train_x,
                        model.labels: train_y,
                        model.learning_rate: learning_rate,
                        model.is_training:True
                       }

                train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)

        feed = {model.inputs: dev_x,
                model.labels: dev_y,
                model.is_training:False
                }
        val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
        
        saver.save(sess, "./titanic.ckpt")
        
        values = [hidden_unit1,hidden_unit2, lmbda, epochs, learning_rate,batch_size, train_loss,train_acc, val_loss, val_acc]
        
        return values


# In[ ]:


# test different hyper parameters to find the best model

def test_hyperparameters():
    import random as r

    learning_rate = 0.001
    hidden_unit1 = np.arange(10, 20, 5)
    hidden_unit2 = np.arange(3, 10, 2)
    epochs = np.arange(100, 200, 50)
    batch_size = np.array([16,32,64])
    colectValues = []


    batch = [16,32,64]

    for x in range(0, 100):
        h1 = r.randint(10, 20)
        h2 = r.randint(5, 15)
        ep = r.randint(1,5) * 100
        bs = r.choice(batch)
        ex = -4 *np.random.rand()
        lmbda = 10**ex
        Results  =  TrainingParameters(int(h1),int(h2),lmbda,int(ep),learning_rate,int(bs))
        print (Results)
        colectValues.append(Results)





# In[102]:


#test_hyperparameters()
finalresult = TrainingParameters(18, 8, 0.0004046098616621213, 300, 0.001, 64)

print(finalresult)


# In[104]:



model = build_neural_network(18, 8, 0.0004046098616621213)

restorer=tf.train.Saver()
with tf.Session() as sess:
    restorer.restore(sess,"./titanic.ckpt")
    feed={
        model.inputs:test_data,
        model.is_training:False
    }
    test_predict=sess.run(model.predicted,feed_dict=feed)
    


from sklearn.preprocessing import Binarizer
binarizer=Binarizer(0.5)
test_predict_result=binarizer.fit_transform(test_predict)
test_predict_result=test_predict_result.astype(np.int32)
test_predict_result[:10]


# In[105]:


passenger_id=test_p_id.copy()
evaluation=passenger_id.to_frame()
evaluation["Survived"]=test_predict_result
evaluation[:10]


# In[106]:


print(evaluation)
evaluation.to_csv("evaluation_submission.csv",index=False)


# In[ ]:




