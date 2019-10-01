#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tflearn
import pandas as pd
import tensorflow as tf
tf.reset_default_graph()


# In[ ]:


def load_train_data(fname): 
    ds = pd.read_csv(fname)
    labels = ds.Survived
    ds=ds[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    ds=pd.get_dummies(ds)
    age_mean = ds.Age.mean()
    age_std = ds.Age.std()
    fare_mean = ds.Fare.mean()
    fare_std = ds.Fare.std()
    ds.Age = ds.Age - age_mean
    ds.Age = ds.Age / age_std
    ds.Fare = ds.Fare - fare_mean
    ds.Fare = ds.Fare / fare_std
    data = ds.fillna(0)
    data = data.as_matrix()
    labels = labels.as_matrix()
    labels = labels.reshape(labels.shape[0],1)
    died = 1 * np.squeeze(labels == 0)
    survived = 1 * np.squeeze(labels == 1)
    labels = np.dstack((died, survived)).squeeze()
    return data, labels

def load_test_data(fname): 
    ds = pd.read_csv(fname)
    pax_id = ds.PassengerId
    ds=ds[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    ds=pd.get_dummies(ds)
    age_mean = ds.Age.mean()
    age_std = ds.Age.std()
    fare_mean = ds.Fare.mean()
    fare_std = ds.Fare.std()
    ds.Age = ds.Age - age_mean
    ds.Age = ds.Age / age_std
    ds.Fare = ds.Fare - fare_mean
    ds.Fare = ds.Fare / fare_std
    data = ds.fillna(0)
    data = data.as_matrix()
 
    return data, pax_id


# In[ ]:


data, labels = load_train_data('../input/train.csv')
data_t, pax_id = load_test_data('../input/test.csv')


# In[ ]:


# Build neural network
net = tflearn.input_data(shape=[None, 10])
net = tflearn.fully_connected(net, 64, activation='relu')
net = tflearn.fully_connected(net, 32, activation='relu')
net = tflearn.dropout(net, 0.75)
net = tflearn.fully_connected(net, 2, activation='softmax')
# With TFLearn estimators
adam = tflearn.optimizers.Adam(learning_rate=0.001, beta1=0.99)
regression = tflearn.regression(net, optimizer=adam)


# In[ ]:


# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=100, batch_size=32, show_metric=True)


# In[ ]:


a = data[5]
a=a.reshape(1,10)
p=model.predict(data_t)
p=np.argmax(p,axis=1)
result = pd.DataFrame(np.dstack((pax_id, p)).squeeze())
result.to_csv('result.csv', header=('PassengerId','Survived'),index=False)


# In[ ]:





# In[ ]:




