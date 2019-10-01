#!/usr/bin/env python
# coding: utf-8

# # Using a Tensorflow Custom Estimator to classify Titanic dataset

# My focus here is just show a basic approach of a Custom Estimator using Google's Open Source TensorFlow library.
# 
# The TensorFlow team developed the Estimator API to make the library more accessible to the everyday developer. This high level API provides a common interface to train(...) models, evaluate(...) models, and predict(...) outcomes of unknown cases similar to (and influenced by) the popular Sci-Kit Learn library, which is accomplished by implementing a common interface for various algorithms. Some functions used here are copied from google tensorflow. 

# ### Load data after feat. engineering and cleanning data

# In[18]:


import pandas as pd
import numpy as np


# I did the feature engineering and cleaning step separately. If want to see more details please, see here: [ Titanic Best Working Classfier:](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier) by Sina

# In[19]:


train = pd.read_csv('../input/titanic-test-ready/train-ready.csv')
test = pd.read_csv('../input/titanic-test-ready/test-ready.csv')


# In[20]:


train.head(5)


# ### Custom Estimator using tensorFlow

# In[4]:


import tensorflow as tf


# We will create a DNN with three hidden layers, and dropout of 0.1 probability. Creating three fully connected layers each layer having a dropout probability of 0.1.

# #### Create the model

# Function to create the model

# In[21]:


def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# In[22]:


def train_input_fn(features, labels, batch_size):
    """An input function for training"""

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10).repeat().batch(batch_size)
    return dataset


# In[23]:


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset


# In[24]:


y = train.pop('Survived')
X = train


# In[25]:


from sklearn.model_selection import train_test_split 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)  


# In[26]:


feature_columns = []

for key in X_train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))


# [](http://)Build 2 hidden layer DNN with first with 2x inputs size and a second with inputs size units respectively

# In[27]:


units = len(X_train.columns) * 2
print (units)


# In[28]:


classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
        'feature_columns': feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [units, int(units/2)],
        # The model must choose between 3 classes.
        'n_classes': 2,
    })


# #### Train and evaluation the Model

# In[29]:


batch_size = 100
train_steps = 400
  
for i in range(100):
    
    classifier.train(
        input_fn=lambda:train_input_fn(X_train, y_train,
                                       batch_size),
        steps=train_steps)


# In[30]:


eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(X_val, y_val,
                                  batch_size)
)



# #### Generate predictions from the model

# In[31]:


predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(test,labels=None,
    batch_size=batch_size))


# In[33]:


results = list(predictions)

def x(res,j):
    class_id = res[j]['class_ids'][0]
    probability = int(results[i]['probabilities'][class_id] *100)

    if int(class_id) == 0:
        return ('%s%% probalitity to %s' % (probability,'Not survive'))
    else:
        return ('%s%% probalitity to %s' % (probability,'Survive!'))

print ('Predictions for 10 first records on test(dataset):')

for i in range(0,10):    
    print (x(results,i))


# #### Generate the csv to submit. 

# In[34]:


len(results)


# In[35]:


len(train)


# In[36]:


passengers = {}
i = len(train) + 1
for x in results:
    passengers[i] = int(x['class_ids'][0])
    i+=1


# In[37]:


import csv
csvfile = './gender_submission.csv'
with open(csvfile, 'w') as f:
    outcsv = csv.writer(f, delimiter=',')
    header = ['PassengerId','Survived']
    outcsv.writerow(header)
    for k,v in passengers.items():
        outcsv.writerow([k,v])


# In[38]:


submissions = pd.read_csv(csvfile)
submissions.head(5)

