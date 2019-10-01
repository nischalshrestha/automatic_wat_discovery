#!/usr/bin/env python
# coding: utf-8

# # Using a Tensorflow DNNClassifier to classify Titanic dataset

# My focus here is just show a basic approach of a Deep Neural Classifier using Google's Open Source TensorFlow library.
# 
# The TensorFlow team developed the Estimator API to make the library more accessible to the everyday developer. This high level API provides a common interface to train(...) models, evaluate(...) models, and predict(...) outcomes of unknown cases similar to (and influenced by) the popular Sci-Kit Learn library, which is accomplished by implementing a common interface for various algorithms

# ### Load data after feat. engineering and cleanning data

# In[1]:


import pandas as pd
import numpy as np


# I did the feature engineering and cleaning step separately. If want to see more details please, see here: [ Titanic Best Working Classfier:](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier) by Sina

# In[2]:


train = pd.read_csv('../input/titanic-test-ready/train-ready.csv')
test = pd.read_csv('../input/titanic-test-ready/test-ready.csv')


# In[3]:


train.head(5)


# ### DNNClassifier using tensorFlow: a basic approach

# Helper functions

# In[4]:


def train_input_fn(features, labels, batch_size):
    """An input function for training"""

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10).repeat().batch(batch_size)
    return dataset


# In[5]:


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


# #### To split out this data (using Sci-Kit Learn's)

# In[6]:


from sklearn.model_selection import train_test_split 
y = train.pop('Survived')
X = train


# In[7]:


# 20% for evaluate
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23) 


# #### Create the model

# In[8]:


import tensorflow as tf

feature_columns = []

for key in X_train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))


# Two hidden layers of 10 nodes each. The model must choose between 2 classes.

# In[9]:


classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=2)


# #### Train the Model

# In[10]:


batch_size = 100
train_steps = 400

for i in range(0,100):
    
    classifier.train(
        input_fn=lambda:train_input_fn(X_train, y_train,
                                                 batch_size),
        steps=train_steps)


# #### Evaluate the model

# In[11]:


eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(X_tmp, y_tmp,batch_size)
)


# #### Generate predictions from the model

# In[12]:


predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(test,labels=None,
    batch_size=batch_size))


# In[13]:


results = list(predictions)

def x(res,j):
    class_id = res[j]['class_ids'][0]
    probability = int(results[j]['probabilities'][class_id] *100)

    if int(class_id) == 0:
        return ('%s%% probalitity to %s' % (probability,'Not survive'))
    else:
        return ('%s%% probalitity to %s' % (probability,'Survive!'))

print ('Predictions for 10 first records on test(dataset):')

for i in range(0,10):    
    print (x(results,i))


# #### Generate the csv to submit. 

# In[14]:


len(results)


# In[15]:


len(train)


# In[16]:


passengers = {}
i = 892
for x in results:
    passengers[i] = int(x['class_ids'][0])
    i+=1


# In[17]:


import csv
csvfile = 'submissions.csv'
with open(csvfile, 'w') as f:
    outcsv = csv.writer(f, delimiter=',')
    header = ['PassengerId','Survived']
    outcsv.writerow(header)
    for k,v in passengers.items():
        outcsv.writerow([k,v])


# In[18]:


submissions = pd.read_csv(csvfile)
submissions.head(5)

