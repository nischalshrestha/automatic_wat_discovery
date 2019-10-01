#!/usr/bin/env python
# coding: utf-8

# # Using a Tensorflow DNNClassifier to classify Titanic dataset with One hot encoder approach

# **This version for each trainning step I evaluted the model**
# 
# My focus here is just show a basic approach of a Deep Neural Classifier using Google's Open Source TensorFlow library.
# 
# The TensorFlow team developed the Estimator API to make the library more accessible to the everyday developer. This high level API provides a common interface to train(...) models, evaluate(...) models, and predict(...) outcomes of unknown cases similar to (and influenced by) the popular Sci-Kit Learn library, which is accomplished by implementing a common interface for various algorithms

# ### Load data after feat. engineering and cleanning data

# In[122]:


import pandas as pd
import numpy as np


# I did the feature engineering and cleaning step separately. If want to see more details please, see here: [ Titanic Best Working Classfier:](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier) by Sina

# In[123]:


train = pd.read_csv('../input/titanic-test-ready/train-ready.csv')
test = pd.read_csv('../input/titanic-test-ready/test-ready.csv')


# In[124]:


train.head(5)


# ### One hot encoder

# One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.  
# 
# The categorical value represents the numerical value of the entry in the dataset. For example: For each **Title **(Master., Miss, Mr, Mrs, Capt, etc)  in the dataset or during a feature engineering process, it would have been given categorical value as number like 1, 2, 3... N . As the number of unique entries increases, the categorical values also proportionally increases.
# 
# Problem with label encoding is that it assumes higher the categorical value, better the category, but is not necessarily true!
# 
# What this form of organization presupposes is Master  < Miss < Mr < Mrs ... ( 1 < 2 < 3 < 4...) based on the categorical values. Say supposing your model internally calculates average, then accordingly we get, 1+3 = 4/2 =2. This implies that: Average of Master and Mr is Miss. This is definitely a recipe for disaster. 
# 
# This model’s prediction would have a lot of errors. 
# 
# This is why we use one hot encoder to perform “binarization” of the category and include it as a feature to train the model.
# 
# Lets do it!

# In[125]:


def one_hot_enconder(df1,df2):
    col_name = []
    cols = {}
    
    col = [x for x in df1.columns if x not in ['Survived']]
    len_df1 = df1.shape[0]
    df = pd.concat([df1,df2],ignore_index=True)
    
    print('Categorical feature',len(col))
    for c in col:
        if df[c].nunique()>2 :
            col_name.append(c)
            cols[c] = c
    
    df = pd.get_dummies(df, prefix=cols, columns=col_name,drop_first=True)

    df1 = df.loc[:len_df1-1]
    df2 = df.loc[len_df1:]
    print('Train',df1.shape)
    print('Test',df2.shape)
    return df1,df2


# In[126]:


train,test = one_hot_enconder(train,test)


# In[127]:


train.head()


# ### DNNClassifier using tensorFlow: a basic approach

# Helper functions

# In[128]:


def train_input_fn(features, labels, batch_size):
    """An input function for training"""

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10).repeat().batch(batch_size)
    return dataset


# In[129]:


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


# In[130]:


y = train.pop('Survived')
X = train


# #### Create the model

# In[131]:


import tensorflow as tf

feature_columns = []

for key in X.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))


# One hidden layer of 37 ( number of freatures x 2 + 1 ). The model must choose between 2 classes.

# In[132]:


classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[37,],
    n_classes=2)


# #### Train and evaluation the Model

# In[133]:


from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split 


# In[134]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12) 
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))  
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))  
print("Testing instances    {}, Testing features    {}".format(test.shape[0], test.shape[1]))  


# In[135]:


batch_size = 100
train_steps = 400

evaluations = []  
for i in range(200):  
    classifier.train(
        input_fn=lambda:train_input_fn(X, y,
                                       batch_size),
                    steps=train_steps)
    
    eval_result = classifier.evaluate(
            input_fn=lambda:eval_input_fn(X_val, y_val,batch_size)
        )
    
    evaluations.append(eval_result)


# In[144]:


import matplotlib.pyplot as plt  
get_ipython().magic(u'matplotlib inline')

# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [14, 10]

loss_values = [ev['loss'] for ev in evaluations]  
training_steps = [ev['global_step'] for ev in evaluations]

plt.scatter(x=training_steps, y=loss_values)  
plt.xlabel('Training steps')  
plt.ylabel('Loss (SSE)')  
plt.show() 


# #### Generate predictions from the model

# In[145]:


predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(test,labels=None,
    batch_size=batch_size))


# In[146]:


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

# In[147]:


len(results)


# In[148]:


len(train)


# In[149]:


passengers = {}
i = len(train) + 1
for x in results:
    passengers[i] = int(x['class_ids'][0])
    i+=1


# In[152]:


import csv
csvfile = 'submissions_ohe2.csv'
with open(csvfile, 'w') as f:
    outcsv = csv.writer(f, delimiter=',')
    header = ['PassengerId','Survived']
    outcsv.writerow(header)
    for k,v in passengers.items():
        outcsv.writerow([k,v])


# In[153]:


submissions = pd.read_csv(csvfile)
submissions.head(10)

