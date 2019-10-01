#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import datetime

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# In[4]:


train = pd.read_csv("../input/train.csv", sep=",")
test = pd.read_csv("../input/test.csv", sep=",")


# In[5]:


train.info()


# In[6]:


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                             
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit      
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(10000)

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


# In[22]:


def preprocess_features(df):
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = df[
    ['Sex', 'Pclass', 'Age']]
  processed_features = selected_features.copy()

  
  return processed_features

def preprocess_targets(df):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Create a boolean categorical feature representing whether the
  # medianHouseValue is above a set threshold.
  output_targets["target"] =  df['Survived'] 
  return output_targets


# In[26]:





# In[9]:


def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    periods,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model of one feature.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.
      
  Returns:
    A `LinearClassifier` object trained on the training data.
  """

  steps_per_period = steps / periods
  
  # Create a linear classifier object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)    
  linear_classifier = tf.estimator.DNNClassifier(
  #linear_classifier = tf.estimator.LinearClassifier(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=[10, 10],
      optimizer=my_optimizer
  )
  
  # Create input functions
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["target"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["target"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["target"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
  
  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("LogLoss (on training data):")
  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.    
    training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
    training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
    
    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
    
    training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    # Occasionally print the current loss.
    print( "  period %02d : %0.2f" % (period, training_log_loss))
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
  print("Model training finished.")
  
  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()

  return linear_classifier


# In[24]:


training_examples = preprocess_features(train.head(700))
training_targets = preprocess_targets(train.head(700))

validation_examples = preprocess_features(train.tail(291))
validation_targets = preprocess_targets(train.tail(291))

# Double-check that we've done the right thing.
print ("Training examples summary:")
display.display(training_examples.describe())
print( "Validation examples summary:")
display.display(validation_examples.describe())

print( "Training targets summary:")
display.display(training_targets.describe())
print( "Validation targets summary:")
display.display(validation_targets.describe())


# In[53]:


def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """

  sex_categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(key='Sex',vocabulary_list=["M", "F"])
  sex_indicator_column = tf.feature_column.indicator_column(sex_categorical_column)
  
  pclass_categorical_column = tf.feature_column.categorical_column_with_identity(key='Pclass',num_buckets=4)
  pclass_indicator_column = tf.feature_column.indicator_column(pclass_categorical_column)
    
  age_boundaries = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
  age_categorical_column = tf.feature_column.numeric_column("Age")
  age_bucket_column = tf.feature_column.bucketized_column(age_categorical_column, boundaries=age_boundaries)
    
  sex_cross_pclass = ['Sex', 'Pclass']
  sex_cross_pclass_column = tf.feature_column.indicator_column(tf.feature_column.crossed_column(sex_cross_pclass , hash_bucket_size=10))
  
  feature_columns = set([sex_indicator_column, pclass_indicator_column, age_bucket_column, sex_cross_pclass_column])
  return feature_columns


# In[54]:


linear_classifier = train_linear_classifier_model(
    learning_rate=0.02,
    steps=100,
    batch_size=100,
    periods=10,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)


# In[55]:


v_examples = preprocess_features(train.tail(291))
v_targets = preprocess_targets(train.tail(291))
predict_validation_input_fn = lambda: my_input_fn(v_examples, 
                                                    v_targets["target"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
print(evaluation_metrics.keys())
print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
print(evaluation_metrics)


# In[56]:


validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
# Get just the probabilities for the positive class
validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
plt.plot(false_positive_rate, true_positive_rate, label="our model")
plt.plot([0, 1], [0, 1], label="random classifier")
_ = plt.legend(loc=2)

true_positive_rate


# In[57]:


result = test.copy()
target = test.copy()
target['Survived'] = 0
v_examples = preprocess_features(result)
v_targets = preprocess_targets(target)
predict_validation_input_fn = lambda: my_input_fn(v_examples, 
                                                    v_targets['target'], 
                                                    num_epochs=1, 
                                                    shuffle=False)
validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
result['probability'] = np.array([item['probabilities'][1] for item in validation_probabilities])


# In[58]:


result.probability.describe()


# In[59]:



result["Survived"] = result["probability"].apply(lambda a: 1 if a > 0.7 else 0)
evaluation = result[["PassengerId", "Survived"]]
evaluation


# In[60]:


evaluation.to_csv("evaluation_submission.csv",index=False)



# In[61]:


get_ipython().system(u'ls')

