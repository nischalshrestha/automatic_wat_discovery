#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction 
# In this notebook we will explore how to use TensorFlow as a substitute for scikit-learn using Canned Estimators.
# 
# ## TensorFlow
# TensorFlow is an open source library for numerical operations released by Google (2015). The library is well known to be used for implementing Machine Learning (ML) models, specifically Deep Learning. The library computes very efficiently numerical operations by optimizing the order of the operations and paralelizing their execution. 
# 
# ### TensorFlow as a ML Foundation
# Let's just pause a moment: how a numerical library helps to develop ML models?  Well, the answer is that all ML algorithms implementations are basically numerical operations and, because of that, TensorFlow is an ideal framework to implement them. 
# 
# ### TensorFlow as a ML Framework
# You might now think: "Does that mean that we need to implement every ML algorithms from (almost) scratch?" and the answer would be "only if you want to". You see, TensorFlow serves as a foundation to create ML Frameworks but it also has many layers of abstraction. One of the layers, is the pre-built and ready to use ML algorithms called **"Pre-made Estimators"** (aka Canned Estimators). We are going to use them as a substitute for scikit-learn :) 
# 
# Below the architecture of TensorFlow showing the different layers of abstraction. We are going to use the highest level. 
# 
# ![](https://3.bp.blogspot.com/-l2UT45WGdyw/Wbe7au1nfwI/AAAAAAAAD1I/GeQcQUUWezIiaFFRCiMILlX2EYdG49C0wCLcBGAs/s1600/image6.png)
# 
# Enought explanation let's get to work. This are all the libraries we are going to import:

# In[ ]:


# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf

print ( tf.__version__)


# # 2. Preprocess Dataset
# First we are going to read the dataset and cleaning it to use it in the classifier. 

# In[ ]:


# read the dataset
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# ## 2.1 Exploratory Analysis
# Let's do just the basic exploration since the objective here is to learn to use Tensorflow. If you want to go deeper on exploratory analysis there are other kernels on Kaggle that focus on this. For now, we are just going to: 
# 1. View the data with head()
# 2. Inspect data types with info()
# 3. Get some stats with describe()

# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# From the previous exploratory analysis we can derive some conclusions: 
# * Name and Ticket might not be good features* because they seem to be very specific and unique.
# * Passenger Id doesn't seem to be a good feature, because it only enumerates the passengers.
# * There are some missing values in the dataset. (NaN values observed in the head() result)
# 
# \* Note: there could be made some feature engineering on Name and Ticket features but we are not going to do that in this notebook. Our goal is to learn how to use TensorFlow ;)
# 
# Moving forward, first,  we are going to drop the the Name, Ticket and Passenger Id. Next, we are going to deal with the missing values.

# In[ ]:


#dropping columns
train_df = train_df.drop(["PassengerId","Name","Ticket"], axis=1)
test_df = test_df.drop(["PassengerId","Name","Ticket"], axis=1)


# ## 2.2 Missing Values
# Let's see how many missing values in each column we have:

# In[ ]:


print ("Train")
print (train_df.isnull().sum() )
print ("-------")
print ("Test")
print (test_df.isnull().sum() )


# As we can see we have missing values on the columns Age, Fare, Cabin and Embarked. Thus we need to treat this values.
# 
# ### 2.2.1 Missing Age Values
# To treat the missing age values we are going to use the mean age per gender.

# In[ ]:


# combine the whole dataset to get the mean values of the total dataset
# (just be careful to not leak data)
combined_df = pd.concat([train_df, test_df])

# get mean values per gender
male_mean_age = combined_df[combined_df["Sex"]=="male"]["Age"].mean()
female_mean_age = combined_df[combined_df["Sex"]=="female"]["Age"].mean()
print ("female mean age: %1.0f" %female_mean_age )
print ("male mean age: %1.0f" %male_mean_age )

# fill the nan values 
train_df.loc[ (train_df["Sex"]=="male") & (train_df["Age"].isnull()), "Age"] = male_mean_age
train_df.loc[ (train_df["Sex"]=="female") & (train_df["Age"].isnull()), "Age"] = female_mean_age

test_df.loc[ (test_df["Sex"]=="male") & (test_df["Age"].isnull()), "Age"] = male_mean_age
test_df.loc[ (test_df["Sex"]=="female") & (test_df["Age"].isnull()), "Age"] = female_mean_age


# ### 2.2.2 Missing Cabin Values
# To treat the missing cabin values we are just going to fill them with an "X"  

# In[ ]:


train_df["Cabin"] = train_df["Cabin"].fillna("X")
test_df["Cabin"] = test_df["Cabin"].fillna("X")


# ### 2.2.3 Missing Embarked Values
# To treat the missing embarked values we are just going to fill them with an "S"  

# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("S")
test_df["Embarked"] = test_df["Embarked"].fillna("S")


# ### 2.2.4 Missing Fare Value
# Finally we are just going to use the mean Fare value to fill the missing fare

# In[ ]:


mean_fare = combined_df["Fare"].mean()
test_df["Fare"] = test_df["Fare"].fillna(mean_fare)


# If we check again if there are missing values we are going to see there are none left.

# In[ ]:


print ("Train")
print (train_df.isnull().sum() )
print ("-------")
print ("Test")
print (test_df.isnull().sum() )


# ## 2.3 Spliting and Cross Validation
# In order to test the performance of our models we are going to create a cross validation set with basically 20% of the whole set. 
#  
# A good practice is to shuffle and do a stratified sampling which is usually applied to unbalanced datasets. For now we are just going to shuffle the dataset before splitting.

# In[ ]:


# sampling 80% for train data
train_set = train_df.sample(frac=0.8, replace=False, random_state=777)
# the other 20% is reserverd for cross validation
cv_set = train_df.loc[ set(train_df.index) - set(train_set.index)]

print ("train set shape (%i,%i)"  %train_set.shape)
print ("cv set shape (%i,%i)"   %cv_set.shape)
print ("Check if they have common indexes. The folowing line should be an empty set:")
print (set(train_set.index) & set(cv_set.index))


# # 3. Training a Classifier with TensorFlow
# ## 3.1 Estimator
# In tensorflow, the ready-to-use model is called an Estimator. The estimator has a very similar interface as a sklearn model.
# The estimator has four main methods:  
# * **train():**  Trains the model. Similar to fit() method in sklearn.
# * **evaluate():** Evaluates the model using a test dataset. Similar to the score() method in sklearn.
# * **predict():** Makes prediction on a given dataset. Similar to the predict() method in sklearn.
# * **export_savedmodel():** exports the trained model to disk.
# 
# Superficially, the biggest difference between a sklearn model and an Estimator is the way the data is fed to the object. The Estimator uses an **Input Function** which basically determines how the data will be fed to the model. The biggest advantage of using an input function is that you can write your own and encapsulate in it the procedure to clean and process the data. This allows you to, for example, change models without changing the way data is processed. We'll so more of this in the implementation.
# 
# ![](https://tensorflow.rstudio.com/tfestimators/articles/images/estimator-apis.png)
# 
# 
# ## 3.2 Instantiating Features
# Another feature of TensorFlow is that you instantiate feature definition objects which the Estimator will use and execute after the dataset has been set. This is very useful to format and engineer your features.
# 
# Below we can see 4 types of feature types:
# - **numeric_column:** It defines that the feature will be a float32 number. 
# - **bucketized_column:** It defines a feature that will be bucketized. You can define the range of the buckets.
# - **categorical_column_with_vocabulary_list:** As the name says, it basically does a one-hot-encoding for the column using a vocabulary list.
# - **categorical_column_with_hash_bucket:** Similarly, this definition encodes the categorical values using a hash bucket. You define the number of hashes it will have. This is very useful when you don't know the vocabulary but may cause hash collisions.
# 

# In[ ]:


# defining numeric columns
pclass_feature = tf.feature_column.numeric_column('Pclass')
parch_feature = tf.feature_column.numeric_column('Parch')
fare_feature = tf.feature_column.numeric_column('Fare')
age_feature = tf.feature_column.numeric_column('Age')

#defining buckets for children, teens, adults and elders.
age_bucket_feature = tf.feature_column.bucketized_column(age_feature,[12,21,60])

#defining a categorical column with predefined values
sex_feature = tf.feature_column.categorical_column_with_vocabulary_list(
    'Sex',['female','male']
)
#defining a categorical columns with dynamic values
embarked_feature =  tf.feature_column.categorical_column_with_hash_bucket(
    'Embarked', 3 
)
cabin_feature =  tf.feature_column.categorical_column_with_hash_bucket(
    'Cabin', 100 
)

feature_columns = [ pclass_feature,age_feature, age_bucket_feature, parch_feature, 
                   fare_feature, embarked_feature, cabin_feature ]


# ## 3.3 The Estimator (aka model)
# We are going to instantiate a simple Linear Classifier. We just need to pass the feature column definition.

# In[ ]:


estimator = tf.estimator.LinearClassifier(
    feature_columns=feature_columns)


# ## 3.4 Input function
# Tensorflow provides (for now) two pre-built input functions __generators*__. One that reads from pandas DataFrame and another one that reads from numpy arrays. For our case we will use the pandas input function.
# 
# The pandas_input_fn()  function receives as parameters:
# - **x:** The DataFrame with the features.
# - **y:** A Series object with the labels.
# - **num_epochs:** The number of times it will use the whole dataset. None means that it will run indefinetely.
# - **shuffle:** If the dataset will be shuffled.
# - **target_column:** The name it will use internally for the label column.
# 
# \* Notice that we store the returning object from calling the generator. The return object is a function, which we will later pass to the Estimator object.
# 
# \* Also notice that we are generating two input functions: one for training and another one for the cross-validation set. This is because these functions behave differently and will return different datasets.

# In[ ]:


# train input function
train_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=train_set.drop('Survived', axis=1),
      y=train_set.Survived,
      num_epochs=None, #For training it can use how many epochs is necessary
      shuffle=True,
      target_column='target',
)

cv_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=cv_set.drop('Survived', axis=1),
      y=cv_set.Survived,
      num_epochs=1, #We just want to use one epoch since this is only to score.
      shuffle=False  #It isn't necessary to shuffle the cross validation 
)


# ## 3.5 Training
# Now we are just going to call the train() method, passing the input function and defining how many steps the estimator will perform.
# 

# In[ ]:


estimator.train(input_fn=train_input_fn, steps=400)


# ## 3.6 Evaluating the model
# Finally, we are going to evaluate the model using the cross validation input function.
# The evaluate function returns various scoring methods. In our case we are interested only on accuracy.
# 

# In[ ]:


scores = estimator.evaluate(input_fn=cv_input_fn)
print("\nTest Accuracy: {0:f}\n".format(scores['accuracy']))


# # 4. Other Estimators
# ## 4.1 Core TensorFlow Estimators
# TensorFlow in it's core modules offers just a few number of pre-made models:
# - LinearClassifier
# - LinearRegressor
# - DNNClassifier
# - DNNRegressor
# - DNNLinearCombinedClassifier
# - DNNLinearCombinedRegressor
# 
# You can also create your own estimator following the instructions from this [tutorial](https://www.tensorflow.org/extend/estimator)
# 
# ## 4.2 DNN
# For now, we are just going to test the Deep Neural Network (DNN) Classifier. The good thing is that we can re-use almost everything we defined before for the Linear Classifier. Since DNN doesn't support categorical with hash bucket columns, the only modification we are going to do is to define the Embarked and Cabin columns as embedding columns.
# 
# Notice that the rest of the steps are the same ones that we did with the Linear Classifier.

# In[ ]:


# DNN doesn't support categorical with hash bucket
embarked_embedding =  tf.feature_column.embedding_column(
    categorical_column = embarked_feature,
    dimension = 3,
)
cabin_embedding =  tf.feature_column.embedding_column(
    categorical_column = cabin_feature,
    dimension = 300,
)

# define the feature columns
feature_columns = [ pclass_feature,age_feature, age_bucket_feature, parch_feature, 
                   fare_feature, embarked_embedding, cabin_embedding ]

# instantiate the estimator
NNestimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 30 , 10])

# call the train function using the train input function
NNestimator.train(input_fn=train_input_fn, steps=1000)


# In[ ]:


# evaluate and print the accuracy using the cross-validation input function
accuracy_score = NNestimator.evaluate(input_fn=cv_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


# ## 4.4 Contrib Estimators
# TensorFlow has other estimators in the contrib module. This module contains volatile and experimental code, which means that sometimes things change from release to release or that some things need tweaking to make them work.
# 
# Some of the estimators found on contrib are:
# - tensor_forest: a random forest estimator
# - crf:  A linear-chain Coditional Random Field
# - KMeansClustering: Found in contrib.learn
# - LogisticRegressor:  Found in contrib.learn
# 
# If you want to explore more options check these links:
# - [contrib](https://www.tensorflow.org/api_docs/python/tf/contrib)
# - [contrib.learn](https://www.tensorflow.org/api_guides/python/contrib.learn#Estimators)
# 

# ## 4.5 Simple Random Forest
# Just to get a glimpse of the contrib estimators we will use the tensor_forest estimator.
# 
# 
# ### 4.5.1 Prepare data
# Tensor Forest implementation only accepts int32 and float32 inputs. So, first, we will prepare the data using a copy of the train and cross-validation sets.

# In[ ]:


def prepare_datasets(df):
    df_copy = df[['Pclass', 'Parch',  'Sex', 'Embarked', "Age"]].copy()
    df_copy.loc[:,"Sex"] = df_copy.Sex.apply(lambda x: 0 if x =="male" else 1)

    e_map = {"C": 0,"Q":1, "S":2}
    df_copy.loc[:,"Embarked"] = df_copy.Embarked.apply(lambda x: e_map[x])

    df_copy.loc[:,"Age"]= df_copy.Age.astype(np.float32)

    x = df_copy[['Pclass', 'Parch', 'Age']].astype(np.float32)
#     y = train_set.Survived.astype(np.int32)
    y = df.Survived.astype(np.bool)
    return x, y

x_train, y_train = prepare_datasets(train_set)
x_cv, y_cv = prepare_datasets(cv_set)


# ### 4.5.2 Customized Input Functions
# There's an issue with tensor_forest and it's that it needs the data to have 2 dimensions and the standard pandas_input_fn only reuturns 1 dimensional inputs. For this reason we will implement a customized input function which internally uses the standard input function but expands the dimension of X. 
# 
# In order to generalize we are gonna create a generator function of this customized input functions.

# In[ ]:


def generate_tf_input_fn(x_input,y_input,num_epochs=None):
    #this is the function we are generating
    def _input_fn_():
        # generate a standard input function
        train_input_fn = tf.estimator.inputs.pandas_input_fn(
            x= x_input,  
            y= y_input,
            num_epochs=num_epochs,
            shuffle=True,
            target_column='target',
        )
        #execute the standard input function 
        x, y = train_input_fn()
        # expand the shape of the results (necessary for Tensor Forest)
        for name in x:
            x[name] = tf.expand_dims(x[name], 1, name= name) 
        return x, y
    
    return _input_fn_


# ### 4.5.3 Train the estimator
# Now that we have the input function we will train the tensor_forest estimator very similar to the way we did for the other estimators. 

# In[ ]:


# generate custom train input function
forest_train_input_fn = generate_tf_input_fn(x_train,y_train,num_epochs=None)

# instantiate the estimator
params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    num_classes=2, num_features=4, regression=False,
    num_trees=50, max_nodes=1000).fill()
classifier2 = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params)
# train the estimator
classifier2.fit(input_fn=forest_train_input_fn)


# ### 4.5.4 Evaluating the tensor forest estimator
# Finally, we are going to test the model's accuracy just as any estimator calling the evaluate() method.

# In[ ]:


# evaluate and print the accuracy using the cross-validation input function
forest_cv_input_fn = generate_tf_input_fn(x_cv, y_cv, num_epochs=1)
accuracy_score = classifier2.evaluate(input_fn=forest_cv_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

