#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf
import math
from sklearn import metrics

import os
print(os.listdir("../"))


# ## Configuring pandas

# In[ ]:


pd.options.display.max_rows = 25
pd.options.display.float_format = '{:.2f}'.format


# ## Importing and displaying data

# In[ ]:


titanic_survival_data = pd.read_csv('../input/train.csv')
titanic_survival_test_data = pd.read_csv('../input/test.csv')

# shuffling data
titanic_survival_data = titanic_survival_data.reindex( np.random.permutation( titanic_survival_data.index ) )

titanic_survival_data


# ## Pre-processing data
# #### 1. Scrubbing
#    * Replacing missing data with mean, if possible
#    * Remove rows that have omitted values
#    * Removing duplicates rows
#    * Removing rows with bad labels
#    * Removing rows with bad feature values
# 
# #### 2. Feature Manipulation
#     * Dropping non-relevant features
#     * One-hot encoding and binning
#     * Feature crosses

# In[ ]:


def normalize( col ):
    return ( col - col.min() ) / col.max()

def preProcess( df, test = False ):
    '''
        Takes a pandas dataframe and performs scrubbing
        
        Args: 
            df - dataframe
        
        Returns:
            a pandas dataframe
    '''
    # filling missing age values with mean
    df["Age"] = df["Age"].fillna((df["Age"].mean()))
    
    # removing rows with omitted values
    df = df.drop(  df[ (df["Age"].isnull() | df["Fare"].isnull()  | df["SibSp"].isnull() ) ].index )
    
    # removing duplicates
    df = df.drop_duplicates()
    
    # removing rows with bad labels( No obvious way of finding out in this dataset )
    
    # taking only relevant features
    if not test:
        relevant_features = [ "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare" ]
        new_df = df[ relevant_features ].copy()
    else:
        relevant_features = [ "Pclass", "Age", "SibSp", "Parch", "Fare" ]
        new_df = df[ relevant_features ].copy()
    
    # converting string column ( sex, Embarked ) to numberic datatype ( One hot encoding )
    Sex = pd.get_dummies( df["Sex"], drop_first = True )
    Embarked = pd.get_dummies( df["Embarked"], drop_first = True )
    
    new_df = pd.concat( [ new_df, Sex, Embarked ], axis = 1 )
    
    # adding new polynomial features
    new_df["Age_qd"] = df["Age"]**4
    new_df["Fare_qd"] = df["Fare"]**4
    new_df["Pclass_qd"] = df["Fare"]**4
    
    # adding new feature crosses
    new_df["Age_Fare"] = df["Age"] * df["Fare"]
    new_df["Age_Pclass"] = df["Age"] * df["Pclass"]
    
    # feature scaling
    new_df["Fare"] = normalize( new_df["Fare"] )
    new_df["Age"] = normalize( new_df["Age"] )
    new_df["Age_qd"] = normalize( new_df["Age_qd"] )
    new_df["Age_Fare"] = normalize( new_df["Age_Fare"] )
    new_df["Fare_qd"] = normalize( new_df["Fare_qd"] )
    new_df["Pclass_qd"] = normalize( new_df["Pclass_qd"] )
    new_df["Age_Pclass"] = normalize( new_df["Age_Pclass"] )
    
    return new_df


# In[ ]:


processed_data = preProcess( titanic_survival_data )
processed_data


# ## Data Insight
# Plotting histograms and graphs helps us understand data relation with lables. It also gives us a hint about bad labels and features that we may miss out during preprocessing.
# 
# We can clearly see that there are some rows that have missing data for "Age". We need to deal with that during preprocessing.

# In[ ]:


processed_data.describe()


# ## Finding correlation
# We make use of the pearson coefficient to find out redundant coefficients and also select the most influencial coefficients.
# 
# As we can see, the "Sex", "Fare" and the "Pclass" feature have major effect on the label. "Age" comes next and finally the "SibSp". "Parch" has the least effect. Thus the features that we should be looking for are "Age", "Fare", "Pclass", "SibSp" and "Parch".

# In[ ]:


processed_data.corr( method="pearson" )


# ## Histogram of age group distribution of passengers
# Let us divide the age groups into 20 bins.
# 
# * The plot shows us that most of the people were between the age of 15 - 50.
# * A good number of infants ( below the age of 5 ) were also present on the ship.

# In[ ]:


plt.figure( figsize = ( 14, 8 ) )

ax = plt.subplot( 1, 1, 1 )

ax.set_title( "Age group distribution" )

ax.set_autoscaley_on( False )
ax.set_ylim( [ 0, 300 ] )

ax.set_autoscalex_on( False )
ax.set_xlim( [ 0, 1 ] )

ax.set_ylabel("Population", fontsize = 12)
ax.set_xlabel("Age", fontsize = 12)

x = processed_data["Age"]

ax.hist( x, bins = 20 )

_ = plt.plot()


# ## Scatter plot of Age vs Fare vs Survived
# **We clearly see a few out liers here**
# 
# #### Looking at the plot below gives us some interesting insight. 
# * Firstly, **poor** people between the age of 15 - 50 mostly ended up dead, while the **rich** people of the same age group mostly lived.
# * Secondly, **most** of the children ( below age 5 ) lived no matter their economic status.
# * Thirdly, **most** the old people ( age more than 50 ) died.
# 

# In[ ]:


plt.figure( figsize = ( 14, 14 ) )

ax = plt.subplot( 1, 1, 1 )

ax.set_title( "Age vs Fare" )

ax.set_autoscaley_on( False )
ax.set_ylim( [ 0, 1 ] )

ax.set_autoscalex_on( False )
ax.set_xlim( [ 0, 1 ] )

ax.set_ylabel("Age", fontsize = 12)
ax.set_xlabel("Fare", fontsize = 12)

x = processed_data["Fare"]
y = processed_data["Age"]
c = [ 'red' if s == 0 else  'green' for s in processed_data["Survived"] ]

ax.scatter( x, y, c = c, alpha = 0.5 )

_ = plt.plot()


# ## Gender distribution in survival
# From the pie charts below, it is evident that from the surviving people, most of them were women, and that clain is backed by the other pie chart saying that the proportion of female deaths was significantly less as compared to male. Thus the data suggests that if you were a women, you would have likely to have lived.

# In[ ]:


plt.figure( figsize = ( 16,12 ) )

ax = plt.subplot( 1, 2, 1 )

labels = 'Male', 'Female'

male_dead = len(processed_data[ (processed_data["male"] == 1) & (processed_data["Survived"] == 0) ])
female_dead = len(processed_data[ (processed_data["male"] == 0) & (processed_data["Survived"] == 0) ])

sizes = [male_dead, female_dead]

ax.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')

ax.set_title("Death ratio")

ax = plt.subplot( 1, 2, 2 )

labels = 'Male', 'Female'

male_dead = len(processed_data[ (processed_data["male"] == 1) & (processed_data["Survived"] == 1) ])
female_dead = len(processed_data[ (processed_data["male"] == 0) & (processed_data["Survived"] == 1) ])

sizes = [male_dead, female_dead]

ax.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')

ax.set_title("Survived ratio")

_ = plt.show()


# ## Splitting the data and prepare it for use
# Now that we have sufficient insight about our data, we will go ahead and split the data into 4 groups: Training data, Training target, Validation data and Validation Target
# 
# Note: We will use the universal rule of splitting the data 70-30
# 
# For the sake of sanity checking, you can go ahead and print the final split that we get.

# In[ ]:


def splitData( df, test = False ):
    '''
    Function to split data into training and validation set
    
    Args: 
        df - A pandas pre-processed dataset
    Test:
        bool - If set to true, then the data would not be split
        
    Return:
        A tuple - ( training set, training target, validation set, validation target )
    '''
    if not test:
        # seperating the target from the data
        data = df[df.columns.difference(['Survived'])]
        target = df["Survived"]

        training_data = data.head( 572 )
        training_target = target.head( 572 )

        validation_data = data.tail( 142 )
        validation_target = target.tail( 142 )

        return ( training_data, training_target, validation_data, validation_target )
    else:
        data = df[df.columns.difference(['Survived'])]
        
        return ( data )


# In[ ]:


training_data, training_target, validation_data, validation_target = splitData( processed_data )


# ## Preparing the model
# 
# Now that we have our data ready, we can now decide which model to use. We have many options here to choose from namely Neural Network and logistic classifier. I am trying to keep things simple so I will use logistic classifier. I am expecting an accuracy of 70 - 80 % from my model.
# 
# We will use the low level tf apis to implement the model. 
# 
# #### There are three major things to do here:
# * Define a prediction model
# * Define a cost function with respect to the model
# * Train on the data using suitable hyper-parameters
# 
# Note: This is a modular approach that I think people should try to follow thus making it easy to experiment with new models without much change to the existing code

# In[ ]:


# define hyper params
learning_rate = 0.9
epochs = 2000
number_of_features = training_data.shape[1]

# defining input and output placeholders ( Fare, Age, Pclass, sibSp, Parch, Sex )
X = tf.placeholder( dtype = tf.float64 )

Y = tf.placeholder( dtype = tf.float64 )

# list of parameters that we wish to learn (  number of features + intercept )
W = tf.Variable( tf.random_normal( shape = [1,number_of_features], dtype=tf.float64 ) )
b = tf.Variable( tf.random_normal( shape = [ 1, 1 ], dtype=tf.float64  ) )

# define prediction model
y_model = tf.sigmoid( tf.matmul( X, tf.transpose( W ) ) + b )

# define cost function
cost =  tf.reduce_mean( -Y * tf.log( y_model ) - ( 1 - Y ) * tf.log( 1 - y_model ) )

# define training op
train_op = tf.train.GradientDescentOptimizer( learning_rate =  learning_rate ).minimize( cost )


# ## Defining training loop
# Now we can finally defnine the training loop and run the trian_op on the data.
# The hyperparameters are user tunable thus may be required to change in order to get he best output.
# 
# We will also make use of our validation set to check if our model is overfitting the training data.

# In[ ]:


W_trained = []

with tf.Session() as sess:
    
    sess.run( tf.global_variables_initializer() )
    
    prev_error = 0
    for epoch in range( epochs ):
        error, _ = sess.run( [ cost, train_op], feed_dict = { 
            X: training_data, 
            Y: training_target
        } )
        
        if epoch % 100 == 0:
            print( epoch, error )
        
#         if( abs(prev_error - error) < 0.001 ):
#             break
            
        prev_error = error
    
    W_trained, b_trained = sess.run( [ W, b] )
    
    print(W_trained)


# ## Evaluation
# 
# Now that we have trained our model and have the final parameters, we can test the accuracy of the model on training and validation data

# In[ ]:


Y_output = tf.round(y_model)
Y_orignal = tf.placeholder( dtype=tf.float64 )

correct_prediction = tf.equal( Y_output, Y_orignal )

accuracy = tf.reduce_mean( tf.cast( correct_prediction, 'float' ) )


# In[ ]:


with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    
    acc = sess.run( accuracy, feed_dict = { 
            X: training_data, 
            W: W_trained,
            b: b_trained,
            Y_orignal: training_target
        } )
    
    print( "Accuracy on the training set is %.2f %%" % (acc * 100) )


# In[ ]:


with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    
    acc = sess.run( accuracy, feed_dict = { 
            X: validation_data, 
            W: W_trained,
            b: b_trained,
            Y_orignal: validation_target
        } )
    
    print( "Accuracy on the validation set is %.2f %%" % (acc * 100) )


# ## Testing the model on test data

# In[ ]:


data = splitData( preProcess(titanic_survival_test_data, test = True), test = True )
data


# In[ ]:


output = []
with tf.Session() as sess:
    
    output = sess.run( Y_output, feed_dict = { 
            X: data, 
            W: W_trained,
            b: b_trained,
        } )
    
output


# In[ ]:




