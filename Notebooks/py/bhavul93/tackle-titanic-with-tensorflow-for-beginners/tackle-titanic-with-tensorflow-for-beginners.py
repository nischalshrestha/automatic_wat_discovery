#!/usr/bin/env python
# coding: utf-8

# # Tackling Titanic with basic Tensorflow
# 
# In this notebook, we're going to explore the data a bit, and try basic pandas and tensorflow to get the job done. It is meant for beginners, and should help those who're just getting started with kaggle and/or tensorflow.
# 
# **What you can learn from it :**
# - How to read the data and do basic preprocessing required to use neural networks
# - Design a 2 layer Neural Network model yourself using tensorflow instead of using in built DNN classifier
# 
# 
# **What is NOT covered in this tutorial :**
# - Data visualization with lot of graphs and their observations
# - Comparison of different classifiers for the problem
# 
# I'm myself trying to improve everyday. This is my first Kernel at Kaggle. Your feedback would be very appreciative. 

# In[ ]:


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


# # Read Data
# 
# Here's what we'll do 
# - Read data into csv files (train and test)
# - Print out a small summary of the data
# - Combine them into one dataset if we require later on
# - Find out how many examples of each class exist in the training data (check if skewed or not)
# - Find out how many features have null values
# - Fix null values for numerical features
# - Fix null values with some values for categorical features

#  ## Read data into csv files

# In[ ]:


# Read data into csv files
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("train_df shape : ",train_df.shape)
print("test_df shape : ",test_df.shape)


# ## Print out summary of data

# In[ ]:


# Print small summary
print("A look at training data:")
train_df.head()


# In[ ]:


print("A look at testing data:")
test_df.head()


# > ***Obvious observation - 'Survived' column is missing in test_df***

# ## Find out how many examples of each class in training data

# In[ ]:


train_df.groupby('Survived')['PassengerId'].count()


# **Observations** : 
# 1. 549+342 = 891. So no data in the training data is missing its class
# 2. It's not such a skewed dataset 

# ## How many features have null values

# In[ ]:


# What any does is return whether any element in a particular axis is true or not. So, it works for us in this case. For each column, it checks if any column has a NaN value or not.
train_df.isnull().any()


# **Age**, **Cabin** and **Embarked** are the only ones having NaN values. We gotta fix them. 

# In[ ]:


# How many NaN values of Age in train_df?
train_df['Age'].isnull().sum()


# In[ ]:


# For Cabin
train_df['Cabin'].isnull().sum()


# In[ ]:


# For Embarked
train_df['Embarked'].isnull().sum()


# ## Fixing null / NaN values for each column one by one

# ### For embarked

# In[ ]:


train_df.groupby('Embarked')['PassengerId'].count()


# We observed earlier that only 2 entries have NaN for Embarked. And here, we see there are only 3 possible values of Embarked - C, Q and S. Out of which, S has the most number. So, let's just assign the missing ones to S. 

# In[ ]:


train_df['Embarked'] = train_df['Embarked'].fillna('S')


# Now, let's check again....

# In[ ]:


train_df.groupby('Embarked')['PassengerId'].count()


# Perfect.

# ### For Age

# In[ ]:


train_df.groupby('Age')['PassengerId'].count()


# So, the first thing to note is, thie Age can be in decimals! So, it's more of a continuous variable than discrete one.
# I think it would make sense to fix the missing ones by filling them with the mean?

# In[ ]:


train_df['Age'].mean()


# In[ ]:


train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())


# Now, let's check how many missing values remain.

# In[ ]:


train_df['Age'].isnull().sum()


# Perfect.

# ### For Cabin

# In[ ]:


train_df.groupby('Cabin')['PassengerId'].count()


# Okay, So : 
# - This can be alphanumeric
# - 147 different vaulues exist for Cabin
# - None of them seem to be far far greater in number than others
# - A lot of values are actually missing - 687!
# 
# So, let's do one thing - Add a new 'Cabin' value as 'UNKNOWN' and fill the data with that

# In[ ]:


train_df['Cabin'] = train_df['Cabin'].fillna('UNKNOWN')


# Check how many NaN now

# In[ ]:


train_df['Cabin'].isnull().sum()


# Perfect.

# ### All NaN values fixed

# In[ ]:


# What any does is return whether any element in a particular axis is true or not. So, it works for us in this case. For each column, it checks if any column has a NaN value or not.
train_df.isnull().any()


# ## Helper Methods we learnt from above
# 
# We'll use these for testing dataset, and maybe in future as well.

# In[ ]:


def get_num_of_NaN_rows(df):
    return df.isnull().sum()

def fill_NaN_values_for_numerical_column(df, colname):
    df[colname] = df[colname].fillna(df[colname].mean())
    return df

def fill_NaN_values_for_categorical_column(df, colname, value):
    df[colname] = df[colname].fillna(value)
    return df


# In[ ]:


# Let's test them on test data (which still might have missing rows!)
num_of_NaN_rows_of_test_set = get_num_of_NaN_rows(test_df)
print("num_of_NaN_rows_of_test_set : ",num_of_NaN_rows_of_test_set)


# One chapter done. 

# # Preprocessing Data
# 
# - Convert Categorical values to numerical ones
# - Divide train_df into train_df_X and train_df_y
# - One hot values

# ### Convert Categorical values to numerical ones

# **1. Find which columns are categorical**
# 
# Ref : https://stackoverflow.com/questions/29803093/check-which-columns-in-dataframe-are-categorical/29803290#29803290

# In[ ]:


all_cols = train_df.columns


# In[ ]:


numeric_cols = train_df._get_numeric_data().columns


# In[ ]:


categorical_cols = set(all_cols) - set(numeric_cols)
categorical_cols


# In[ ]:


# Let's make a helper method from this now.
def find_categorical_columns(df):
    all_cols = df.columns
    numeric_cols = df._get_numeric_data().columns
    return set(all_cols) - set(numeric_cols)


# **2. Convert to numerical ones using get_dummies of Pandas**
# 
# Ref : http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/

# In[ ]:


# First, let's backup our train_df and test_df till now
train_df_backup_filledna_still_having_categorical_data = train_df
train_df_backup_filledna_still_having_categorical_data.head()


# In[ ]:


# Now, let's convert it.
train_df_dummies = pd.get_dummies(train_df, columns=categorical_cols)
train_df_dummies.shape


# In[ ]:


# However, backup's shape is still 
train_df_backup_filledna_still_having_categorical_data.shape


# In[ ]:


# Let's check out data once
train_df_dummies.head()


# In[ ]:


train_df.shape


# ### Another way to convert Categorical columns data into numerical is assigning them integers
# Ref : https://stackoverflow.com/questions/42215354/pandas-get-mapping-of-categories-to-integer-value

# In[ ]:


# 2nd way to convert is having integers represent different values of each categorical column
train_df_numerical = train_df.copy()
for col in categorical_cols:
    train_df_numerical[col] = train_df_numerical[col].astype('category')
    train_df_numerical[col] = train_df_numerical[col].cat.codes
train_df_numerical.shape


# In[ ]:


train_df_numerical.head()


# In[ ]:


# Let's make helper function here also
def convert_categorical_column_to_integer_values(df):
    df_numerical = df.copy()
    for col in find_categorical_columns(df):
        df_numerical[col] = df_numerical[col].astype('category')
        df_numerical[col] = df_numerical[col].cat.codes
    return df_numerical


# *Perfect*.
# 
# Now, we have all of these available for our use : 
# 
# * **train_df**                    : original training dataset   (891,12)
# * **train_df_dummies**  : training dataset with dummies (891, 1732)
# * **train_df_numerical** : training dataset with integers for categorical attributes (891,12) 

# # Running a model in Tensorflow
# 
# This will again involve a set of steps
# - Get data converted to numpy arrays so tensorflow can read them
# - Write tensorflow model
# - Run a session of tensorflow model and check accuracy on training data set
# 
# Try the above for both train_df_dummies and train_df_numerical

# In[ ]:


# import tensorflow stuff...
import tensorflow as tf


# In[ ]:


# Dividing data between X and Y
# Ref : https://stackoverflow.com/questions/29763620/how-to-select-all-columns-except-one-column-in-pandas

train_df_dummies_Y = train_df_dummies['Survived']
# Don't worry. drop does not change the existing dataframe unless inplace=True is passed.
train_df_dummies_X = train_df_dummies.drop('Survived', axis=1)

train_df_numerical_X = train_df_numerical.drop('Survived', axis=1)
train_df_numerical_Y = train_df_numerical['Survived']

print("train_df_numerical_X shape : ",train_df_numerical_X.shape)
print("train_df_numerical_Y shape : ",train_df_numerical_Y.shape)
print("train_df_dummies_X shape : ",train_df_dummies_X.shape)
print("train_df_dummies_Y shape : ",train_df_dummies_Y.shape)


# ### Converting to numpy arrays so tensorflow variables can pick it up

# In[ ]:


trainX_num = train_df_numerical_X.as_matrix()
trainY_num = train_df_numerical_Y.as_matrix()

trainX_dummies = train_df_dummies_X.as_matrix()
trainY_dummies = train_df_dummies_Y.as_matrix()

print("trainX_num.shape = ",trainX_num.shape)
print("trainY_num.shape = ",trainY_num.shape)
print("trainX_dummies.shape = ",trainX_dummies.shape)
print("trainY_dummies.shape = ",trainY_dummies.shape)


# In[ ]:


# Reshaping the rank 1 arrays formed to proper 2 dimensions
trainY_num = trainY_num[:,np.newaxis]
trainY_dummies = trainY_dummies[:,np.newaxis]

print("trainX_num.shape = ",trainX_num.shape)
print("trainY_num.shape = ",trainY_num.shape)
print("trainX_dummies.shape = ",trainX_dummies.shape)
print("trainY_dummies.shape = ",trainY_dummies.shape)


# ### Tensorflow Model
# 
# Now, let's build our model. 
# We could use existing DNN classifier. But instead, we're gonna build this one with calculations ourselves.
# 2 layers. Hence, W1, b1, W2, b2 as parameters representing weights and biases to layer 1 and layer 2 respectively. 
# 
# We'll use RELU as our activation function for first layer. Why? Because it performs better in general.
# And sigmoid for the 2nd layer. Since output is going to be a binary classification, it makes sense to use sigmoid.

# In[ ]:


### Tensorflow model
def model(learning_rate, X_arg, Y_arg, num_of_epochs):
    # 1. Placeholders to hold data
    X = tf.placeholder(tf.float32, [11,None])
    Y = tf.placeholder(tf.float32, [1, None])

    # 2. Model. 2 layers NN. So, W1, b1, W2, b2.
    # This is basically coding forward propagation formulaes
    W1 = tf.Variable(tf.random_normal((20,11)))
    b1 = tf.Variable(tf.zeros((20,1)))
    Z1 = tf.matmul(W1,X) + b1             # This is also called logits in tensorflow terms
    A1 = tf.nn.relu(Z1)

    W2 = tf.Variable(tf.random_normal((1, 20)))
    b2 = tf.Variable(tf.zeros((1,1)))
    Z2 = tf.matmul(W2,A1) + b2
    A2 = tf.nn.sigmoid(Z2)

    # 3. Calculate cost
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z2, labels=Y)
    cost_mean = tf.reduce_mean(cost)

    # 4. Optimizer (Gradient Descent / AdamOptimizer ) - Using this line, tensorflow automatically does backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_mean)
    
    # 5. initialize variabls
    session = tf.Session()
    tf.set_random_seed(1)
    init = tf.global_variables_initializer()
    session.run(init)
    
    # 6. Actual loop where learning happens
    for i in range(num_of_epochs):
        _, cost_mean_val = session.run([optimizer, cost_mean], feed_dict={X:X_arg, Y:Y_arg})
        if i % 100 == 0:
            print("i : ",i,", cost : ",cost_mean_val)
            
    return session.run([W1,b1,W2,b2,A2,Y],feed_dict={X:X_arg, Y:Y_arg})


# In[ ]:


W1_tr,b1_tr,W2_tr,b2_tr,A2,Y = model(0.01, trainX_num.T, trainY_num.T, 3000)


# In[ ]:


# Validating that our formulaes were correct by checking shapes of ouput prediction
A2.shape


# In[ ]:


Y.shape


# In[ ]:


# Let's see the predictions variable
A2[:,0:5]


# **As we see, our predictions array isn't 0s or 1s. So, we must convert it to 0s / 1s. **

# In[ ]:


A2_bool = A2 > 0.5
Y_prediction_training = A2_bool.astype(int)
Y_int = Y.astype(int)


# In[ ]:


Y_int


# In[ ]:


Y_prediction_training


# In[ ]:


accuracy = (Y_prediction_training == Y_int).mean()
accuracy


# ### Awesome
# 
# 81.48% accuracy isn't bad on training dataset. That too, with just 3000 epochs!
# 
# People got near 85% with 40000 epochs. So, it's fine. This is good enough.
# 

# In[ ]:


# First, let's list our helper functions we could make from logic used above.
def convert_sigmoid_output_to_boolean_array(array, threshold):
    array = array > threshold
    return array

def convert_boolean_array_to_binary_array(array):
    array_binary = array.astype(int)
    return array_binary


# * **Let's try now with dummies (one-hot vectors) data.**
# 
# This is the time. Let's generalize the model we wrote above to take more arguments and not be specific to shapes of our X or Y.
# Also, let's now print the training accuracy in the model itself with the cost at each 100th epoch!

# In[ ]:


### Tensorflow model
def model_generic(learning_rate, X_arg, Y_arg, num_of_epochs, hidden_units, threshold):
    # 1. Placeholders to hold data
    X = tf.placeholder(tf.float32, [X_arg.shape[0],None])
    Y = tf.placeholder(tf.float32, [1, None])

    # 2. Model. 2 layers NN. So, W1, b1, W2, b2.
    # This is basically coding forward propagation formulaes
    W1 = tf.Variable(tf.random_normal((hidden_units,X_arg.shape[0])))
    b1 = tf.Variable(tf.zeros((hidden_units,1)))
    Z1 = tf.matmul(W1,X) + b1
    A1 = tf.nn.relu(Z1)

    W2 = tf.Variable(tf.random_normal((1, hidden_units)))
    b2 = tf.Variable(tf.zeros((1,1)))
    Z2 = tf.matmul(W2,A1) + b2
    A2 = tf.nn.sigmoid(Z2)

    # 3. Calculate cost
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z2, labels=Y)
    cost_mean = tf.reduce_mean(cost)

    # 4. Optimizer (Gradient Descent / AdamOptimizer ) - Using this line, tensorflow automatically does backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_mean)
    
    # 5. Accuracy methods
    predicted_class = tf.greater(A2,threshold)
    prediction_arr = tf.equal(predicted_class, tf.equal(Y,1.0))
    accuracy = tf.reduce_mean(tf.cast(prediction_arr, tf.float32))
    
    # 5. initialize variabls
    session = tf.Session()
    tf.set_random_seed(1)
    init = tf.global_variables_initializer()
    session.run(init)
    
    # 6. Actual loop where learning happens
    for i in range(num_of_epochs):
        _, cost_mean_val, accuracy_val = session.run([optimizer, cost_mean, accuracy], feed_dict={X:X_arg, Y:Y_arg})
        if i % 100 == 0:
            print("i:",i,", cost : ",cost_mean_val,", training accuracy : ",accuracy_val)
            
    return session.run([W1,b1,W2,b2,A2,Y,accuracy],feed_dict={X:X_arg, Y:Y_arg})


# In[ ]:


W1_dum,b1_dum,W2_dum,b2_dum,A2_dummies,Y_dummies,training_accuracy_val = model_generic(0.005, trainX_num.T, trainY_num.T, 3000, 100,0.5)


# In[ ]:


training_accuracy_val


# So, for when we use dummies data, accuracy goes up and down, and after 3000 epochs is somewhere near 85.52%. This is good only! 

# # Prediction on Test Data
# 
# Let's use 'numerical' vector  data only now.
# - Converting test data in the same form
# - Pass it through the network to get the value of A2
# - Concatenate this with the data and write that into csv
# - Submit the csv

# In[ ]:


test_df


# In[ ]:


test_df.isnull().any()


# In[ ]:


test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())
test_df['Cabin'] = test_df['Cabin'].fillna('UNKNOWN')


# In[ ]:


test_df.isnull().any()


# In[ ]:


# Converting to numerical data
test_df_numerical = test_df.copy()
for col in categorical_cols:
    test_df_numerical[col] = test_df_numerical[col].astype('category')
    test_df_numerical[col] = test_df_numerical[col].cat.codes
test_df_numerical.shape


# In[ ]:


test_df_numerical.head()


# In[ ]:


import math
# Ref : https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
# Ref : https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
def predict(W1,b1,W2,b2,X):
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.maximum(Z1, 0, Z1)
    
    Z2 = np.dot(W2,A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    return A2


# In[ ]:


# Let's predict
X_test = test_df_numerical.as_matrix()
X_test.shape


# In[ ]:


W1_tr.shape


# In[ ]:


W2_tr.shape


# In[ ]:


final_prediction = predict(W1_tr,b1_tr,W2_tr,b2_tr,X_test.T)


# In[ ]:


final_prediction_int = final_prediction > 0.5
final_prediction_int = final_prediction_int.astype(int)
final_prediction_int.shape


# In[ ]:


final_survived_df = pd.DataFrame(data=final_prediction_int.T, columns=['Survived'])
final_survived_df


# In[ ]:


test_df['PassengerId']


# In[ ]:


final_df = pd.concat([test_df['PassengerId'], final_survived_df], axis=1)
final_df


# In[ ]:


# Exporting to a csv file
final_df.to_csv("output-prediction.csv", index=False)


# # One function to sum the whole notebook
# 
# Now that we've reached here, we would want to execute the same notebook for different values of hyperparameters - to see how well our ouput csv file does on the leaderboard, and if we can improve our position. 
# For this, I've tried to utilize the helper functions we kept writing above and made one method which does everything from loading data, to fixing null values, to evaluating the model and then predicting and outputing a csv file. 
# Ultimately, you could just call this method with a range of hyperparameters, and let it do its magic. I'm gonna do the same on a 

# In[ ]:


# helper exercise which does the whole thing for any training dataframe given 
def execute_steps_for_titanic(columns_to_use, output_file_name, learning_rate=0.01, num_of_epochs=3000, hidden_units=50, threshold_for_output=0.5, ):
    # read data
    training_df_orig = pd.read_csv("../input/train.csv")
    testing_df_orig = pd.read_csv("../input/test.csv")
    # get X and Y separated
    train_df_Y = training_df_orig['Survived']
    train_df_X = training_df_orig[columns_to_use]
    test_df_X = testing_df_orig[columns_to_use]
    # fix missing data
    categorical_columns = find_categorical_columns(train_df_X)
    replace_values_dict = {'Embarked':'S', 'Cabin':'UNKNOWN'}
    for col in columns_to_use:
        num_of_NaN_rows = get_num_of_NaN_rows(train_df_X)[col]
        num_of_NaN_rows_test = get_num_of_NaN_rows(test_df_X)[col]
        if(num_of_NaN_rows > 0):
            print("Filling NaN values for column:",col)
            if col not in categorical_columns:
                train_df_X[col] = train_df_X[col].fillna(train_df_X[col].mean())
            else:
                train_df_X[col] = train_df_X[col].fillna(replace_values_dict[col])
        if(num_of_NaN_rows_test > 0):
            print("Filling NaN values for column:",col," in test data")
            if col not in categorical_columns:
                test_df_X[col] = test_df_X[col].fillna(test_df_X[col].mean())
            else:
                test_df_X[col] = test_df_X[col].fillna(replace_values_dict[col])
    print("Fixed NaN values in training and testing data.")
    # convert categorical to numerical data
    train_df_X_num = convert_categorical_column_to_integer_values(train_df_X)
    test_df_X_num = convert_categorical_column_to_integer_values(test_df_X)
    # Get numpy arrays for this data
    train_X = train_df_X_num.as_matrix()
    test_X = test_df_X_num.as_matrix()
    train_Y = train_df_Y.as_matrix()
    # fix rank-1 array created
    train_Y = train_Y[:,np.newaxis]
    # call model and get values 
    W1,b1,W2,b2,A2,Y,final_tr_accuracy = model_generic(learning_rate, train_X.T, train_Y.T, num_of_epochs, hidden_units, threshold_for_output)
    print("Final training accuracy : ",final_tr_accuracy)
    # get prediction and save it to output file
    prediction = predict(W1,b1,W2,b2,test_X.T)
    # if prediction value > threshold, then set as True, else as False
    prediction = prediction > threshold_for_output
    # Convert the True/False array to a 0 , 1 array
    prediction = prediction.astype(int)
    # Convert back to dataframe and give the column name as 'Survived'
    prediction_df = pd.DataFrame(data=prediction.T, columns=['Survived'])
    # Make a final data frame of the required output and output to csv
    final_df = pd.concat([testing_df_orig['PassengerId'], prediction_df], axis=1)
    final_df.to_csv(output_file_name+"_tr_acc_"+"{0:.2f}".format(final_tr_accuracy)+"_prediction.csv", index=False)
    print("Done.")


# In[ ]:


# Let's try this once?
# All this while, we kept including Name and PassengerId as 2 important columns however in real life, they actually don't really matter in deciding whether a person would live or not. 
# So, now let's check without them.
columns_to_use = ['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
execute_steps_for_titanic(columns_to_use, "bhavul", learning_rate=0.005, num_of_epochs=5000, hidden_units=30, threshold_for_output=0.5)


# In[ ]:




