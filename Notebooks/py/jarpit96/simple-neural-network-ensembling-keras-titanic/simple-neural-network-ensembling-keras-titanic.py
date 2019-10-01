#!/usr/bin/env python
# coding: utf-8

# # A simple Neural Network in Keras with Basic Ensembling on Titanic Problem.

# This is a a very simple neural network in Keras and uses a simple ensemble technique i.e. voting. This is my first problem on Kaggle, so any suggestions are appreciated.
# ## 1. Import Packages
# We begin by import relevant packages. The packages we will be using are:
# + Pandas: For easy dataset handling
# + Numpy: For efficient manipulation of matrices and arrays
# + Keras: For building the neural network model
# + scikit-learn: For cross validation

# In[30]:


#Import Packages
import pandas as pd
import numpy as np

#Keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

#Sklearn
from sklearn.model_selection import StratifiedKFold


# ## 2. Data Loading and Preprocessing
# We load the data from csv files saved in input folders using read_csv function of pandas to get train dataframe and test dataframe.

# In[31]:


#Load Data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# Display the column names from the train dataframe. The dataset consists of following features, where Survived is the target class and other columns are input features. 

# In[32]:


print(train_df.columns.values)


# Display the first few records of train dataframe to analyze manually.

# In[33]:


train_df.head()


# Perform the same steps for test dataframe and analyze.

# In[34]:


print(test_df.columns.values)
test_df.head()


# We drop Name, Ticket, Cabin features from train and test dataframes as it will be easier to work with numerical and categorical data. Also, drop PassengerId from train dataframe as it does not provide any information for the model. We do not drop PassengerId from test dataframe as it will be required to make the submission file further.
# We display the shape of train_df and test_df before and after dropping the columns.

# In[35]:


#Dropping columns
print("Before: ", train_df.shape, test_df.shape)
train_df = train_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
print("After: ", train_df.shape, test_df.shape)


# Create a function to display the number of empty or null cells for each column. This helps us to either fill in the missing values or drop the records with missing values. The function loops over the columns and prints the number of null cells in that column.

# In[36]:


def print_empty_cells(dataset):
    print("Empty Cells->")
    cols = dataset.columns.values
    for col in cols:
        print(col, dataset[col].isnull().sum())


# Display the empty cells for train_df. 

# In[37]:


#print empty/null number of cells in each column
print_empty_cells(dataset = train_df)


# Age and Embarked columns contain null values, we have to replace these missing values.
# For Age column, we can use the median age value and for Embarked column we pick the most frequent value i.e. 'S'.

# In[38]:


#Fill Null or Empty values with default values
freq_port = train_df.Embarked.dropna().mode()[0]
print(freq_port)
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].dropna().median()) # Median Age
train_df['Embarked'] = train_df['Embarked'].fillna(freq_port) #Most Frequent Port S


# Repeat the same process for test_df.

# In[39]:


#print empty/null number of cells in each column
print_empty_cells(dataset = test_df)


# Age and Fare columns contain null values, we have to replace these missing values or drop records with missing values.
# For Age column, we can use the median age value and for Fare column we drop the record.

# In[40]:


#Fill Null or Empty values with default values
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].dropna().median()) # Median Age
test_df['Fare'] = test_df['Fare'].dropna() 


# Now, we convert the categorical columns Sex and Embarked from string values to numerical values. We create two maps and change the values from string to numbers.

# In[41]:


#Maps for Sex and Embarked
sex_mapping = {'male' : 0, 'female' : 1}
embarked_mapping = {'S' : 0, 'Q' : 1, 'C' : 2}

#Categorical to Numerical Sex and Embarked, Fill NA with Most Frequent Female and S
train_df['Sex'] = train_df['Sex'].map(sex_mapping)
train_df['Embarked'] = train_df['Embarked'].map(embarked_mapping)
test_df['Sex'] = test_df['Sex'].map(sex_mapping)
test_df['Embarked'] = test_df['Embarked'].map(embarked_mapping)

print(train_df.head())


# Extraxt X_train, Y_Train and X_Test sets from the dataframes.

# In[42]:


#Extracting Test and Train Sets for NN, Dataframe to Numpy
X_train = train_df.drop(['Survived'], axis=1).values
Y_train = train_df['Survived'].values
X_test = test_df.drop(['PassengerId'], axis = 1).values


# ## 3. Model 
# Define the model hyper-parametrs or config parameters, like number of epochs and batch size.

# In[44]:


#Config Parameters
num_epochs = 200
num_cv_epochs = num_epochs
train_batch_size = 32
test_batch_size = 32
folds = 10


# Now, we create a simple feed forward dense model in Keras. It consists of a single hidden layer with 20 neurons and relu activation function. The output unit is a single sigmoidal neuron as it is a binary classification problem. Batch Normalization is used to normalize the inputs.
# Binary cross entropy loss and Adam optimizer is used.

# In[60]:


def get_model():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(7,)))
    m.add(Dense(20, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    return m


# We get the model from the get_model function and fit on the complete training data set.

# In[56]:


#Make Keras Model and Fit on Training Data
model = get_model()
model.fit(X_train, Y_train, epochs=num_epochs, batch_size=train_batch_size, verbose=1)


# Now, we create a cross validation function for two major purposes:
# 1. Find model accuracy on the training data without risking overfitting.
# 2. Use cross validation models for ensembling.
# 
# #### 1. Cross Validation to Determine Model Accuracy
# If we evaluate the trained model on complete train dataset then we are learning and evaluating on the same data i.e. data which is already seen the model is used to evaluate it's performance. This can lead to overfitting as it incentivises the model to perform better on the training set by compromising it's generalizing abilility. 
# **K-Fold Cross Validation** divides the dataset into k folds and learns a model on k-1 folds and evaluates on the remaining k<sup>th</sup> fold. This process is repeated for every fold and accuracy is averaged.
# 
# #### 2. Cross Validation Models for Ensembling
# The models trained in every iteration are used to make predictions on the test dataset. The output generated by prediction is the probabilities for every record to have value survived=1. These probabilities are summed over all the models in y_pred. y_pred is returned after function completion.

# In[53]:


def cross_validation(X_train, Y_train, X_test, num_cv_epochs, k = 5):
    print("------------Cross Validation--------")
    k = max(k, 2) #Minimum 2 folds
    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    cvscores = [] #metric scores/accuracy for each iteration
    y_pred = np.zeros((X_test.shape[0], 1)) #sum of predicted probabilities by models trained on different k-1 folds 
    for train, test in kfold.split(X_train, Y_train): #for every iteration 
        model_acc = get_model() #get a new keras model
        model_acc.fit(X_train[train], Y_train[train], epochs=num_cv_epochs, batch_size=train_batch_size, verbose=0) #fit/train on k-1 folds
        scores = model_acc.evaluate(X_train[test], Y_train[test], verbose=0) #evaluate on kth fold
        y_pred += model_acc.predict(X_test) #predict on test dataset for soft voting/ensembling
        print("%s: %.2f%%" % (model_acc.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%%" % (np.mean(cvscores)))
    return y_pred


# Cross Validation is called to generate the Y_pred predicted probabilities.

# In[57]:


#Evaluate Model On Train Data To Find Model Accuracy
#KFold Cross Validation
Y_pred = cross_validation(X_train=X_train, Y_train=Y_train, X_test=X_test, num_cv_epochs=num_cv_epochs, k = folds)


# The probabilities are predicted over the model trained using the complete training dataset. These probabilities are added with previous probabilities from croos validation function and averaged. The probability values are then converted to integer values (0 or 1) using 0.5 as the threshold probability.

# In[58]:


#Evaluate Test Data to Get Prediction
Y_pred += model.predict(X_test, batch_size=32)
Y_pred = Y_pred.reshape((Y_pred.shape[0],)) #reshape (418,1) to (418,)
Y_pred = Y_pred / (folds+1) #Average predicted probabilty
Y_pred = [int(p > 0.5) for p in Y_pred] #Converting class probabilities to Binary value


# The submission file is created using Y_pred and PassengerId from test_df and wriiten to file.

# In[59]:


#Make submission Dataframe and Save to file
submission = pd.DataFrame({ "PassengerId": test_df["PassengerId"], "Survived": Y_pred })
submission.to_csv('submission.csv', index=False)


# In[ ]:




