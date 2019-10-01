#!/usr/bin/env python
# coding: utf-8

# # Titanic: Simplified Solution for Beginners Using Dense Neural Networks
# **[Renan Gomes Barreto](https://www.kaggle.com/renangomes)**  -  *May, 2018*
#  
# ![Titanic](https://media.giphy.com/media/Uj3SeuVfg2oCs/giphy.gif)
# 
# ## Introduction
# 
# This notebook contains a brief introduction on how to create a multi-layered Neural Network and solve the Titanic problem using a simple model in Keras. Density models, also called Multi-layer perceptrons (MLP), can be used as a basis and, from them, more complex models can be constructed.
# 
# In this notebook, we will create a binary classifier using the data of passengers of the Titanic. With this dataset, we want to predict whether or not the passenger survived the shipwreck.
# ****
# In order to solve problems with Neural Networks, just like any machine learning problem, it is first necessary to understand the problem and, above all, to understand the data before elaborating an architecture. This process is extensive and often comprises most of the work. In this way, this notebook aims to introduce some important points and is organized as follows:
# 
# * [Introduction](#Introduction)
# * [The Problem](#The-Problem)
# * [Loading the dataset](#Loading-the-dataset)
# * [Preprocessing](#Preprocessing)
# * [Implementing the Neural Network](#Implementing-the-Neural-Network)
# * [Results](#Results)
# * [Conclusion](#Conclusion)

# ## The Problem
# 
# In this problem, information from the Titanic passengers will be used as a database to identify which passengers have survived. On the Titanic, one of the reasons that caused the wreck was that there were not enough lifeboats for the passengers and the crew. Among the passengers, some groups of people were more likely to survive than others, such as women, children, and the upper class. In this way, the problem is to use a Neural Network to identify which people could survive.

# ## Loading the Dataset

# ### Reading the dataset files
# 
# To start, you must analyze the dataset's input attributes, their types, and the target attribute. This can be done through Pandas, an important Python library for parsing and preprocessing data.

# In[ ]:


import numpy as np
np.random.seed(10)

import pandas as pd 

train = pd.DataFrame(pd.read_csv("../input/train.csv", index_col=[0], header=0))
test  = pd.DataFrame(pd.read_csv("../input/test.csv", index_col=[0], header=0))
display(train.head())


# ## Preprocessing
# 
# The Name, Ticket and Cabin columns appear to be unique to the passenger, so we will discard them. On closer examination, we could use these columns to improve data or even deduce missing data.

# In[ ]:


train.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
test.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
display(train.head())


# ### Handling missing Data
# 
# Missing data is a serious problem in machine learning. Somehow we should handle them. The easiest way to solve this issue is simply by deleting all the rows in the dataset that have this data or by replacing them with a fixed value. In our case, for the Age and Fare,  numeric columns, we replace the missing data by the mean. The SibSp and Parch columns had their data replaced with -1.

# In[ ]:


train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Fare'].fillna(train['Fare'].mean(), inplace=True)
train['SibSp'].fillna(-1, inplace=True)
train['Parch'].fillna(-1, inplace=True)

test['Age'].fillna(train['Age'].mean(), inplace=True)
test['Fare'].fillna(train['Fare'].mean(), inplace=True)
test['SibSp'].fillna(-1, inplace=True)
test['Parch'].fillna(-1, inplace=True)


# ### Encoding categorical columns
# 
# The Pclass, Sex, and Embarked columns appear to be categorical. These columns should be mapped to numbers, and each possible value should preferably become a new binary column. This can be done easily by using the pandas get_dummies function.

# In[ ]:


train = pd.get_dummies(train, dummy_na=True, columns=['Pclass', 'Sex', 'Embarked']).astype(float)
test = pd.get_dummies(test, dummy_na=True, columns=['Pclass', 'Sex', 'Embarked']).astype(float)

display(train.head())
display(test.head())


# ## Implementing the Neural Network

# ### Separating attributes from output
# 
# The output attributes will be separated. In addition, we separated the original training dataset into two (training and validation).

# In[ ]:


X_train = train.drop(columns=["Survived"])[:-120]
y_train = train["Survived"][:-120]

X_val = train.drop(columns=["Survived"])[-120:]
y_val = train["Survived"][-120:]

X_test = test

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_val: ",   X_val.shape)
print("y_val: ",   y_val.shape)
print("X_test: ",   X_test.shape)


# ### Model definition
# 
# We will create a simple model using Keras. Feel free to change the number of neurons, layers, activation functions, etc.

# In[ ]:


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.summary()


# ### Training
# 
# We will train the network for 1 epochs with a batch size of 32. If you want to see statistics during training, activate the verbose parameter.

# In[ ]:


import time

epochs = 1750
start_time = time.time()

history = model.fit(X_train.as_matrix(), y_train.as_matrix(), epochs=epochs, batch_size=32, 
                    validation_data=(X_val.as_matrix(), y_val.as_matrix()), verbose=0, shuffle=True)

print("Time spent: %d seconds" % (time.time() - start_time), "\r\nEpochs: %d" % (epochs))


# ### Training Charts

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['acc'], color="r")
plt.plot(history.history['val_acc'], color="g")
plt.title('Training Chart')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

plt.plot(history.history['loss'], color="r")
plt.plot(history.history['val_loss'], color="g")
plt.title('Training Chart')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()


# ## Results

# ### Confusion Matrix - Training and Validation
# 
# In order to understand the training result, we will use the accuracy_score and confusion_matrix functions from the sklearn library.
# 
# Remember that the variables y_train and X_train are Pandas dataframe, so we will usually have to use the as_matrix() function before using them.

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np

print("Accuracy (Training dataset):", accuracy_score(y_train.as_matrix(), np.round(model.predict(X_train.as_matrix()))), "\r\n")

confusionMatrixDF = pd.DataFrame( confusion_matrix(y_train.as_matrix(), np.round(model.predict(X_train.as_matrix()))),
                                 index=('Survivor', 'Victim'), columns=('Survivor', 'Victim'))

heatmap = sns.heatmap(confusionMatrixDF, annot=True, fmt="d", cmap="Blues",  vmin=0)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("Accuracy (Validation dataset):", accuracy_score(y_val.as_matrix(), np.round(model.predict(X_val.as_matrix()))), "\r\n")

confusionMatrixDF = pd.DataFrame( confusion_matrix(y_val.as_matrix(), np.round(model.predict(X_val.as_matrix()))),
                                 index=('Survivor', 'Victim'), columns=('Survivor', 'Victim'))

heatmap = sns.heatmap(confusionMatrixDF, annot=True, fmt="d", cmap="Blues",  vmin=0)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# ### Preparing the file for submission

# In[ ]:


y_test_pred = model.predict(X_test.as_matrix())

X_test_submission = X_test.copy()
X_test_submission['Survived'] = np.round(y_test_pred).astype(int)
X_test_submission['Survived'].to_csv('submission.csv', header=True)


# ## Conclusion
# 
# In this notebook, we show a simple solution to the Titanic Dataset problem.
# 
# It was implemented a Dense Neural Network with two layers that obtained a satisfactory accuracy in the validation dataset.
