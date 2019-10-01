#!/usr/bin/env python
# coding: utf-8

# # This is my first Machine Learning Project. Very Excited

# In[ ]:


#Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras import Input
from keras import Model
from keras.optimizers import Adam, SGD
import os


# In[ ]:


#Data set paths
train_path = "../input/train.csv"
test_path = "../input/test.csv"


# In[ ]:


#Reading the data
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)


# ## Analyse the data

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# #### From the two above statements, we notice something strange about the number of entries(count) in both the training set and the testing set. The count should be the same for all the columns, but it's not. Age column values are less on both sets. Let's see why.

# In[ ]:


df_train[["Age"]]


# In[ ]:


df_test[["Age"]]


# #### You see it? I do.
# #### The issue that's causing problems with the values under the Age column in our datasets is that some passengers ages were either not recorded, or lost. These are values marked "NaN", which stands for Not a Number.

# #### To resolve this issue(At least for now), we'll have to preprocess the data

# ## Preprocess the data

# ### To counter this problem, we'll have to either disregard the rows with missing age values or impute(impose) some data in those rows.

# ### For now, we'll just impute data(mean average) in those rows

# In[ ]:


#Our Important Features(For now) are SibSp, Class, Sex, Survived(target output) and Age
train_data = df_train[["SibSp", "Parch", "Pclass", "Sex", "Age", "Survived"]].replace("male", 1).replace("female", 0)
test_data = df_test[["SibSp", "Parch","Pclass", "Sex", "Age"]].replace("male", 1).replace("female", 0)
train_data.fillna(train_data.mean(),inplace=True)
test_data.fillna(test_data.mean(),inplace=True)
#The .fillna method will impute mean values in cells that have no data
#The .replace method will change male strings to 1 and female strings to 0


# In[ ]:


x_train = train_data[["SibSp","Parch", "Pclass", "Sex", "Age"]]
x_test = test_data[["SibSp", "Parch", "Pclass", "Sex", "Age"]]


# In[ ]:


y_train = train_data[["Survived"]]


# ## The Artificial Neural Netwrok

# In[ ]:


n_inputs = len(x_train.columns) #Number of inputs(columns/nodes/neurons)
n_outputs = len(y_train.columns)#Number of outputs(columns/nodes/neurons)


# In[ ]:


inputs = Input(shape = (n_inputs, ))
hl = Dense(units=10, activation="relu")(inputs) #Hidden Layer 1
hl = Dense(units=10, activation="relu")(hl) #Hidden Layer 2
outputs = Dense(units=1, activation="sigmoid")(hl)


# In[ ]:


model = Model(inputs, outputs)
model.summary()


# In[ ]:


optimizer = Adam(lr=0.0001,decay=1e-6) #Learning rate decay
pi = 3.14159265359
golden_ratio = 1.61803398875
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=optimizer)
history =  model.fit(x_train, y_train, epochs=int(len(x_train)*pi),validation_split = 0.2, batch_size=int(len(x_train)/golden_ratio), verbose=2)
prediction = model.predict(x_test, batch_size = int(len(x_train)/golden_ratio))


model.save("Kaggle Titanic Comp.h5")


# ## Plot History Summaries for loss and accuracy against number of epochs

# In[ ]:


'''
Source: Jason Brownlee
Site: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
'''

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# In[ ]:


'''
Source: Jason Brownlee
Site: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
'''
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# In[ ]:




