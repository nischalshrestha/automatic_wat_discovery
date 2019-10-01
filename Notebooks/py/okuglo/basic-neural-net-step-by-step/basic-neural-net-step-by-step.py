#!/usr/bin/env python
# coding: utf-8

# # Deep Learning survival
# 
# 
# 
# My Idea for this Kernel was that I wanted to do an absolute minimum of Data analytics and preprocessing and to let a neural net do the heavy lifting.
# 
# In this notebook you will find that I first did some basic preprocessing to prepare the data for the Model.
# 
# I then built a neural net using Keras(3 Layers fully connected)
# 
# The final Result is an accurancy around xx percent.
# 
# At some points the code is not very efficient however I decided to focus on readability.
# 
# This is my first go on a Kaggle project so I'm looking forward to your Feedback.

# In[ ]:


import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn import preprocessing


# In[ ]:


#Loading the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# # Preprocessing

# Lets first take a quick look:

# In[ ]:


train.head(5)


# So there are some values which do not seem critical for survival, so lets just drop these

# In[ ]:


del train['Name']
del train['Ticket']
del train['Cabin']
del train['PassengerId']


# The same has to be done for test

# In[ ]:


del test['Name']
del test['Ticket']
del test['Cabin']
del test['PassengerId']


# In[ ]:


list(train)


# For Convenience I move the survive Column to the end

# In[ ]:


survive = train['Survived']
train.drop(labels=['Survived'], axis=1,inplace = True)
train.insert(7, 'Survived', survive)


# There are some Colums which contain NaN values. This will hurt us later on so lets see which columns we have to clean up:

# In[ ]:


train.isnull().any()


# So lets just Repalce the nan Values with a placeholder

# In[ ]:


#replaces NaN in embarked
train["Embarked"] = train["Embarked"].fillna("N")
test["Fare"] = test["Fare"].fillna("N")


# For the age I assume, that people with NaN values for age have the average age of the crew. So I calculate that and replace the values.
# 
# Here i actually made an aussmtion, which is that people without age specification are probably passengers with cheap tickets or no right identification. So I assumed that they should be younger than the average.
# 
# I tested this later on and found, that it does not amke a difference in model performance

# In[ ]:


#Average Age
av_age = train["Age"]
av_age = av_age.mean()
#Age correction - 8 years
av_age = av_age-8
print(av_age)


# In[ ]:


#Replace NaN with the average Age
train["Age"] = train["Age"].fillna(av_age)
test["Age"] = test["Age"].fillna(av_age)
test=test.fillna(0)


# As a next step I replace alle the strings. As the neural net can only accept numbers

# In[ ]:


#maps sex
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})

#maps embarked
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1,'Q': 2, 'N':3})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1,'Q': 2, 'N':3})


# So lets's take a look at the resuls:

# In[ ]:


train.head(10)


# In[ ]:


test.head(10)


# When running the code I found that there was a string left somewhere in the test data. Here I'm correcting this

# In[ ]:


from ipykernel import kernelapp as app

test.columns = range(test.shape[1])
train.columns = range(train.shape[1])
test_norm=test
train_norm=train
test_norm = test.convert_objects(convert_numeric=True)
train_norm = train.convert_objects(convert_numeric=True)


# As a next step I'm converting the dataframes to numpy, which is the expected format of Keras

# In[ ]:


#pandas to numpy
trainnum = train_norm.as_matrix([0,1,2,3,4,5,6])
testnum = test_norm.as_matrix([0,1,2,3,4,5,6])

labels = train_norm.as_matrix([7])
#make printing numpy pretty
np.set_printoptions(precision=3)


# # Normalization
# This step is optional and the model still funtions if it is removed. However it has increased the models Performance by 4% during my testing. This does make sense. Because Higher Values mean that the impact on the model is higher. This means that without Normalization the Age has a much bigger impact than gender. Normalization helps the model to put things in the right perspective.

# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
trainnum_norm = min_max_scaler.fit_transform(trainnum)


# In[ ]:


trainnum_norm [3]


# In[ ]:


testnum = np.nan_to_num(testnum)


# In[ ]:


min_max_scaler2 = preprocessing.MinMaxScaler()
testnum_norm = min_max_scaler2.fit_transform(testnum)


# In[ ]:


testnum_norm[3]


# # Let the learning begin!

# In[ ]:


import tensorflow as tf
import keras    


# In[ ]:


from keras.models import Sequential


# In[ ]:


np.broadcast(trainnum_norm).shape


# This is where the model is defined. I decided to go for 3 Layers. 

# In[ ]:


from keras.layers import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(units=6240, input_dim=7))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=3000))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(units=128))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(units=1))
model.add(Activation('sigmoid'))


# This is where the Hyperparameters are set. I'm using a fairly low learning rate, as the model learns pretty quickly anyway. 

# In[ ]:


from keras import optimizers

opt=keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# This is where the training happens. During my experimentation I found taht roughly 100 epochs do lead to convergence. However for a final run I've increased the number of epochs. During parameter optimization I've held back 10% of the data for Validation. However for the final run I've used all of the data.

# Batch size is another highly relevant parameter. Huge batch sizes make model a bit more robust, while small sizes make the results better. However this can lead to overfitting. I found 25 to be a good compromise

# In[ ]:


model.fit(trainnum_norm, labels, epochs=25, batch_size=25, validation_split=0.1)

scores = model.evaluate(trainnum_norm, labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# The next step is the prediction

# In[ ]:


survival = model.predict(testnum_norm, verbose=1)


# Here I am reshaping the data to gereate the right output for Kaggle

# In[ ]:



survived = (survival + 0.5 ).astype("int")
ids = np.asarray(list(range(892,1310)))

survive = survived.reshape(418) 

output = pd.DataFrame({ 'PassengerId' : ids,  'Survived': survive }, index =(range(891,1309)) )


# In[ ]:


output.head(8)


# In[ ]:


output.to_csv('../working/submission.csv', index=False)


# In[ ]:




