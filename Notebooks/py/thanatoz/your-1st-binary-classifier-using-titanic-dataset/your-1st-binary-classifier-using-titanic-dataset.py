#!/usr/bin/env python
# coding: utf-8

# # Binary classification using Keras  
# ### A systematic approach to your first competition submission for
# ### Titanic: Machine Learning from Disaster
# 
# This notebook contains the systematic approach to solving [Titanic: Machine Learning from Disaster](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjXzcGbiJPcAhWCfSsKHTbaBkAQFggoMAA&url=https%3A%2F%2Fwww.kaggle.com%2Fc%2Ftitanic&usg=AOvVaw3u32GjGoKUk5tZpQiCIcHU) contest using Keras framework.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# We will apply deep learning to achieve maximum possible accuracy.

# We will first process the data using *Pandas* framework. Clean it, normalize and then finally we will use it in our network for training and testing.

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# We can see that we have 3 files in our contest. We will process the train.csv and test.csv together to maintain a homogenity all through our data

# In[ ]:


# Reading the data
df=pd.read_csv('../input/train.csv')
dft=pd.read_csv('../input/test.csv')


# In[ ]:


# Let's see what all data is there in our training dataset
df.head()


# In[ ]:


# And the data in our testing dataset
dft.head()


# Now, to preprocess our data, we will perform the following options:
# - Fill all the empty cells will 0 or col.mean() (Here we have chosen 0)
# - convert the string to integers such that male=0 and female=1
# - As the 'PassengerId' column is not useful for training, we will discard that column
# - Discard all the useless columns that does not contain arithmetic data and keep the rest
# - Store the reduced dataframe into new dataframe
# and finally visualize the data

# In[ ]:


df=df.fillna(0)
dft=dft.fillna(0)
df=df.replace(['male','female'],[0,1])
dft=dft.replace(['male','female'],[0,1])
df=df.drop(columns=['PassengerId'])
out_targets=dft['PassengerId'].values
cols=[i for i in df.describe()]
colt=[i for i in dft.describe()]
df_reduced=df[cols]
dft_red=dft[colt[1:]]
df_reduced.head()


# In[ ]:


dft_red.head()


# Seperate the training dataset and test dataset, Normaize it before and then proceed 

# In[ ]:


train_labels = df_reduced['Survived'].values
df_reduced=df_reduced.drop(columns=['Survived'])
train_data=df_reduced.values
test_data=dft_red.values
print(train_data.shape, train_labels.shape)
print(test_data.shape)


# In[ ]:


mean=train_data.mean(axis=0)
std=train_data.std(axis=0)
train_data-=mean
train_data/=std

meant=test_data.mean(axis=0)
stdt=test_data.std(axis=0)
test_data-=mean
test_data/=std


# ## Its finally time to create our model
# 
# ### Creating the network
# 
# As this is supposed to classify the review into positives and negetives. There is need to only output a single value. Such type of classification is called as **Binary Classification**.
# 
# Architecture of a simple binary classifier could be as shown below:
# 
# ![network](https://cdn-images-1.medium.com/max/1600/0*hzIQ5Fs-g8iBpVWq.jpg "Network Architecture")
# 
# We would be using *keras Sequential* model for this purspose which could be imported using `from keras import models` and the layers in the model could be defined  by calling layers `from keras import layers`

# In[ ]:


from keras import models, layers, optimizers


# ##### Split into training and testing data and spliting the validation set
# Split complete data into training and testing data using a sklearn function `train_test_data`
# to maintain homogenous distribution, we will also shuffle our dataset
# Further we will take to first 200 points and create a validation dataset
# For this, we follow the following model
# ![Image](http://www.cs.nthu.edu.tw/~shwu/courses/ml/labs/08_CV_Ensembling/fig-holdout.png "test,train,validation split")
# 
# or in a simpler way
# ![dataset split](https://cdn-images-1.medium.com/max/889/1*Nv2NNALuokZEcV6hYEHdGA.png)

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_data, train_labels, test_size=0.20, shuffle=True)


# In[ ]:


print(x_train.shape, x_test.shape)


# In[ ]:


index=200
x_val=x_train[:index]
y_val=y_train[:index]
partial_x_train = x_train[index:]
partial_y_train=y_train[index:]


# ### Importing all the imports for performance visualization

# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set_style('dark')


# ### The network architecture
# 
# The model contains two fully connected hidden layers each containing 16 units.
# ```
# model.add(layers.Dense(10, activation='relu', input_shape=(6,)))
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# ```
# for the first two layers, we have used the *ReLU (Rectified Linear Unit)* and for the last output layer, we will be using the *Sigmoid* activation
# 
# ![activation function](http://numpydl.readthedocs.io/en/latest/_images/init_0.svg "Sigmoid and ReLU")
# 
# As we are working on a smaller dataset with somewhat larger network, our model could overfit and its accuracy could be affected.
# In order to overcome this, we will introduce a regulaization technique called as Dropout here.
# Dropout could be visualized as follows
# ![Dropout](https://www.learnopencv.com/wp-content/uploads/2018/06/dropoutAnimation.gif)

# In[ ]:


model=models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(6,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(1, activation='sigmoid'))


# Finally after deciding the right number of bones to pick, we are ready with our structure. The only need now is to put everything together and create a create a final body of the model.
# A few things are also required beforehand compiling the model. These are:
# + [Optimizer](http://www.deeplearningbook.org/contents/optimization.html)
# + [Loss function](https://www.quora.com/What-does-loss-mean-in-deep-neural-networks)
# + [Metrics](https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/)
# 
# For the choice of optimizer, we have chosen 
# - [*RMSprop*](https://medium.com/100-days-of-algorithms/day-69-rmsprop-7a88d475003b) as our optimizer function
# - [*Binary_crossentropy*] as our loss function
# 
# We will need to set all our hyperparameters in our optimizer before using it in our network

# In[ ]:


sgd = optimizers.SGD(lr=0.001, decay=1e-6 ,momentum=0.9)
# adm = optimizers.Adam(lr=0.1, decay=1e-6)
model.compile(optimizer=sgd,
             loss='binary_crossentropy',
             metrics=['accuracy'])


# After compiling our model, we will need to fit our data into the model
# The fitting of data can be done directly but it is always better to create a validation set and feed the model with the remaining data so that easy model evaluvation could be done.
# We will follow both the approaches and see the difference in the loss matrics graph.

# #### Data fitting into the model

# In[ ]:


history = model.fit(partial_x_train, partial_y_train,
                   epochs=500,
                   batch_size=32,
                   validation_data=(x_val,y_val),
                   verbose=0)


# ### Visualizing the model performance
# 
# metrics visualizations could be carried out using matplotlib pyplot

# In[ ]:


hist = history.history
acc=hist['acc']
a=200
b=len(acc)
val_loss=hist['val_loss'][a:b]
loss=hist['loss'][a:b]
val_acc=hist['val_acc']
epc = range(1,(b-a)+1)

plt.figure(figsize=(15,4))
plt.clf()
plt.subplot(1,2,1)
plt.plot(epc, loss, 'r', label='Training_loss')
plt.plot(epc, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

epc=range(1,len(acc)+1)
plt.subplot(1,2,2)
plt.plot(epc, acc, 'r', label='Training_acc')
plt.plot(epc, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()


plt.show()


# We can evaluvate our model now on the test data we split in our previous steps

# In[ ]:


model=models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(6,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.001, decay=1e-6 ,momentum=0.9)
# adm = optimizers.Adam(lr=0.1, decay=1e-6)
model.compile(optimizer=sgd,
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(train_data, train_labels,
                   epochs=500,
                   batch_size=32,
                   verbose=0)


# In[ ]:


results = model.evaluate(x_test, y_test)
results


# In[ ]:


predictions=model.predict(test_data)
# predictions
pred=[1 if predictions[i]>0.25 else 0 for i in range(len(test_data))]


# In[ ]:


pred


# In[ ]:


res=pd.DataFrame()


# In[ ]:


pd.read_csv('../input/gender_submission.csv').head()


# In[ ]:


res['PassengerId']=out_targets
res['Survived']=pred
res.head()


# In[ ]:


res.to_csv('Submission.csv', index=False)


# In[ ]:




