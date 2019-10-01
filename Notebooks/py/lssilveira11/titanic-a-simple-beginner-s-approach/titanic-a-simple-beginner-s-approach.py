#!/usr/bin/env python
# coding: utf-8

# # Titanic survivors: a simple beginner's approach
# 
# 
# This is my first project in data science and machine learning. First, I'll analize the data and try to identify correlations, to get some intuition in how can I use this data to predict survivors of Titanic. Then, I'll build a simple neural net using TensorFlow, to make the predictions and then submit my results.
# 
# Before start, some background about the survivors of this tragedy :
# 
# > Fewer than a third of those aboard Titanic survived the disaster. Some survivors died shortly afterwards; injuries and the effects of exposure caused the deaths of several of those brought aboard Carpathia. The figures show stark differences in the survival rates of the different classes aboard Titanic. Although only 3% of first-class women were lost, 54% of those in third class died. Similarly, five of six first-class and all second-class children survived, but 52 of the 79 in third class perished. The differences by gender were even bigger: nearly all female crew members, first and second class passengers were saved. Men from the First Class died at a higher rate than women from the Third Class. In total, 50% of the children survived, 20% of the men and 75% of the women.
# 
# Reference: [RMS Titanic - Survivors and victims](https://en.wikipedia.org/wiki/RMS_Titanic#Survivors_and_victims)
# 
# This give us some initial thoughts about what data will have more effect on the activations of the neural network.

# ## Import Datasets
# 
# Let's import the data. As explained in the competition overview, there's two CSV files containing the data to train and test the neural network. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl

import tensorflow as tf
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# ## Analyzing train dataset
# 
# For now, let's see the train data. Here is the first 10 records:

# In[ ]:


train_data.head(10)


# According with the history, the **Sex**, **Age** and **Pclass** data probably will be the most important, but I have some thoughts that Fare and Embarked could help. Also, some columns have NaN values that maybe could decrease performance.
# 
# Let's see the counts:

# In[ ]:


train_data.info()


# As expected, there's little data about Cabin, but Age has several missings too. We will have to cleanup the data use it. The Fare data was filled for all passengers, what is good.
# 
# We can also see a description of the data:

# In[ ]:


train_data.describe()


# There is a large interval in the Fare amounts and the standard deviation (49.7) is higher than mean (32.2), which indicates a high variance. The average age was about 30 years old, but 25% had up to 20 years old. More than a half of passengers was in third class.
# 
# Following, let's generate some visualizations that could brings some interpretations.

# ### Survived vs. Pclass
# 
# First, let's see how many survived or not by your respective class.

# In[ ]:


df_s = train_data[train_data.Survived==1].groupby('Pclass').size()
df_d = train_data[train_data.Survived==0].groupby('Pclass').size()

x_idx = [1,2,3]

fig = plt.figure(figsize=(12,6))

b1 = plt.bar(x_idx, df_d, width=0.5, color='red')
b2 = plt.bar(x_idx, df_s, width=0.5, bottom=df_d, color='green')

plt.title('Survivors vs. Pclass')
plt.xticks(x_idx, ('1st Class', '2nd Class', '3rd Class'))
plt.xlabel('Pclass')
plt.legend((b1[0], b2[1]), ('Not survived', 'Survived'))


plt.show()


# More than 75% of third class passengers didn't survived, but that was the class with most passengers. On the other hand, more than a half of the first class  passengers survived. This follows the history, as mentioned in background at the beginning.

# ### Survived vs. Gender
# 
# Another information we have is that survivors is mostly women, so let's plot this data:

# In[ ]:


df = train_data[['Survived','Sex']].groupby(['Sex', 'Survived']).Sex.count().unstack()

fig, ax = plt.subplots(figsize=(10,6))
cmap = mpl.colors.ListedColormap(['red', 'green'])

df.plot(kind='Bar', stacked=True, title='Survived vs. Gender', ax=ax, colormap=cmap, width=0.5)


# The most of passengers was men. They also represent the most of the deaths, as expected.

# ## Survived vs Embarked vs Pclass
# 
# Another thought is that the port where they embarked could influence, in some way their survival.

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

emb_surv = train_data[['Survived','Embarked']]
emb_surv = emb_surv.groupby(['Embarked','Survived']).Embarked.count().unstack()
emb_surv.plot(kind='Bar', stacked=True, ax=ax[0], colormap=cmap, width=0.5, title='Survived vs Embarked')

emb_pclass = train_data[['Pclass','Embarked']]
emb_pclass = emb_pclass.groupby(['Embarked','Pclass']).Pclass.count().unstack()
emb_pclass.plot(kind='Bar', stacked=True, ax=ax[1], title='Embarked vs Pclass')


# That was an interesting plot. The first plot shows that most of passengers embarked at Southampton and this port had most of the deads, but almost everyone that embarked at Queenstown not survived. In the second plot, we can see that almost everyone that embarked at Queenstown were from 3rd class. This correlation shows that maybe this information will be very important to predict that a passenger not survived.

# ## Histograms
# 
# Here some histograms, trying to identify some correlations. First, let's see an age histogram of survived and non-survived.

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(12,6))
fig.subplots_adjust(wspace=.05)

df_survived = train_data[(train_data.Survived==1) & (train_data.Age > 0)].Age;
df_deads = train_data[(train_data.Survived==0) & (train_data.Age > 0)].Age;

ax[0].hist(df_survived, bins=10, range=(0,100))
ax[0].set_title('Histogram of Age (survived)')
ax[1].hist(df_deads, bins=10, range=(0,100))
ax[1].set_title('Histogram of Age (not survived)')

plt.show()


# There's clearly more survivors with ages up to 10 years old. That was expected, because women and children were prioritized.

# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(16,8))
fig.subplots_adjust(wspace=.05, hspace=.3)

df_1c = train_data[train_data.Pclass == 1]
df_2c = train_data[train_data.Pclass == 2]
df_3c = train_data[train_data.Pclass == 3]

ax1[0].hist(df_1c[df_1c.Survived == 1].Age, bins=10, range=(0,100))
ax1[0].set_title('1st Class (survived)')
ax1[0].set_ylabel('Amount')
ax1[1].hist(df_2c[df_2c.Survived == 1].Age, bins=10, range=(0,100))
ax1[1].set_title('2nd Class (survived)')
ax1[2].hist(df_3c[df_3c.Survived == 1].Age, bins=10, range=(0,100))
ax1[2].set_title('3rd Class (survived)')

ax2[0].hist(df_1c[df_1c.Survived == 0].Age, bins=10, range=(0,100))
ax2[0].set_title('1st Class (not survived)')
ax2[0].set_xlabel('Age')
ax2[0].set_ylabel('Amount')
ax2[1].hist(df_2c[df_2c.Survived == 0].Age, bins=10, range=(0,100))
ax2[1].set_title('2nd Class (not survived)')
ax2[1].set_xlabel('Age')
ax2[2].hist(df_3c[df_3c.Survived == 0].Age, bins=10, range=(0,100))
ax2[2].set_title('3rd Class (not survived)')
ax2[2].set_xlabel('Age')

plt.show()
#train_data['Age'].hist(by=train_data['Pclass'], bins=20)


# These plots gives more refined visualization then the "Survived vs Pclass" above. We can still see that most of 3rd class passengers not survived, but we can see now that passengers between 20 and 40 years old were the most of the dead.
# 
# All the children up to 10 years old of 2nd class survived. The  1st class had older passengers than the 3rd class.

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(12,6))
fig.subplots_adjust(wspace=.05)

df_survived = train_data[train_data.Survived==1].Fare;
df_deads = train_data[train_data.Survived==0].Fare;

ax[0].hist(df_survived, bins=10)
ax[0].set_title('Histogram of Fare (survived)')
ax[1].hist(df_deads, bins=10)
ax[1].set_title('Histogram of Fare (not survived)')

plt.show()


# The fare are slightly proportional between the survived and non-survived. There was more survivors that paid above 200 than deads. This is related to the observation about "Survived vs Pclass", where we saw that most of the 1st class passengers survived.

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(16,5))
fig.subplots_adjust(wspace=0.05)

#cherbourg
ax[0].hist(train_data[train_data.Embarked == 'C'].Fare, bins=10)
ax[0].set_title('Cherbourg')
ax[0].set_xlabel('Fare')

# queenstown
ax[1].hist(train_data[train_data.Embarked == 'Q'].Fare, bins=5)
ax[1].set_title('Queenstown')
ax[1].set_xlabel('Fare')

#southampton
ax[2].hist(train_data[train_data.Embarked == 'S'].Fare, bins=10)
ax[2].set_title('Southampton')
ax[2].set_xlabel('Fare')

plt.show()

df_emb = train_data.groupby(['Embarked']).agg(['size', 'sum'])
df_emb['Fare'].head()


# As Southampton embarked the most of 3rd class passengers, it had much more passengers with low fare, between 0 and 25 bucks.

# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(16,10), sharey=True)
fig.subplots_adjust(wspace=0.05, hspace=0.2)

df_surv = train_data[train_data.Survived == 1]
df_dead = train_data[train_data.Survived == 0]

# SibSp Survived
ax1[0].hist(df_surv.SibSp, bins=10, range=(0,8))
ax1[0].set_title('SibSb Survived')

# SibSp not Survived
ax1[1].hist(df_dead.SibSp, bins=10, range=(0,8))
ax1[1].set_title('SibSb Not Survived')

# Parch Survived
ax2[0].hist(df_surv.Parch, bins=10, range=(0,6))
ax2[0].set_title('Parch Survived')

# Parch not Survived
ax2[1].hist(df_dead.Parch, bins=10, range=(0,6))
ax2[1].set_title('Parch Not Survived')


# ## Building the Neural Network
# 
# Let's build a neural network to predict. Based on the analysis above, We will only use data of Sex, Age, Pclass, Fare and Embarked values. First, let's filter the train data to get an train set just with these columns.

# In[ ]:


train_set = train_data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']]

train_set.info()


# In[ ]:


train_set.head(10)


# The columns Sex and Embarked needs to be mapped into numbers and the Age and Fare data needs to be normalized. Also, the Age and Fare columns have NaN values, that will be replaced by the mean. The Embarked has two NaN values, that will be replaced by 'S' (Southampton), because that is the port that embarked more passengers.
# 
# All of this is done bellow:

# In[ ]:


sex_map = {'male':1, 'female':2}
emb_map = {'S':1, 'C':2, 'Q':3}
min_max = preprocessing.MinMaxScaler()

def normMinMax(df_input):
    aux = df_input.values.reshape(-1,1)
    aux_norm = min_max.fit_transform(aux)
    return pd.DataFrame(aux_norm)

def dataPreparation(input):

    features = input.copy()
    
    # mappings
    features['Sex'] = features['Sex'].map(sex_map)
    features['Embarked'] = features['Embarked'].map(emb_map)
    
    features.loc[features.Embarked.isnull(), 'Embarked'] = emb_map['S']

    # normalization
    print("Mean age: ", round(features.Age.mean()))
    features.loc[features.Age.isnull(), 'Age'] = features.Age.mean()
    features['Age'] = normMinMax(features['Age'])

    print("Mean fare: ", features.Fare.mean())
    features.loc[features.Fare.isnull(), 'Fare'] = features.Fare.mean()    
    features['Fare'] = normMinMax(features['Fare'])
    
    # normalize SibSp and Parch data
    features['SibSp'] = normMinMax(features['SibSp'])
    features['Parch'] = normMinMax(features['Parch'])
    
    features['has_SibSp'] = (features['SibSp'] > 0).astype(int)
    features['has_Parch'] = (features['Parch'] > 0).astype(int)
    
    return features


# The features is the final DataFrame that we will use in training. Here is the final top 10 records:

# In[ ]:


train_features = dataPreparation(train_set)

train_labels = pd.DataFrame(train_features.pop('Survived'))

train_features.head(10)


# In[ ]:


train_features.info()


# After the preparation, we maintained the 891 records. Following, it's time to build and train the neural network.

# ## Training the network
# 
# At this point, we are going to build and train the neural network to predict the survivors.

# In[ ]:


# global parameters
epochs = 10000
learning_rate=0.00001


# In[ ]:


# building the neural network
tf.reset_default_graph()

y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
x = tf.placeholder(tf.float32, shape=[None, train_features.shape[1]], name='x')

#x_tensor = tf.contrib.layers.fully_connected(x, 2**5)

x_tensor = tf.contrib.layers.fully_connected(x, 2**6)

x_tensor = tf.contrib.layers.fully_connected(x_tensor, 2**6)
x_tensor = tf.contrib.layers.fully_connected(x_tensor, 2**6)
x_tensor = tf.layers.dropout(x_tensor)
x_tensor = tf.contrib.layers.fully_connected(x_tensor, 2**6)
x_tensor = tf.contrib.layers.fully_connected(x_tensor, 2**6)
x_tensor = tf.layers.dropout(x_tensor)
x_tensor = tf.contrib.layers.fully_connected(x_tensor, 2**6)

yhat= tf.contrib.layers.fully_connected(x_tensor, 1, activation_fn=tf.nn.sigmoid)

yhat= tf.identity(yhat, name='logits')

# usual logistic regression cost
cost = tf.reduce_mean( -y*tf.log(yhat)  - (1-y)*tf.log(1-yhat))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

# we presume survived if prediction says that a passenger has more than 50% chance of survival
prediction = (yhat> 0.5)

# accuracy
correct_pred = tf.equal(prediction, (y > 0.5))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# For this neural network, the 5 features are fully connected by 32 neurons at first layer, then 64 neurons at second layer and finally outputs one value at final layer.

# In[ ]:


df_loss = []
df_acc = []

# training the neural network
sess = tf.Session()
# Initializing the variables
sess.run(tf.global_variables_initializer())

for epoch in range(epochs):
    sess.run(train, feed_dict={x: train_features, y: train_labels})

    loss = sess.run(cost, feed_dict={
        x: train_features,
        y: train_labels})

    valid_acc = sess.run(accuracy, feed_dict={
        x: train_features,
        y: train_labels})

    df_loss.append(loss)
    df_acc.append(valid_acc)

    if ((epoch+1) % 1000 == 0):
        print('Epoch {:>5}, Loss: {:>10.4f} Accuracy: {:.6f}'.format(epoch+1, loss, valid_acc))
        
df_loss = pd.DataFrame(df_loss)
df_acc = pd.DataFrame(df_acc)


# Here is the loss and accuracy plots:

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16,6))

ax[0].plot(df_loss)
ax[0].set_title('Loss')

ax[1].plot(df_acc)
ax[1].set_title('Accuracy')


# ## Test model
# 
# Now, let's test the model with test data.

# In[ ]:


test_data.head()


# In[ ]:


test_set = test_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']]
test_set.head(10)


# In[ ]:


test_features = dataPreparation(test_set)

test_features.head()


# In[ ]:


test_features.info()


# In[ ]:


# predict the test set
predicted = pd.DataFrame(sess.run(prediction, feed_dict={x:test_features}))
predicted.columns = ['Survived']

submission = pd.DataFrame({'PassengerId' : test_data['PassengerId'], 'Survived' : predicted['Survived'].astype(int)})

submission.info()

submission.head(10)


# In[ ]:


submission.to_csv('results.csv', index=False)

