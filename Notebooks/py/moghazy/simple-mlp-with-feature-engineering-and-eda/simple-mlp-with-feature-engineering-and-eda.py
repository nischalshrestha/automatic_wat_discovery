#!/usr/bin/env python
# coding: utf-8

# # Simple Deep Neural Networks with Keras
# In this tutorial i am going to show how to implement Deep Neural Networks in keras and Also we will have a look on simple feature engineering to be able to classify the dataset efficiently

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# First let's start by importing our dataset into dataframes using pandas. Dataframes enables us to work easily with data usign its built in functions.

# In[ ]:


import pandas as pd
training = pd.read_csv("../input/train.csv");

x_test = pd.read_csv("../input/test.csv");


# Now let's have a look on the dataset and search for important information manually.

# In[ ]:


training.head()


# Now I will explore the correlations of the featues relative to the target variable.
# #### Note: not all features are listed but only numeric features.

# In[ ]:


import seaborn as sns


import matplotlib.pyplot as plt


corr = training.corr()
f, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)


# I will explore the size of the dataset to compare it with the number of NANs in the dataset

# In[ ]:


print("The number of traning examples(data points) = %i " % training.shape[0])
print("The number of features we have = %i " % training.shape[1])


# I will count the number of data examples in each class in the target to determine which metric to use while evaluationg performance.

# In[ ]:


unique, count= np.unique(training["Survived"], return_counts=True)
print("The number of occurances of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )


# The numbers doesn't seem to be very far from each other, So i will use accuracy for performance eval..

# Now i will check the number of Null values. If most of a column's values are Nulls or NaNs i will drop it because filling it will not be accurate but if the number is small then i will fill it with the mean values.

# In[ ]:


training.isna().sum()


# From the results of correlation matrix and the Nan number and manual checking of dataset, I will drop "Name" ,"Ticket","Cabin" and "PassengerId".
# <lb>I will also store the label column and drop it from training.
# <lb>Then i will engineer the catagorical features and map the strings to integers to be able to use it in the model later.

# In[ ]:


np.random.seed(0)
training.drop([ "Name" , "PassengerId","Ticket","Cabin"], inplace = True, axis = 1 )
training.dropna( inplace = True)
x_train = training
repCol3 = {  "male":0, "female" : 1}
repCol8 = {"C" : 0 ,   "Q" : 1 , 'S' : 2  }

x_train['Family'] = x_train ['SibSp'] + x_train['Parch']
x_train['IsAlone'] = 1
x_train['IsAlone'].loc[x_train['Family'] > 0] = 0
    
# mean = x_train.mean().astype(np.int32)
# print (mean)
# x_train.fillna( mean , inplace = True)
x_train.replace({"Sex": repCol3, "Embarked": repCol8} , inplace = True )
# x_train = x_train / x_train.max() # Normalizing x_train data
print( x_train.shape )
x_train.head(100)


# In[ ]:


from scipy import stats
import numpy as np

z = np.abs(stats.zscore(x_train))
zee = (np.where(z > 3))[1]

print("number of data examples greater than 3 standard deviations = %i " % len(zee))


# In[ ]:


# x_train = x_train[(z < 2.5).all(axis=1)]


# In[ ]:


x_train.head()


# In[ ]:


y_train = x_train["Survived"]
x_train = x_train.drop(['Survived'], axis = 1)

print(y_train.shape )
y_train.head(20)


# I will Plot some features to see if there is any pattern in the data.

# In[ ]:


import matplotlib.pyplot as plt
classes = np.array(list(y_train.values))

def plotRelation(first_feature, sec_feature):
    
    plt.scatter(first_feature, sec_feature, c = classes, s=10)
    plt.xlabel(first_feature.name)
    plt.ylabel(sec_feature.name)
    
f = plt.figure(figsize=(25,20))
f.add_subplot(331)
plotRelation(x_train.Pclass, x_train.Embarked)
f.add_subplot(332)
plotRelation(x_train.Pclass, x_train.Age)
f.add_subplot(333)
plotRelation(x_train.Age, x_train.Sex)
f.add_subplot(334)
plotRelation(x_train.SibSp, x_train.Parch)
f.add_subplot(335)
plotRelation(x_train.Fare, x_train.Embarked)
f.add_subplot(336)
plotRelation(x_train.Fare, x_train.Embarked)


# Since i dropped some features from the training set I have to drop the same featurres from the test set and do the same steps of feature eng.

# In[ ]:


x_test.drop(["Name","Ticket","Cabin" ], inplace = True, axis = 1 )
x_test.replace({"Sex": repCol3, "Embarked": repCol8} , inplace = True )
x_test.head()


# Now it is time to design the ML Pipeline. I will use Deep Neural Networks in Keras to classify the dataset. The number of layers i am using is optmized using some error analys of the results.
# <lb> I will use early stopping to stop if the error is not decreasing.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import callbacks
from keras import optimizers
#y_train = np_utils.to_categorical(y_train)

InputDimension = 9
print(y_train.shape )

model = Sequential()
model.add(Dense(10, input_dim=InputDimension, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(2, activation='softmax'))


earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min')
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
history = model.fit(x_train, pd.get_dummies(y_train), epochs=1000, batch_size=200, validation_split=0.2, verbose=0, callbacks=[earlystopping])


# Now let's see how good is my training with respect to validation accuracies.

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])


# Now i will normalize the test set as i did before and will fill null values in the dataset as well.

# In[ ]:


id = x_test['PassengerId']

x_test.drop(['PassengerId'], inplace = True, axis = 1)

x_test['Family'] = x_test ['SibSp'] + x_test['Parch']
x_test['IsAlone'] = 1
x_test['IsAlone'].loc[x_test['Family'] > 0] = 0


x_test.fillna( x_test.median() ,inplace = True)
# x_test = x_test/ x_test.max()


# Now every thing is okay with the dataset so i will predict the output values for submission

# In[ ]:



predictions = model.predict(x_test)


# Predictions will return probability between 0 and 1 for survived or non survived so i will take the argmax() of the array to get the max index for each test example

# In[ ]:


predictions = np.rint(predictions)


# I will convert the array into dataframe with one col target value instead of two since my model returns 2 cols because of the 2 classes i have in the last layer.

# In[ ]:


predict = pd.DataFrame(predictions, columns=['0', '1']).astype('int')
predict['Survived'] = 0
predict.loc[predict['0'] == 1, 'Survived'] = 0
predict.loc[predict['1'] == 1, 'Survived'] = 1


# It is time to make submission.

# In[ ]:


id.reset_index(drop=True, inplace=True)
output = pd.concat([id,predict['Survived'] ], axis=1)
output.to_csv('titanic-predictions.csv', index = False)
output.head(100)

