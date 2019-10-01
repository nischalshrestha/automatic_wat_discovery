#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import models 
from keras import layers

from keras import regularizers

from sklearn.metrics import accuracy_score

import os
print(os.listdir("../input/"))


# I am going to imput real historical data as Validation set, so we could get real accuracy without submit.
# That maybe will give us some new insight, at least that save a lot of time.

# In[ ]:


#import train and test CSV files
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
#import real historical data.
history=pd.read_csv("../input/test-dataset/TitanicHistory.csv")


# In[ ]:


#this is real data on history
history['Survived']=history['Survived'].map({"No":0,"Yes":1})
history[['Name','Survived']].head()


# We don't split validation set from train.
# We use history data as validation set.
# maybe this try will give us something different.

# In[ ]:


#combine historical data with test
testWithHistory=pd.merge(test,history[['Name','Survived']],on='Name', how = 'left')
testWithHistory.head()


# In[ ]:


train['Mark']='train'
testWithHistory['Mark']='test'
#combine train and test for convenience, making them easy for feature engineer
dataInput=pd.concat([train,testWithHistory])
dataPrepare=dataInput.copy(deep=True)


# In[ ]:


#let's see the null situation.
print(pd.isnull(test).sum())
print(pd.isnull(train).sum())
print(pd.isnull(dataPrepare).sum())


# Age feature will fill null with 0 because deep net could recognize 0 if we leave enough space.
# Enough space means we make move everyting close to 0 equal 1 , 
# then let deep net see the difference:
# 0 as null 
# 1 still means age
# 

# In[ ]:


dataPrepare['Age'][dataPrepare['Age']<1]=1

dataPrepare['Age']=dataPrepare['Age'].fillna(0)
dataPrepare['Age']=dataPrepare['Age'].astype('float32')

dataPrepare['Age']=dataPrepare['Age']
dataPrepare.head()


# Let's change some char to number.

# In[ ]:


dataPrepare['Sex']=dataPrepare['Sex'].map({"male":1,"female":2})
dataPrepare['Sex'].value_counts()


# In[ ]:


dataPrepare['Embarked']=dataPrepare['Embarked'].map({"S":1,"C":2,"Q":3})
dataPrepare['Embarked']=dataPrepare['Embarked'].fillna(0)


# In[ ]:


#exact title from name
dataPrepare['Title'] = dataPrepare.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
dataPrepare['Title']=dataPrepare['Title'].map({"Mr":1,"Miss":2,"Mrs":3,"Master":4})
dataPrepare['Title']=dataPrepare['Title'].fillna(0)
dataPrepare['Title']=dataPrepare['Title'].astype('int8')
dataPrepare['Title'].value_counts()


# Now we move on model part.
# some nominal feature should be convert use dummy() or something like that,  but may be deep net could understand if I give it more layers.
# so Let's try

# In[ ]:


dataPrepare.head()


# In[ ]:


selectColumns=['PassengerId', 'Embarked', 'Title', 'Age', 'Fare', 'Mark','Parch',
        'Pclass', 'Sex', 'SibSp', 'Survived']
dataPrepare[selectColumns].head()
dataPrepare['Fare']=dataPrepare['Fare'].fillna(0)
print(pd.isnull(dataPrepare[selectColumns]).sum())


# In[ ]:


dataPrepare[selectColumns].head()


# In[ ]:


#set train and validation set.
#as previous mentioned, we use historical data as Validation set.
#let's what will give us.
train_input=dataPrepare[dataPrepare['Mark']=='train']
validation_input=dataPrepare[dataPrepare['Mark']=='test']
validation_input=validation_input[validation_input['Survived'].notnull()]
featureColumns=[ 'Age','Title','Embarked', 'Fare','Parch','Pclass', 'Sex', 'SibSp']

x=train_input[featureColumns].values
y=train_input['Survived'].values

x_val=validation_input[featureColumns].values
y_val=validation_input['Survived'].values


# In[ ]:


#keras deep net

model = models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(len(featureColumns),))) 
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10, activation='relu')) 
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10, activation='relu')) 
model.add(layers.Dropout(0.1))

model.add(layers.Dense(10, activation='relu')) 
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10, activation='relu')) 
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation='sigmoid',))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit(x, y, epochs=100,
batch_size=32, validation_data=(x_val, y_val),verbose=0)


# In[ ]:


history_dict = history.history
import matplotlib.pyplot as plt
history_dict = history.history 
loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss') 
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 
plt.title('Training and validation loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend()
plt.show()


# In[ ]:


plt.clf()
acc = history_dict['acc'] 
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc') 
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title('Training and validation accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend()
plt.show()


# In[ ]:


y_pred = model.predict(x_val)
acc_randomforest = round(accuracy_score(np.where(y_pred<0.5,0,1), y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


# Random Forest as reference
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x, y)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


# Gradient Boosting Classifier as reference
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x, y)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[ ]:


#submit upload and you will see that equal the accuracy of this notebook. 
ids = test['PassengerId']
predits=dataPrepare[dataPrepare['Mark']=='test']
predits=predits[featureColumns].values
predictions = model.predict(predits)
predictions=np.where(predictions<0.5,0,1)
predictions=pd.Series(predictions.reshape(418))

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output['Survived']=output['Survived'].astype('int8')
output.to_csv('submission.csv', index=False)

