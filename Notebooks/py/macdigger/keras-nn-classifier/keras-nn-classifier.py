#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("../input/train.csv")
del df['Name']


# In[ ]:


sb = lambda x: 0 if x == 'male' else 1
df["Sex"] = df["Sex"].apply(sb)


# In[ ]:


del df['Ticket']
del df['Cabin']


# In[ ]:


def emb(a):
    cases = {'Q': 1, 'S': 2, 'C': 3}
    if a in cases:
        return cases[a]
    else:
        return 0
df['Embarked'] = df['Embarked'].apply(emb)


# In[ ]:


del df['PassengerId']


# In[ ]:


df = df.fillna(df.mean())


# In[ ]:


dfn = (df - df.min()) / (df.max() - df.min())


# In[ ]:


trainY = dfn['Survived'].values
trainX = dfn
del trainX['Survived']
trainX = trainX.values


# In[ ]:


from keras.utils import np_utils
trainY = np_utils.to_categorical(trainY, 2)


# In[ ]:


from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential

model = Sequential()
model.add(Dense(20, input_shape=[7,]))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile('adam','mean_squared_error', metrics=['accuracy'])


# In[ ]:


tf = pd.read_csv("../input/test.csv")
del tf['Name']
tf["Sex"] = tf["Sex"].apply(sb)
del tf['Ticket']
del tf['Cabin']
tf['Embarked'] = tf['Embarked'].apply(emb)
del tf['PassengerId']
tf = tf.fillna(tf.mean())
tfn = (tf - tf.min()) / (tf.max() - tf.min())


# In[ ]:


testX = tfn.values


# In[ ]:


model.fit(trainX, trainY, epochs=150, verbose=1, batch_size=100)


# In[ ]:


import numpy as np


# In[ ]:


preds = [int(i) for i in model.predict_classes(testX)]


# In[ ]:


preds = [[a, b] for a, b in zip(range(892, 1310), preds)]
testY = pd.DataFrame(preds, columns=['PassengerID','Survived'])


# In[ ]:


testY.to_csv('pred.csv', index=False)

