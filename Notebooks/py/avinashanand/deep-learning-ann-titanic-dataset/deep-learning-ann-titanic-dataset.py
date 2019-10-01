#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from time import strftime,gmtime


# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import Adam,RMSprop
from keras.activations import elu,relu,sigmoid,tanh
from keras.utils import np_utils
from keras.regularizers import l1,l2


# In[ ]:


from sklearn.model_selection import StratifiedKFold


# In[ ]:


np.random.seed(42)


# In[ ]:


epochs = 500
batch_size = 700
verbose = 1
classes = 2
optimiser = Adam()
validation_split = 0.3


# In[ ]:


model = Sequential()
model.add(Dense(50))
model.add(Activation(relu))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Activation(relu))
model.add(Dropout(0.1))
model.add(Dense(250))
model.add(Activation(relu))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation(relu))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Activation(relu))
model.add(Dropout(0.1))
model.add(Dense(classes))
model.add(Activation(sigmoid))


# In[ ]:


df_train = pd.read_csv('../input/dataset/feature_engg_train.csv')
df_gs = pd.read_csv('../input/titanic/gender_submission.csv')
df_test = pd.read_csv('../input/dataset/feature_engg_test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train.columns


# In[ ]:


y = df_train['Survived']
df_train.drop('Survived',axis=1,inplace=True)
df_train = df_train[['Age', 'Parch', 'Fare', 'Title', 'Name_length', 'FamilySize',
       'Cabin_null', 'Cabin_B', 'Embarked_C', 'Embarked_S', 'Male', 'Female',
       'Pclass_1', 'Pclass_2', 'Pclass_3']]
df_test = df_test[['Age', 'Parch', 'Fare', 'Title', 'Name_length', 'FamilySize',
       'Cabin_null', 'Cabin_B', 'Embarked_C', 'Embarked_S', 'Male', 'Female',
       'Pclass_1', 'Pclass_2', 'Pclass_3']]


# In[ ]:


model.compile(optimizer=optimiser,loss='binary_crossentropy',metrics=['binary_accuracy'])


# In[ ]:


X_train = df_train.as_matrix()
y_train = np_utils.to_categorical(y,classes)
# kfold = StratifiedKFold(random_state=42)
# for i,(train_idx,valid_idx) in enumerate(kfold.split(df_train,y)):
#     X_train, y_train = df_train.iloc[train_idx],y.iloc[train_idx]
#     X_valid, y_valid = df_train.iloc[valid_idx],y.iloc[valid_idx]
    
# X_train = X_train.as_matrix()
# X_valid = X_valid.as_matrix()
# y_train = np_utils.to_categorical(y[train_idx],classes)
# y_valid = np_utils.to_categorical(y[valid_idx],classes)
history = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=verbose,validation_split=validation_split)
# score = model.evaluate(X_valid,y_valid,verbose=verbose)
# print(score)
#     print('Test Score '.format(score[0]))
#     print('Test Accuracy '.format(score[1]))


# In[ ]:


y_pred = model.predict_classes(df_test)


# In[ ]:


model.predict_classes(df_test)


# In[ ]:


df_gs['Survived'] = y_pred


# In[ ]:


curr_time = strftime('%Y-%m-%d-%H-%M-%S')
df_gs.to_csv(f'output{curr_time}.csv',index=False)


# In[ ]:


df_gs['Survived'].value_counts()

