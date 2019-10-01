#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

get_ipython().magic(u'matplotlib inline')


# In[ ]:


data_train = pd.read_csv("../input/train.csv")
data_val = pd.read_csv("../input/test.csv")
dataset = [data_train, data_val]

ids = data_val.PassengerId.copy()
output = ids.to_frame()


# In[ ]:


data_train.info()


# In[ ]:


print("data_train null")
print(data_train.isnull().sum())
print('/'*30)
print("data_val null")
print(data_val.isnull().sum())


# In[ ]:


from keras.utils import to_categorical

for data in dataset:
    data['Age'].fillna(data['Age'].median(), inplace = True)
    data['Fare'].fillna(data['Fare'].median(), inplace = True)
    data.Age = data.Age.astype(int)
    data.Fare = data.Fare.astype(int)
#     data['AgeBin'] = pd.cut(data['Age'], bins=[0,10,20,30,40,50,60,70,80,np.inf])
#     data['FareBin'] = pd.cut(data['Fare'], bins=[0,50,100,150,200,250,300])
#     data['FareBin'] = pd.qcut(data['Fare'], 5)
    
    data.Sex[data.Sex == 'male'] = 1
    data.Sex[data.Sex == 'female'] = 0
    data['Sex'] = data['Sex'].astype(int)


# In[ ]:


drop_column = ["PassengerId", "Cabin", "Ticket", "Name", "Embarked"]
for data in dataset:
    data.drop(drop_column, axis=1, inplace = True)


# In[ ]:


titles = ["Pclass", "Sex", "SibSp", "Parch"]
for x in titles:
    print('Survival Correlation by:', x)
    print(data_train[[x, "Survived"]].groupby(x, as_index=False).mean())


# In[ ]:


from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data_train, test_size=0.2)

train_y = data_train["Survived"]
train_X = data_train.drop("Survived", axis=1)
test_y = data_test["Survived"]
test_X = data_test.drop("Survived", axis=1)


# In[ ]:


train_X.info()


# In[ ]:


from keras import layers, models

Nin = 6
Nh_l = [100, 50, 50]
Nout = 2
epochs = 200
batch_size = 50


# In[ ]:


class DNN(models.Sequential):
    def __init__(self, Nin, Nh_l, Nout):
        super().__init__()
        self.Nh_l = Nh_l
        self.Nout = Nout
        self.make()
    
    def make(self):
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dropout(rate = 0.1))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dense(Nh_l[2], activation='relu', name='Hidden-3'))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model = DNN(Nin, Nh_l, Nout)
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
performace_test = model.evaluate(test_X, test_y, batch_size=100)
print('Test Loss and Accuracy ->', performace_test)


# In[ ]:


predictions = model.predict_classes(data_val)
output["Survived"]=predictions
output.columns = ['PassengerId', 'Survived']
output.to_csv("output.csv",index=False)


# In[ ]:




