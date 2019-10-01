#!/usr/bin/env python
# coding: utf-8

# Neural network for Titanic Problem. However my model is not doing great. 
# But I hope it might help those who are just starting with Machine learning.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# When i started with Mahcine learning, As I was not having  data science background. I was always in hurry to apply machine learning algorithm. And guess what it has always depressed me.  I have solved the mystery now. So the my new rule says
# 
# 1.  Do not hurry. 
# 2. Look at the data provide to you  and try to find relationship between each column and the output column.
# 3. Drop column which does not have any realtion with your output column
# 
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
#first step load train data and looka the column values
traindata = pd.read_csv("../input/train.csv")
traindata.head()


# Lets look at each columns and find the realtion. Survived Column is the output column and reamaining columns are input columns
# 
# Lets look at passenger id.  It doesnot show any realation with Survived column. It is just the unique ID

# Lety look at PClass now. Look at the graph below .  It looks like people travelling in first class has more chances of survival.  we should use this column to predict survival
# 
# **Conclusion: ** Poor always dies. Money always help :)
# 
# Pclass
# 1    0.629630
# 2    0.472826
# 3    0.242363
# 

# In[ ]:


traindata.groupby("Pclass").Survived.mean().plot(kind="bar")


# I am skipping Name column for now. We will discuss it later. Lets try Sex column now.  WoW, It shows female has more survival chances.   We will keep this column as well.
# 
# Input column to be considered while making prediction:
# 1. PClass
# 2. Sex

# In[ ]:


traindata.groupby("Sex").Survived.mean().plot(kind="bar")


# Lets try Age Column. Younger Kids has more chances of survival. Age is important column to keep. But  it has lot of missing data. it has 177 rows which does not have data. So we can ignore Age column
# 
# 1.  PClass
# 2. Sex
# 

# In[ ]:


traindata[traindata["Age"].isna()].shape


# Now SibSP. Oh so passengers those who are travelling with  1/2 sIbling had higher survival rates. Lets keep this columns as well
# 
# 1.  PClass
# 2. Sex
# 3. Sibsp

# In[ ]:


print(traindata.groupby("SibSp").Survived.mean())
traindata.groupby("SibSp").Survived.mean().plot(kind="bar",figsize=(25,25))


# Lets look at the Parch column now.  # of parents / children aboard . It seems some connection . So lets consider this column as well
# 
# 1.  PClass
# 2. Sex
# 3. Sibsp
# 4. Parch
# 

# In[ ]:


print(traindata.groupby("Parch").Survived.mean())
traindata.groupby("Parch").Survived.mean().plot(kind="bar",figsize=(25,25))


# lets look at  the Ticket columns now. Does ticket number make any connection with survival column? . Nope It does not look like . So we can ignore this column. Our column reamins same
# 
# 1.  PClass
# 2. Sex
# 3. Sibsp
# 4. Parch
# 

# In[ ]:


print(traindata.groupby("Ticket").Survived.mean())
traindata.groupby("Ticket").Survived.mean().plot(kind="bar",figsize=(25,25))


# Lets look the the Fare column. Does people paid more has higher chances of survival. It seems it does. Look at the graph. people with higher prices has denser graph. which shows higher chanses of survival
# 
# 1.  PClass
# 2. Sex
# 3. Sibsp
# 4. Parch
# 5. Fare

# In[ ]:


print(traindata.groupby("Fare").Survived.mean())
traindata.groupby("Fare").Survived.mean().plot(kind="bar",figsize=(25,25))


# Embarked . Let see if it has some connection. Oh yeah it show some connection so lets keep this as well

# In[ ]:


print(traindata.groupby("Embarked").Survived.mean())
traindata.groupby("Embarked").Survived.mean().plot(kind="bar",figsize=(25,25))


# Now What. We have all input column ready . Lets make our neural network and see how it works. But hold on before that we have to  make our data in such a way so that neural network can understand. If you pass on text to it it won't work. It requries number to do matrix multiplication
# 
# 1. Encode sex : {"male":0,"female":1}
# 

# In[ ]:


traindata = pd.read_csv("../input/train.csv")
traindata["Sex"] = traindata["Sex"].map({"male":0,"female":1})
traindata["Embarked"] = traindata["Embarked"].map({"C":0,"Q":1,"S":2})


#xtrain.count()


# Data is ready . Shall we run our neural network now.. Hold on. We need to clean NA values. Look at embarked column it has only 2 na so marking them as c/0 won't hurt

# In[ ]:


print(traindata.loc[traindata["Embarked"].isna()])
traindata["Embarked"] = traindata["Embarked"].fillna(0)

xtrain  =  traindata.loc[:,["Pclass","Sex","SibSp","Parch","Fare","Embarked"]]
ytrain  =  traindata["Survived"]


# Lets run Neural network and train our data.. Wow our accuracy on train data is between 75 -79 percent. Is it good?. Nope .. It is not.. what we can do now. Shall we improve our neural network or check if we can add few more columns

# In[ ]:


import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(200,input_shape=(6,), activation="relu"))
model.add(Dense(300, activation="relu"))
model.add(Dense(40, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam",loss=keras.losses.binary_crossentropy,metrics=['mae','acc'])
model.fit(xtrain,ytrain,epochs=10)


# In[ ]:


testdata = pd.read_csv("../input/test.csv")
testdata["Sex"] = testdata["Sex"].map({"male":0,"female":1})
testdata["Embarked"] = testdata["Embarked"].map({"C":0,"Q":1,"S":2})
testdata["Embarked"] = testdata["Embarked"].fillna(0)

Y_pred = model.predict(testdata.loc[:,["Pclass","Sex","SibSp","Parch","Fare","Embarked"]])

submission = pd.DataFrame({
        "PassengerId": testdata["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# lets try Name column now. Just check Title if it give some help. Look at the graph. Yes it is helping so let add it our input coulmn list

# In[ ]:


traindata["title"] = traindata["Name"].apply(lambda x: x.split()[1].split()[0]).map({'Mr.':0, 'Mrs.':1, 'Miss.':2, 'Master.':3,"Dr.":4})
traindata["title"] = traindata["title"].fillna(-1)
traindata.groupby("title").Survived.sum().plot(kind="bar")


# In[ ]:


traindata = pd.read_csv("../input/train.csv")
traindata["Sex"] = traindata["Sex"].map({"male":0,"female":1})
traindata["Embarked"] = traindata["Embarked"].map({"C":0,"Q":1,"S":2})
traindata["title"] = traindata["Name"].apply(lambda x: x.split()[1].split()[0]).map({'Mr.':0, 'Mrs.':1, 'Miss.':2, 'Master.':3,"Dr.":4})
traindata["title"] = traindata["title"].fillna(5)
xtrain  =  traindata.loc[:,["title","Pclass","Sex","SibSp","Parch","Fare","Embarked"]]
ytrain  =  traindata["Survived"]


# In[ ]:


import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(200,input_shape=(7,), activation="relu"))
model.add(Dense(300, activation="relu"))
model.add(Dense(40, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam",loss=keras.losses.binary_crossentropy,metrics=['mae','acc'])
model.fit(xtrain,ytrain,epochs=10)


# In[ ]:


import seaborn as sns

grid = sns.FacetGrid(rr, row='Pclass', size=2.2, aspect=1.6)
grid.map(sns.pointplot,"Cabin","Survived", palette='deep')
grid.add_legend()


# In[ ]:




#model.evaluate(xtest,ytest)


# In[ ]:




