#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network using Keras
# Titanic <br>
# Damien Park

# ---

# In[1]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense
import keras
from keras.optimizers import *
from keras.initializers import *

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from itertools import chain


# In[2]:


df = pd.read_csv("../input/train.csv")


# ## Data Dictionary

# |Variable|Definition|Key|
# |----------|-----------|:---:|
# |Survival|Survival|0 = No, 1 = Yes|
# |Pclass|Ticket class|1 = 1st, 2 = 2nd, 3 = 3rd|
# |Sex|Sex|-|
# |Age|Age in years|-|
# |Sibsp|# of siblings / spouses aboard the Titanic|-|
# |Parch|# of parents / children aboard the Titanic|-|
# |Ticket|Ticket number|-|
# |Fare|Passenger fare|-|
# |Cabin|Cabin number|-|
# |Embarked|Port of Embarkation|C = Cherbourg, Q = Queenstown, S = Southampton|

# ---

# ## 1. Data Preprocessing

# ## Take Log of Fare

# In[3]:


df.query("Fare == 0").head()


# In[5]:


plt.figure(figsize=(20,10))
plt.subplot(2, 4, 1)
plt.title("Fare")
plt.hist(df.Fare, bins=45)

plt.subplot(2, 4, 2)
plt.title("+1, log2, log10")
plt.hist(np.log2(np.log10(df.Fare + 1) + 1), bins=45)

plt.subplot(2, 4, 3)
plt.title("+2, log2, log2")
plt.hist(np.log2(np.log2(df.Fare + 2) + 2), bins=45)

plt.subplot(2, 4, 4)
plt.title(">0, log10, round")
plt.hist(np.round(np.log10(df.query("Fare > 0").Fare), 2), bins=45)

plt.subplot(2, 4, 5)
plt.title("log2")
plt.hist(np.log2(df.Fare + 0.1), bins=45)

plt.subplot(2, 4, 6)
plt.title("log10")
plt.hist(np.log10(df.Fare + 0.1), bins=45)

plt.subplot(2, 4, 7)
plt.title(">0, log2, round")
plt.hist(np.round(np.log2(df.query("Fare > 0").Fare), 2), bins=45)

plt.subplot(2, 4, 8)
plt.title("normalization, +1, log10")
plt.hist(np.log10((df.Fare + 1 - df.Fare.min())/df.Fare.max()) +1.5, bins=45)
plt.show()


# In[6]:


np.mean(df.Fare), np.std(df.Fare)


# In[21]:


# Fare normalization
for i in range(len(df.Fare)):
    df.loc[i, "nor_Fare"] = np.log10(np.abs((df.Fare[i]+0.1-np.mean(df.Fare)))/np.std(df.Fare))


# In[23]:


plt.hist(df.nor_Fare, bins=45)
plt.show()


# In[ ]:


# df.Fare = np.log10((df.Fare + 1 - df.Fare.min())/df.Fare.max()) +1.5


# In[ ]:


# for idx, value in enumerate(df.Fare):
#     if value != 0:
#         df.loc[idx, "Fare"] = np.round(np.log10(value), 2)


# In[24]:


df.Fare = df.nor_Fare


# In[25]:


df.head()


# ## Encoding
# Sex(One-Hot-Encoding) <br>
# Pclass(Label Encoding) <br>
# Embarked(Label Encoding)

# In[26]:


# Sex Encoding
LabelBinarizer().fit(df.loc[:, "Sex"])
df["Sex"] = LabelBinarizer().fit_transform(df["Sex"])
df.head()


# In[27]:


# Pclass Encoding
df_Pclass = pd.DataFrame(OneHotEncoder().fit_transform(np.array(df["Pclass"])[:,np.newaxis]).toarray(), columns=["A_Class", "B_Class", "C_Class"])
df_Pclass = df_Pclass.astype(int)
df_Pclass.head()


# In[28]:


# Embarked Encoding
df_Embarked = pd.get_dummies(df.Embarked)
df_Embarked.head()


# ## Interpolation for Age

# In[29]:


plt.hist(df.query("Age>0").Age, bins=40)
plt.show()


# In[30]:


df.Age.isna().sum()


# In[31]:


# Nan Age is fill using average age
#df.loc[:, "Age"].fillna(int(df["Age"].mean()), inplace=True)
#df.loc[:, "Age"].fillna(int(df["Age"].median()), inplace=True)


# In[32]:


#df.query("Sex == 'male' & Age == 'Nan'").fillna(30, inplace=True)
#df.query("Sex == 'female' & Age == 'Nan'").fillna(28, inplace=True)


# In[33]:


df.groupby("Sex").Age.mean()


# In[34]:


df.loc[:, "Age"].fillna(0, inplace=True)

for idx, value in enumerate(df.Age):
    if value == 0:
        if df.Sex[idx] == 'male':
            df.Age[idx] = 30
        else:
            df.Age[idx] = 28


# In[35]:


plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.title("Age")
plt.hist(df.Age, bins=40)
plt.subplot(1, 4, 2)
plt.title("Age, log10")
plt.hist(np.log10(df.Age), bins=40)
plt.subplot(1, 4, 3)
plt.title("Age, log2")
plt.hist(np.log2(df.Age), bins=40)
plt.subplot(1, 4, 4)
plt.title("Age, log e")
plt.hist(np.log(df.Age), bins=40)
plt.show()


# In[36]:


df.Age = np.log10(df.Age)


# ## New columes
# Boarding Together or Alone

# In[37]:


# Boarding Together or Alone
for i in range(len(df)):
    if df.loc[i, "SibSp"] + df.loc[i, "Parch"] == 0:
        df.loc[i, "Alone"] = 0
    else:
        df.loc[i, "Alone"] = 1

df.Alone = df.Alone.astype(int)
df.head()


# ## Data Marge

# In[38]:


df_new = pd.concat([df, df_Pclass, df_Embarked], axis=1)
df_new.head()


# In[39]:


#feature_name = ["Sex", "Age", "Fare", "Alone", "A_Class", "B_Class", "C_Class", "C", "Q", "S"]
feature_name = ["Sex", "Age", "nor_Fare", "Alone", "A_Class", "B_Class", "C_Class", "C", "Q", "S"]

dfX = df_new[feature_name]
dfY = df_new["Survived"]


# ## Split data(train, test)

# In[40]:


X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, test_size=0.25, random_state=1)


# In[41]:


#X_train, X_test, y_train, y_test


# ---

# ## 2. Set Artifial neural network and Learning
# multi layers

# In[42]:


df.Survived.sum() / len(df)


# In[43]:


model = Sequential()


# In[44]:


model.add(Dense(64, input_dim=10, activation="relu", kernel_initializer="he_normal", bias_initializer=Constant(0.01)))
model.add(Dense(128, activation="relu", kernel_initializer="he_normal"))
model.add(Dense(256, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dropout(0.01))
model.add(Dense(256, activation="relu", kernel_initializer="he_normal"))
model.add(Dense(128, activation="relu", kernel_initializer="he_normal"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="glorot_normal"))
model.compile(optimizer=rmsprop(), loss='binary_crossentropy', metrics=["binary_accuracy"])


# In[45]:


model.layers


# In[46]:


model_result = model.fit(X_train.values, y_train.values, epochs=550, batch_size=20, validation_data=(X_test.values, y_test.values), verbose=2)


# ## Result

# In[47]:


plt.figure(figsize=(15, 10))
plt.plot(model_result.history["loss"])
plt.axhline(0.55, c="red", linestyle="--")
plt.axhline(0.35, c="yellow", linestyle="--")
plt.axhline(0.15, c="green", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[48]:


plt.figure(figsize=(15, 10))
plt.plot(model_result.history["binary_accuracy"], label="training")
plt.plot(model_result.history["val_binary_accuracy"], label="test")
plt.axhline(0.75, c="red", linestyle="--")
plt.axhline(0.80, c="green", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[49]:


y_predict = model.predict_classes(X_test.values)


# In[50]:


print(classification_report(y_test, y_predict))


# In[51]:


model.summary()


# ---

# ## 3. Predict_OUTPUT

# In[52]:


test = pd.read_csv("../input/test.csv")


# In[53]:


# Log Fare
# for idx, value in enumerate(test.Fare):
#     if value != 0:
#         test.loc[idx, "Fare"] =  np.round(np.log10(value), 2)
# test.Fare = np.log10((test.Fare + 1 - test.Fare.min())/test.Fare.max()) +1.5
# for i in range(len(test.Fare)):
#     test.loc[i, "nor_Fare"] = (test.Fare[i]-np.mean(test.Fare))/np.std(test.Fare)


# In[54]:


# fare
for i in range(len(test.Fare)):
    test.loc[i, "nor_Fare"] = np.log10(np.abs((test.Fare[i]+0.1-np.mean(test.Fare)))/np.std(test.Fare))
test.Fare = test.nor_Fare


# In[55]:


# Sex Encoding
LabelBinarizer().fit(test.loc[:, "Sex"])
test["Sex"] = LabelBinarizer().fit_transform(test["Sex"])


# In[56]:


# Pclass Encoding
test_Pclass = pd.DataFrame(OneHotEncoder().fit_transform(np.array(test["Pclass"])[:,np.newaxis]).toarray(), columns=["A_Class", "B_Class", "C_Class"])
test_Pclass = test_Pclass.astype(int)


# In[57]:


# Embarked Encoding
test_Embarked = pd.get_dummies(test.Embarked)


# In[58]:


# Nan Age is filled using average age
#test.loc[:, "Age"].fillna(int(test["Age"].mean()), inplace=True)
#test.loc[:, "Age"].fillna(int(test["Age"].median()), inplace=True)


# In[59]:


# Nan Age filled by sex
test.loc[:, "Age"].fillna(0, inplace=True)

for idx, value in enumerate(test.Age):
    if value == 0:
        if test.Sex[idx] == 'male':
            test.Age[idx] = 30
        else:
            test.Age[idx] = 28


# In[60]:


test.Age = np.log10(test.Age)


# In[61]:


# Boarding Together or Alone
for i in range(len(test)):
    if test.loc[i, "SibSp"] + test.loc[i, "Parch"] == 0:
        test.loc[i, "Alone"] = 0
    else:
        test.loc[i, "Alone"] = 1

test.Alone = test.Alone.astype(int)


# In[62]:


test_new = pd.concat([test, test_Pclass, test_Embarked], axis=1)
test_new.head()


# In[63]:


feature_name = ["Sex", "Age", "Fare", "Alone", "A_Class", "B_Class", "C_Class", "C", "Q", "S"]
testX = test_new[feature_name]


# In[64]:


predict = model.predict_classes(testX)


# In[65]:


predict = list(chain.from_iterable(predict))


# In[66]:


my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predict})
my_submission.to_csv('submission.csv', index=False)


# -End of Analysis <br>
# R.I.P

# In[ ]:




