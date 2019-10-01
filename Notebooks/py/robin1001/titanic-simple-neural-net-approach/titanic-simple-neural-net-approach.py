#!/usr/bin/env python
# coding: utf-8

# ## Data processing

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set_style("whitegrid")


# In[ ]:


train_data_path = "../input/train.csv"
train = pd.read_csv(train_data_path)


# In[ ]:


train = train.fillna(np.nan)
train.isnull().sum()


# While there are missing values 
# I decide to fill ages with this instruction follow(:https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)

# In[ ]:


index_NaN_age = list(train["Age"][train["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = train["Age"].median()
    age_pred = train["Age"][((train['SibSp'] == train.iloc[i]["SibSp"]) & (train['Parch'] == train.iloc[i]["Parch"]) & (train['Pclass'] == train.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        train['Age'].iloc[i] = age_pred
    else :
        train['Age'].iloc[i] = age_med


# In[ ]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]
train["Title"] = pd.Series(dataset_title)
train["Title"].head()


# In[ ]:


train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
train["Title"] = train["Title"].astype(int)


# In[ ]:


train["Sex"] = train["Sex"].map({"male":0,"female":1})
train["Sex"] = train["Sex"].astype(int)
train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


train["Survived"] = train["Survived"].astype(int)
Y = train['Survived']
train.drop(labels = ["Survived"], axis = 1, inplace = True)
train.drop(labels = ["Cabin"], axis = 1, inplace = True)
train.drop(labels = ["Name"], axis = 1, inplace = True)
train.drop(labels = ["Ticket"], axis = 1, inplace = True)
train.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# In[ ]:


Y = pd.get_dummies(Y, columns = ["Survived"], prefix="sv")


# In[ ]:


Y.head()


# In[ ]:


train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")


# In[ ]:


train.head()


# In[ ]:


Y = np.asarray(Y.values,np.float32)
X = np.asarray(train.values,np.float32)
params = 10


# ## Neural Network Modeling

# In[ ]:


from keras.models import Sequential
from keras import layers
from keras.layers import Dense,Dropout,BatchNormalization,Activation
from keras.models import model_from_json
from keras import optimizers

model = Sequential()
# Input - Layer
model = Sequential()
model.add(Dense(24, input_shape=(params,)))
model.add(Dropout(0.4))
for i in range(0, 15):
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.40))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=500,batch_size=50)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# ## Make prediction

# In[ ]:


test_data_path = "../input/test.csv"
test = pd.read_csv(test_data_path)
test = test.fillna(np.nan)
test.isnull().sum()


# In[ ]:


index_NaN_age = list(test["Age"][test["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = train["Age"].median()
    age_pred = train["Age"][((train['SibSp'] == train.iloc[i]["SibSp"]) & (train['Parch'] == train.iloc[i]["Parch"]) & (train['Pclass'] == train.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        test['Age'].iloc[i] = age_pred
    else :
        test['Age'].iloc[i] = age_med


# In[ ]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]
test["Title"] = pd.Series(dataset_title)
test["Title"].head()
test["Title"] = test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test["Title"] = test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
test["Title"] = test["Title"].astype(int)
test["Sex"] = test["Sex"].map({"male":0,"female":1})
test["Sex"] = test["Sex"].astype(int)
test["Fare"] = test["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
test.drop(labels = ["Cabin"], axis = 1, inplace = True)
test.drop(labels = ["Name"], axis = 1, inplace = True)
test.drop(labels = ["Ticket"], axis = 1, inplace = True)
test.drop(labels = ["PassengerId"], axis = 1, inplace = True)
test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")
test.head()


# In[ ]:


X_test = np.asarray(test.values,np.float32)


# In[ ]:


idx = 0
data_size = len(X_test)
survived = 0
print("data size: ",len(X_test))
#print("Pclass Sex age sibs fare Embarked")
Y_test = []
for i in X_test:
    y_ = model.predict(i.reshape(1,10))
    if(1 == np.rint(y_[0][0])):
        survived += 1
    #print(idx,i, np.rint(y_))
    idx +=1
    Y_test.append(int(np.rint(y_[0][0])))
print("Model predict",survived,"survived (",survived/data_size,"%)")


# In[ ]:


#csv write
import csv    
f = open('result.csv', 'w', encoding='utf-8', newline='')
w = csv.writer(f)
w.writerow(['PassengerId','Survived'])
id_start = 892
idx = 0
for i in Y_test:
    w.writerow([id_start + idx,i])
    idx +=1
f.close()


# In[ ]:




