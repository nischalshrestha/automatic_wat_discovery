#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import  keras


# In[ ]:


train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
data=pd.concat([train_data,test_data])
data.head()


# In[ ]:


get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.distplot(data[data.Age.notnull()]['Age'])


# # feature enginee

# In[ ]:


import re
def parse_name(x):
    x=re.sub(r'(.*?)','',x)
    x=x.replace(',','')
    x=x.replace('.','')
    x=x.split()
    return x

data['fm_name']=data['Name'].map(lambda x:parse_name(x)[0])
data['title']=data['Name'].map(lambda x:parse_name(x)[1])

data.head()


# 
# #  fill null value
# 

# In[ ]:


data.loc[data.Age.isnull(),'Age']=data['Age'].mean()
data.loc[data.Embarked.isnull(),'Embarked']='S'
data.loc[data.Fare.isnull(),'Fare']=data['Fare'].mean()
print(data.info())


# In[ ]:


ana_feature=['Survived','Age','Fare','Parch','Pclass','SibSp']
corr_data=data[ana_feature].corr()
sns.heatmap(corr_data)


# In[ ]:


label_enc_feature=['Embarked','Pclass','Sex','title']
from sklearn.preprocessing import  LabelBinarizer

train_data=data[data.Survived.notnull()]
test_data=data[data.Survived.isnull()]


# In[ ]:


import numpy as np
for (index,i) in enumerate(label_enc_feature):
    lbe=LabelBinarizer()
    lbe.fit(data[i].values.reshape(-1,1))
    tr_x=lbe.transform(train_data[i].values.reshape(-1,1))
    te_x=lbe.transform(test_data[i].values.reshape(-1,1))
    if index==0:
        Tr_x=tr_x 
        Te_x=te_x
    else:
        print(Tr_x.shape,tr_x.shape)
        Tr_x=np.hstack((Tr_x,tr_x))
        Te_x=np.hstack((Te_x,te_x))
    
Tr_x.shape


# In[ ]:


scalar_feature=['Age','Parch','SibSp','Fare']
from sklearn.preprocessing import  StandardScaler

for i in scalar_feature:
    stc=StandardScaler()
    stc.fit(data[i].values.reshape(-1,1))
    tr_x=stc.transform(train_data[i].values.reshape(-1,1))
    te_x=stc.transform(test_data[i].values.reshape(-1,1))
    
    Tr_x=np.hstack((Tr_x,tr_x))
    Te_x=np.hstack((Te_x,te_x))
      


# In[ ]:


print(Tr_x.shape)


# In[ ]:


from keras.layers import  Dense,Input,Dropout,Activation
from keras.models import  Model
def dnn_model():
    inp=Input(shape=(45,))
    fc1=Dense(256,activation='relu')(inp)
    fc1=Dropout(0.5)(fc1)
    fc2=Dense(64,activation='relu')(fc1)
    fc2=Dropout(0.5)(fc2)
    out=Dense(1,activation='sigmoid')(fc2)
    model=Model(inp,out)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
    
model=dnn_model()


# In[ ]:


Tr_y=train_data['Survived'].values
model.fit(Tr_x,Tr_y,epochs=20,batch_size=128,validation_split=0.2)


# In[ ]:


prediction=model.predict(Te_x)
prediction=(prediction>0.7).astype(int).reshape(-1)
print(prediction)


# In[ ]:


submit=pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':prediction})
submit.head()


# In[ ]:


submit.to_csv("titanic.csv",index=False)


# In[ ]:




