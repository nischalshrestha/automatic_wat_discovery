#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/train.csv")
#df.head()
df.info()


# In[ ]:


#df.dropna(axis=0,inplace=True)
vals = pd.DataFrame(data=df,columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Survived'])
vals.dropna(axis=0,inplace=True)
x = pd.DataFrame(data=vals,columns=['Pclass','Sex','Age','SibSp','Parch','Fare'])

y = pd.DataFrame(data=vals,columns=['Survived'])
def gender_to_no(x):
    if x=='male':
        return 0
    elif x=='female':
        return 1
    else:
        return -1
    
#print(type(x['Sex'].iloc[0]))    
x['Sex']=x['Sex'].apply(gender_to_no)
#x.head()
x.info()
#x.drop(columns=[1],axis=1)
#x2 = pd.DataFrame(data=x,columns=['Pclass','Age','SibSp','Parch','Fare'])
#2.join(k)x2.head()
#print(gender_to_no("female"))

    
    


# In[ ]:


y.info()


# In[ ]:


train_x = x.values
train_y = np.squeeze(y.values)
print(train_x.shape)
print(train_y.shape)


# In[ ]:


split = int(0.8*train_x.shape[0])
x_tr = train_x[:split,:]
x_tst = train_x[split:,:]
y_tr = train_y[:split]
y_tst = train_y[split:]

rf = RandomForestClassifier(n_estimators=500)
rf.fit(x_tr,y_tr)
    
    
    


# In[ ]:


rf.score(x_tst,y_tst)


# In[ ]:


svm = SVC()
svm.fit(x_tr,y_tr)


# In[ ]:


svm.score(x_tst,y_tst)


# In[ ]:


ad = AdaBoostClassifier(n_estimators=100)
ad.fit(x_tr,y_tr)


# In[ ]:


ad.score(x_tst,y_tst)


# In[ ]:


gb = GaussianNB()
gb.fit(x_tr,y_tr)


# In[ ]:


gb.score(x_tst,y_tst)


# In[ ]:


etc = ExtraTreesClassifier(n_estimators=100)
etc.fit(x_tr,y_tr)


# In[ ]:


etc.score(x_tst,y_tst)


# In[ ]:


gbc = GradientBoostingClassifier(n_estimators=200)
gbc.fit(x_tr,y_tr)


# In[ ]:


gbc.score(x_tst,y_tst)


# In[ ]:


class Bin_Classifier(nn.Module):
    def __init__(self):
        super(Bin_Classifier,self).__init__()
        self.input = nn.Linear(6,64)
        self.output = nn.Linear(64,2)
        
    def forward(self,x):
        x = self.input(x)
        x = self.output(x)
        return(nn.Sigmoid(x))


# In[ ]:


BC = Bin_Classifier()
print(BC)


# In[ ]:


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(BC.parameters(),lr = 0.002)


# In[ ]:


print(x1.size())


# In[ ]:


x1 = Variable(torch.FloatTensor(x_tr),requires_grad=True)
y1 = Variable(torch.LongTensor(y_tr))
    


# In[ ]:


"""for epoch in range(101):
    out1 = BC(x1)
    loss = loss_func(out1,y1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%10==0:
        inp =Variable(torch.FloatTensor(x_tst)) 
        preds = BC(inp)
        out = np.argmax(preds.data.numpy(),axis=1)
        acc = (out==y_tst)*100.0/y_tst.shape[0]
        print("Epoch:{}||Acc:{}".format(epoch,acc))"""


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.describe()


# In[ ]:


pid = pd.DataFrame(data=test,columns=['PassengerId'])
test_x = pd.DataFrame(data=test,columns=['Pclass','Sex','Age','SibSp','Parch','Fare'])
test_x['Sex']=test_x['Sex'].apply(gender_to_no)
#test_x.dropna(axis=0,inplace=True)
def rem_age_nan(x):
    if np.isnan(x)==True:
        return 30
    else:
        return x
def rem_fare_nan(x): 
    if np.isnan(x)==True:
        return 35.6
    else:
        return x
test_x['Age']=test_x['Age'].apply(rem_age_nan)
test_x['Fare']=test_x['Fare'].apply(rem_fare_nan)
test_x.info()


# In[ ]:


rf2 = RandomForestClassifier(n_estimators=500)
rf2.fit(train_x,train_y)


# In[ ]:


arr = rf2.predict(test_x)
Survived =pd.DataFrame(data=arr,columns=['Survived'])
Survived.head()


# In[ ]:


pid = pid.join(Survived)
pid.head()


# In[ ]:


pid.info()


# In[ ]:


pid.to_csv('submission.csv',index=False)

