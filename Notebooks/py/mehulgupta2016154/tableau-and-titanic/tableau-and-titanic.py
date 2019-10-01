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


# Reading training and testing Dataset as train and test as pandas Dataframe........ 

# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
submission=pd.read_csv("../input/gender_submission.csv")


# Going with basic analysis of the given data to know
# ---->different datatypes
# ---->total number of entries
# ---->null values
# ---->different features available
# ---->various statistical information like std,mean,count etc
#          This all is done using info() and describe()

# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# checking sample data from train and test....

# In[ ]:


train.head()


# In[ ]:


test.head()


# FEATURE ENGINEERING
# Extracting iMportant feature from Name column (Honorific info) using regex

# In[ ]:


import re
def extractor(name):
    extract=re.findall('(Mrs|Mr|Master|Miss)',name)
    for x in extract:
        return x
sexTrain=train['Name'].transform(extractor)
sexTest=test['Name'].transform(extractor)


# In[ ]:


train['Honorific']=sexTrain
test['Honorific']=sexTest


# Filling NA values of newly created feature Honorific and Age

# In[ ]:


train['Honorific']=train['Honorific'].fillna('unknown')
test['Honorific']=test['Honorific'].fillna('unknown')
from scipy.stats import mode
train['Age']=train['Age'].fillna(train['Age'].mode()[0])
test['Age']=test['Age'].fillna(test['Age'].mode()[0])


# As missing data in Embarked is very low , we should drop the rows having missing value

# In[ ]:


train=train.dropna(subset=['Embarked'])
test=test.dropna(subset=['Embarked'])


# Though more than 50% of data is missing for Cabin,it can be very useful as cabin location should have affected survival rate, so we aren't dropping it

# In[ ]:


train['Cabin']=train['Cabin'].fillna('unknown')
test['Cabin']=test['Cabin'].fillna('unknown')
test['Fare']=test['Fare'].fillna(test['Fare'].mode()[0])


# Extracting useful info from Cabin column .... 

# In[ ]:


def extract2(cabin):
    x=re.findall('([a-zA-Z]+)',cabin)
    if x==[]:
        return 'nil'
    for c in x:
        return c
train['Cabin']=train['Cabin'].transform(extract2)
test['Cabin']=test['Cabin'].transform(extract2)


# Dropping columns that aren't required for prediction...

# In[ ]:


target=train['Survived']
train=train.drop(['PassengerId','Name','Ticket'],axis=1)
test=test.drop(['PassengerId','Name','Ticket'],axis=1)


# Using LabelEncoding to convert categorical data to numeric data and multiplying by 10 so as to properly diffferentiate between various sub categories 

# In[ ]:


from sklearn.preprocessing import LabelEncoder as le
for col in train.columns:
    if train[col].dtype=='object':
        train[col]=le().fit_transform(train[col].astype(str))
        test[col]=le().fit_transform(test[col].astype(str))


# Adding features Family and Alone as these can be critical info for survival prediction....

# In[ ]:


def alone(d):
    if d>0:
        return 1
    else:
        return 0
train['Family_Size']=train['SibSp']+train['Parch']
train['Alone']=train['Family_Size'].transform(alone)
test['Family_Size']=test['SibSp']+test['Parch']
test['Alone']=test['Family_Size'].transform(alone)


# In[ ]:


train=train.drop(['Survived'],axis=1)


# Data Visualization to show features and target relation using TABLEAU
# 

# In[ ]:


get_ipython().run_cell_magic(u'html', u'', u"<div class='tableauPlaceholder' id='viz1535710855713' style='position: relative'><noscript><a href='#'><img alt='various features alongside Survived ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic_610&#47;Story1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Titanic_610&#47;Story1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic_610&#47;Story1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1535710855713');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1016px';vizElement.style.height='991px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# Some more relations between features

# In[ ]:


get_ipython().run_cell_magic(u'html', u'', u"<div class='tableauPlaceholder' id='viz1535718122614' style='position: relative'><noscript><a href='#'><img alt='Story 2 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic2_32&#47;Story2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Titanic2_32&#47;Story2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic2_32&#47;Story2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1535718122614');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1016px';vizElement.style.height='991px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# **PREDICTION**
# After testing the model on various classifiers....Neural Networks produces one of the best results.Here,I would be implementing my NN using keras library 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Dropout
seq=Sequential()
seq.add(Dense(8,activation='sigmoid'))
seq.add(Dropout(0.1))
seq.add(Dense(4,activation='softmax'))
seq.add(Dropout(0.1))
seq.add(Dense(1,activation='relu'))


# converting the train and test data to np.array alongside target

# In[ ]:


train=np.array(train)
test=np.array(test)


# In[ ]:


target=np.array(target).reshape((-1,1))


# Splitting training data to train and validation data in the ratio of 7:3

# In[ ]:


from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ztrain,ztest=tts(train,target,train_size=0.7)


# Finally making our NN fit over the train dataset.....

# In[ ]:


seq.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='rmsprop')

seq.fit(xtrain,ztrain,validation_data=(xtest,ztest),epochs=2000,batch_size=16)


# Crosschecking the final model against validation data for checking the overall accuracy

# In[ ]:


from sklearn.metrics import accuracy_score as acs
print(acs(ztest,seq.predict_classes(xtest)))


# Predict survivors from test dataset

# In[ ]:


a=seq.predict_classes(test)


# preparing result file that has to be submitted to competition

# In[ ]:


o=pd.DataFrame(a,columns=['Survived'])
o.index=pd.read_csv('../input/test.csv')['PassengerId']
o.index.name='PassengerId'


# In[ ]:


o.to_csv('result.csv')


# In[ ]:





# In[ ]:





# In[ ]:




