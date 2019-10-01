#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


# cargar datos de entrenamiento
titanicDF=pd.read_csv('../input/train.csv')
print(titanicDF.columns)


# In[3]:


features=titanicDF[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
features.head(5)


# In[4]:


# convertir todo a un solo tipo de dato numerico

featuresDummies=pd.get_dummies(features,columns=['Pclass','Sex','Embarked'])
featuresDummies.head(5)


# In[5]:


np.isnan(featuresDummies).any()


# In[6]:


# etiquetas 
labels=titanicDF[['Survived']]


# In[7]:


#separar set de entrenamiento y prueba
from sklearn.model_selection import train_test_split
train_data,test_data, train_labels, test_labels=train_test_split(featuresDummies, labels, random_state=0)


# In[8]:


from sklearn.preprocessing import Imputer
imp = Imputer()
imp.fit(train_data)
train_data_finite=imp.transform(train_data)
test_data_finite=imp.transform(test_data)


# In[9]:


np.isnan(train_data_finite).any()


# In[10]:


# random forest walker

from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(max_depth=10,random_state=0)
classifier.fit(train_data_finite,train_labels)


# In[11]:


classifier.predict(test_data_finite)


# In[12]:


classifier.score(test_data_finite, test_labels)


# In[13]:


#intento con regresion logistica
from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(train_data_finite,train_labels)


# In[14]:


clf.predict(test_data_finite)
clf.score(test_data_finite,test_labels)


# In[15]:


#cargar datos de prueba para competencia
titanicTest=pd.read_csv('../input/test.csv')
featuresTest=titanicTest[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
featuresTestDummies=pd.get_dummies(featuresTest,columns=['Pclass','Sex','Embarked'])
#testData=imp.transform(featuresTestDummies)


# In[16]:


np.isnan(featuresTestDummies).any()


# In[17]:


impTest=Imputer(missing_values='NaN',strategy='mean',axis=0)
impTest.fit(featuresTestDummies)
testData=imp.transform(featuresTestDummies)


# In[18]:


np.isnan(testData).any()


# In[19]:


survived=classifier.predict(testData)


# In[20]:


ID=titanicTest['PassengerId'].values


# In[21]:


Results=pd.DataFrame({'PassengerId':ID,'Survived':survived})


# In[22]:


Results


# In[23]:


Results.to_csv('submission.csv',sep=',',index=False)


# In[ ]:





# In[ ]:





# In[25]:


elim=['art', 'thou']


# In[26]:


elim


# In[27]:


for word in elim:
    print(word)


# In[ ]:





# In[37]:


mapa={'a':2,'b':3,'c':1,'d':3}


# In[38]:


mapa


# In[ ]:





# In[39]:


max(mapa,key=mapa.get)


# In[40]:


texto='hola gola gola'


# In[42]:


for word in texto:
    print(word)


# In[ ]:





# In[43]:


literaturre='romeo romeo wherefore art thou romeo'


# In[45]:


exclude=['art','thou']


# In[48]:


dictionary={}
for word in literaturre.split():
    if word not in dictionary:
        dictionary[word]=1
    else: 
        dictionary[word]+=1

for word in exclude:
    if word in dictionary:
        del dictionary[word]


# In[49]:


dictionary


# In[52]:


maxValue=max(dictionary.values())


# In[53]:


maxValue


# In[55]:


resultado=[]
for key in dictionary:
    if dictionary[key]==maxValue:
        resultado.append(key)


# In[56]:


resultado


# In[63]:


loglines=[['mi2','jog','mid','pet'],['wz3',34,54,398],['a1','alps','cow','bar']]


# In[64]:


temp=[]
for i in range(3):
    temp.append(loglines[i][0])


# In[65]:


temp


# In[66]:


print(temp.sort())


# In[69]:


lista=sorted(temp)
indices=[]


# In[72]:



            


# In[73]:


indices


# In[ ]:




