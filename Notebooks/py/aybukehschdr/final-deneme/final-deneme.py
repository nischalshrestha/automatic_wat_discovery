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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
# ---

get_ipython().magic(u'matplotlib inline')
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
pd.options.display.max_rows = 100

import csv
#from sklearn import cross_validation
traindf = pd.read_csv('../input/train.csv' ,header=0)
test_df    = pd.read_csv('../input/test.csv' ,header=0)
train_df = pd.read_csv('../input/train.csv' ,header=0)


# In[ ]:


def tum_verileri_birlestir():
  
   
    targets = traindf.Survived
    traindf.drop('Survived',1,inplace=True)
    

    tumveri = traindf.append(test_df)
    tumveri.reset_index(inplace=True)
    tumveri.drop('index',inplace=True,axis=1)
    
    return tumveri


# In[ ]:


train_df = pd.read_csv('../input/train.csv' ,header=0)


# In[ ]:


tumveri = tum_verileri_birlestir()
tumveri.head()


# In[ ]:


train_df['aile']=train_df['Parch']+train_df['SibSp']
train_df['aile'].loc[train_df['aile']>0]=1
train_df['aile'].loc[train_df['aile']==0]=0


# In[ ]:


train_df['aile_boyut']=train_df['SibSp']+train_df['Parch']
train_df['kisibasiucret']=train_df['Fare']/(train_df['aile_boyut']+1)


# In[ ]:


train_df['Unvan'] = train_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())


# In[ ]:


unvans = list(enumerate(np.unique(train_df['Unvan'])))
unvan_dict = {unvan : i for i, unvan in unvans } 
#Burada dataframemiz içindeji gelen her elemean sırası ile unique sayısal değerler ile değiştirilmiş dizi elemanları ile yer değiştirir
train_df.Unvan = train_df.Unvan.map( lambda x: unvan_dict[x]).astype(int)


# In[ ]:


train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# In[ ]:


if len(train_df.Embarked[train_df.Embarked.isnull()]) > 0:
     train_df.Embarked[train_df.Embarked.isnull()]=tumveri.Embarked.dropna().mode().values


# In[ ]:


#enumerate içerden dönen değerleri yazar.
#np.unique ise arraydeki elemanlar için unique bir değer döndürür. astype(int) ile de bu değerler sayısallaştırılır
Ports = list(enumerate(np.unique(train_df['Embarked'])))
Ports_dict = { name : i for i, name in Ports } 
#Burada dataframemiz içindeji gelen her elemean sırası ile unique sayısal değerler ile değiştirilmiş dizi elemanları ile yer değiştirir
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# In[ ]:



median_age = tumveri['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age


# In[ ]:


train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','SibSp','Parch'], axis=1) 


# In[ ]:


test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# In[ ]:


if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
     test_df.Embarked[ test_df.Embarked.isnull() ] = tumveri.Embarked.dropna().mode().values
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# In[ ]:


median_age = tumveri['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age
    


# In[ ]:


if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):
          median_fare[f] = tumveri[ tumveri.Pclass == f+1 ]['Fare'].dropna().median()
                                              
      
    for f in range(0,3):                                              
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]


# In[ ]:


test_df['Unvan'] = test_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

unvans = list(enumerate(np.unique(test_df['Unvan'])))
unvan_dict = {unvan : i for i, unvan in unvans } 
#Burada dataframemiz içindeji gelen her elemean sırası ile unique sayısal değerler ile değiştirilmiş dizi elemanları ile yer değiştirir
test_df.Unvan = test_df.Unvan.map( lambda x: unvan_dict[x]).astype(int)


# In[ ]:


test_df['aile']=test_df['Parch']+test_df['SibSp']
test_df['aile'].loc[test_df['aile']>0]=1
test_df['aile'].loc[test_df['aile']==0]=0

test_df['aile_boyut']=test_df['SibSp']+test_df['Parch']
test_df['kisibasiucret']=test_df['Fare']/(test_df['aile_boyut']+1)


# In[ ]:


ids = test_df['PassengerId'].values

test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','SibSp','Parch'], axis=1) 


# In[ ]:


print ("train dataset.keys(): {}".format(train_df.keys()))
print ("test dataset.keys(): {}".format(test_df.keys()))


# In[ ]:


X=train_df.ix[:,1:9]
y=train_df.Survived
print("my_train shape:{}".format(train_df.shape))
print("X_train shape: {}".format(X.shape))
print("y_train shape: {}".format(y.shape))


# In[ ]:


X_test=test_df.ix[:,0:8]

print("test_df shape:{}".format(test_df.shape))
print("Xrest shape: {}".format(X_test.shape))
print ("train dataset.keys(): {}".format(X_test.keys()))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0, max_depth=40,learning_rate=0.1)
gbrt.fit(X,y)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X, y)))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=40,random_state=0)
forest.fit(X,y)
print("Accuracy on training set: {:.3f}".format(forest.score(X, y)))


# In[ ]:


prediction_y=forest.predict(X_test)


# In[ ]:


test_df["PassengerId"]=ids


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction_y
    })
submission.to_csv('titanic.csv', index=False)


# In[ ]:




