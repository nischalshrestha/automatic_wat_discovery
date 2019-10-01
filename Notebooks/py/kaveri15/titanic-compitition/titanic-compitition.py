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
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
print("Dimension of Train data:",train.shape)
print("Dimension of Test data:", test.shape)


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train.head()


# In[ ]:


import matplotlib.pyplot as plt
sex_pivot = train.pivot_table(index= ['Sex'],values = ['Survived'])
sex_pivot.plot.bar()
plt.show()


# In[ ]:


class_pivot = train.pivot_table(index = ['Pclass'],values = ['Survived'])
class_pivot.plot.bar()
plt.show()


# In[ ]:


train['Age'].describe()


# In[ ]:


survived = train[train['Survived'] == 1]
died = train[train['Survived'] == 0]
survived['Age'].plot.hist(alpha = 0.6, color = 'red',bins = 50)
died['Age'].plot.hist(alpha = 0.4, color ='blue', bins = 50)
plt.legend(['Survived','Died'])
plt.show()


# In[ ]:


#Age Category
def process_age(df,cut_points,label_names):
    df['Age'] = df['Age'].fillna(-0.5)
    df['Age_categories'] = pd.cut(df['Age'],cut_points,labels = label_names)
    return df
cut_points =[-1,0,5,12,18,32,60,100]
label_names = ['Missing','Infant','Child','Teenager','Young Adult','Adult','Senior']
train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

train_pivot = train.pivot_table(index='Age_categories',values = 'Survived')
train_pivot.plot.bar()
plt.show()
test.head()


# In[ ]:


# Dummies for Sex,Pclass columns
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix = column_name)
    df = pd.concat([df,dummies],axis =1)
    return df
column_names =['Sex','Pclass','Age_categories','Embarked']
for column in column_names:
    train = create_dummies(train, column)
    test = create_dummies(test,column)
test.head()


# In[ ]:


from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split,cross_val_score
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score

#train.dropna(axis = 0)
#print(train.Pclass)



# In[ ]:


# using train_test_split

y = train.Survived
columns =['Fare','Sex_male','Pclass_1' ,'Pclass_2','Age_categories_Child' ,
          'Age_categories_Teenager' ,'Age_categories_Young Adult' ,'Age_categories_Adult' ,'Age_categories_Senior' ,'Embarked_C' ,'Embarked_Q']
X = train[columns]
X.head()


# In[ ]:


train_X,test_X,train_y,test_y = train_test_split(X,y,random_state = 1)


# In[ ]:


#To peform Predictions
my_model = make_pipeline(Imputer(),XGBRegressor(silent = True, n_estimator = 1000,learning_rate = 0.19,n_jobs = 500, random_state = 1))
my_model.fit(train_X,train_y)
predicts = my_model.predict(test_X)
#print(predicts)

def get_binary(x):
    prediction = []
    for i in x:
        if i >=0.5:
            prediction.append(1)
        if i < 0.5:
            prediction.append(0)
    return prediction
result = get_binary(predicts)
print('Acccuracy :',balanced_accuracy_score(test_y,result, sample_weight=None, adjusted=False))


# In[ ]:


test_data = test[columns]
test_predict = my_model.predict(test_data)
test_predicts = get_binary(test_predict)
print('Survived:',test_predicts)
output = pd.DataFrame({'PassengerId': test.PassengerId,'Survived':test_predicts })

output.to_csv('submission.csv', index=False)



# In[ ]:




