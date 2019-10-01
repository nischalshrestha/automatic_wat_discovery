#!/usr/bin/env python
# coding: utf-8

# **Implementing Titanic Solution using LOGISTIC REGRESSION**
# 
# accuracy = 0.83

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting the data 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.


# In[ ]:


data_init = pd.read_csv('../input/train.csv') # read the train data in data DataFrame
test_data = pd.read_csv('../input/test.csv') #read the test data
sol = pd.read_csv('../input/gender_submission.csv') #read the solutions
test_data['Survived']=sol['Survived'] #adding the solution to the test set 


# In[ ]:


data=(data_init.append(test_data)) #it is formed by appending the test and train data 
data.head()  #Reading the data to check the header


# In[ ]:


data.describe() #used to check if there are missing values


# In[ ]:


#fill the NaN value
data['Age'].fillna(data['Age'].median(),inplace=True)
data.Cabin.fillna('U',inplace=True)
data.Embarked.fillna('S',inplace=True)
data.Fare.fillna(data.Fare.mean(),inplace=True)
data.Age.fillna(data.Age.median(), inplace=True)


# **VISUALISING THE DATA**

# **1. Sex versus Survival**

# In[ ]:


total = data['Sex'].value_counts()
survived_sex = data[data['Survived']==1]['Sex'].value_counts()
died_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([total,survived_sex,died_sex])
df.index = ['Total','Survived','Died']
print(df)
df.plot(kind='bar')


# **2. Age vs Survival**

# In[ ]:


figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], color = ['g','r'],
         bins = 10,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# **3. Class versus Survival**

# In[ ]:


survived_1 = data[data['Pclass']==1]['Survived'].value_counts()
survived_2 = data[data['Pclass']==2]['Survived'].value_counts()
survived_3 = data[data['Pclass']==3]['Survived'].value_counts()
df = pd.DataFrame([survived_1,survived_2,survived_3])
df['total']=df[0]+df[1]
df.index = ['1st class','2nd class','3rd class']
df.rename(index=str,columns={0:'Survived',1:'Died'})
print (df)
df.plot(kind='bar',label=['Survived','Died'])


# **4. Fare versus Survival**

# In[ ]:


figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']],bins=10,label=['Survived','Died'])
plt.xlabel('Fare')
plt.ylabel('No. of People')
plt.legend()


# **5. Fare and Age versus Survival **

# In[ ]:


#Age versus Fare
data.plot.scatter('Age','Fare',c='Survived',colormap='jet',alpha=0.8,figsize=(15,8))


# **6. Embarkment versus Survival**

# In[ ]:


survived_embarkment  = data[data['Survived']==1]['Embarked'].value_counts()
died_embarkment = data[data['Survived']==1]['Embarked'].value_counts()
df = pd.DataFrame([survived_embarkment,died_embarkment])
df.index=['survived','died']
df.plot(kind='bar',stacked=True)


# **Encoding the value for Logistic Regression**
# 
#  One Hot Encoding for using the logistic reg. 
#  
#  For more reading [this link](http://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding)

# In[ ]:


data_set = data[['Pclass','Sex','Age','Fare','SibSp','Cabin']]
one_hot_encoded_training_predictors = pd.get_dummies(data_set)
one_hot_encoded_training_predictors.head()
X = one_hot_encoded_training_predictors
y = data['Survived']


# In[ ]:


#dividing the data in training and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.319, random_state=1)
logreg = LogisticRegression() #logistic regression using python
logreg.fit(X_train, y_train)


# In[ ]:


y_pred = logreg.predict(X_test) #predicting the values
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

