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


import keras
import seaborn as sns 
import matplotlib.pyplot as plt


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


#checking null values in the training set
train.isnull().sum()


# In[ ]:


#checking null values in the test set
test.isnull().sum()


# In[ ]:


cleaned_train=train[pd.notnull(train["Age"])]


# In[ ]:


cleaned_test=test[pd.notnull(test["Age"])]


# In[ ]:


sns.distplot(cleaned_train["Age"])


# In[ ]:


cleaned_train["Age"].std()


# In[ ]:


sns.distplot(cleaned_test["Age"])


# In[ ]:


cleaned_test["Age"].std()


# In[ ]:


#handling missing values of age
mean_age_train=train["Age"].mean()
mean_age_test=test["Age"].mean()
std_age_train=train["Age"].std()
std_age_test=test["Age"].std()


# In[ ]:


age_null_train=train["Age"].isnull().sum()
age_null_test=test["Age"].isnull().sum()


# In[ ]:


train_age=np.random.randint(mean_age_train-std_age_train,mean_age_train+std_age_train,size=age_null_train)
test_age=np.random.randint(mean_age_test-std_age_test,mean_age_test+std_age_test,size=age_null_test)


# In[ ]:


train["Age"][np.isnan(train["Age"])]=train_age
test["Age"][np.isnan(test["Age"])]=test_age


# In[ ]:


train["Age"].isnull().sum()


# In[ ]:


test["Age"].isnull().sum()


# In[ ]:


#there is one missing of fare in test set. Replacing it with the median of fare
test["Fare"].fillna(test["Fare"].mean(),inplace=True)


# In[ ]:


y_train=train.iloc[:,1]


# In[ ]:


plt.figure(figsize=(14,12))
sns.heatmap(train.corr(), vmax=0.6, square=True, annot=True)


# In[ ]:


sns.barplot(train["Pclass"],y_train)


# In[ ]:


sns.barplot(train["SibSp"],y_train)


# In[ ]:


sns.barplot(train["Parch"],y_train)


# In[ ]:


sns.barplot(train["Sex"],y_train)


# In[ ]:


sns.barplot(train["Embarked"],y_train)


# In[ ]:


train["Embarked"]=train["Embarked"].fillna("S")


# In[ ]:


train["Family"]=train["SibSp"]+train["Parch"]+1       
test["Family"]=test["SibSp"]+test["Parch"]+1      


# In[ ]:


test.Fare[152]=test.Fare.median()    


# In[ ]:


plt.figure(figsize=(10,8))
train["Age_Band"]=pd.cut(train["Age"],5)
sns.barplot("Age_Band","Survived",data=train)


# In[ ]:


train.loc[train["Age"]<=16,"Age"]=0
train.loc[(train["Age"]>16) & (train["Age"]<=32),"Age"]=1
train.loc[(train["Age"]>32) & (train["Age"]<=48),"Age"]=2
train.loc[(train["Age"]>48) & (train["Age"]<=64),"Age"]=3
train.loc[(train["Age"]>64) & (train["Age"]<=80),"Age"]=4


# In[ ]:


test.loc[test["Age"]<=16,"Age"]=0
test.loc[(test["Age"]>16) & (test["Age"]<=32),"Age"]=1
test.loc[(test["Age"]>32) & (test["Age"]<=48),"Age"]=2
test.loc[(test["Age"]>48) & (test["Age"]<=64),"Age"]=3
test.loc[(test["Age"]>64) & (test["Age"]<=80),"Age"]=4


# In[ ]:


#### plt.figure(figsize=(10,8))
train["Fare_Band"]=pd.qcut(train["Fare"],4)
sns.barplot("Fare_Band","Survived",data=train)


# In[ ]:


train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3


# In[ ]:


test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[ test['Fare'] > 31, 'Fare'] = 3


# In[ ]:


train.head()


# In[ ]:


train.drop("Age_Band",axis=1,inplace=True)
train.drop("Fare_Band",axis=1,inplace=True)


# In[ ]:


combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

#replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


sns.barplot("Title","Survived",data=train)


# In[ ]:


X_train=train[["Sex","Age","Pclass","Family","Embarked","Title","Fare"]]
X_test=test[["Sex","Age","Pclass","Family","Embarked","Title","Fare"]]
y_train=train["Survived"]


# In[ ]:


X_train.head()


# In[ ]:


X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test)


# In[ ]:


#X_train.drop("Fare",axis=1,inplace=True)             
#X_test.drop("Fare",axis=1,inplace=True)              
X_train.drop("Title_Royal",axis=1,inplace=True)
        


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 14))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 30)


# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

y_pred=y_pred.astype(int)
y_pred=y_pred.reshape(-1)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('kernel.csv', index=False)

