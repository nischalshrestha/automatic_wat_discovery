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


import pandas as pd
import numpy as np
import os


# In[ ]:


train_df = pd.read_csv("../input/train.csv", index_col='PassengerId')
test_df = pd.read_csv("../input/test.csv", index_col='PassengerId')


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


test_df['Survived']= -888 #adding survived with a default value


# In[ ]:


df = pd.concat((train_df, test_df) , axis=0)


# In[ ]:


df.info()


# In[ ]:


df.head()


# ## Data Munging

# In[ ]:


df[df.Embarked.isnull()]


# In[ ]:


df.Embarked.value_counts()


# In[ ]:


pd.crosstab(df[df.Survived !=-888].Survived,df[df.Survived != -888].Embarked)


# In[ ]:


df.groupby(['Pclass', 'Embarked']).Fare.median()


# In[ ]:


df.Embarked.fillna('C', inplace=True)


# In[ ]:


df[df.Embarked.isnull()]


# In[ ]:


df.info()


# In[ ]:


df[df.Fare.isnull()]


# In[ ]:


median_fare=df.loc[(df.Pclass ==3) & (df.Embarked == 'S'),'Fare'].median()


# In[ ]:


print(median_fare)


# In[ ]:


df.Fare.fillna(median_fare, inplace=True)


# In[ ]:


pd.options.display.max_rows =15


# In[ ]:


df[df.Age.isnull()]


# In[ ]:


df.Age.plot(kind='hist',bins=20,color='c')


# In[ ]:


df.groupby('Sex').Age.median()


# In[ ]:


df[df.Age.notnull()].boxplot('Age','Sex')


# In[ ]:


df[df.Age.notnull()].boxplot('Age','Pclass')


# In[ ]:


def GetTitle(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]


# In[ ]:


df['Title']= df.Name.map(lambda x :GetTitle(x))


# In[ ]:


df[df.Age.notnull()].boxplot('Age','Title')


# In[ ]:


title_age_median=df.groupby('Title').Age.transform('median')
df.Age.fillna(title_age_median, inplace=True)


# ## Outliers

# In[ ]:


df.loc[df.Age>70]


# In[ ]:


df.loc[df.Fare == df.Fare.max()]


# In[ ]:


LogFare=np.log(df.Fare+1.0)


# In[ ]:


LogFare.plot(kind='hist')


# In[ ]:


pd.qcut(df.Fare,4)


# In[ ]:


pd.qcut(df.Fare,4,labels=['very low','low','high','very high'])


# In[ ]:


pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high']).value_counts().plot(kind='bar', color='c', rot=0);


# In[ ]:


df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high'])


# In[ ]:


df['AgeState']=np.where(df['Age']>=18,'Adult','Child')


# In[ ]:


df['AgeState'].value_counts()


# In[ ]:


pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].AgeState)


# In[ ]:


df['FamilySize']=df.Parch+df.SibSp+1


# In[ ]:


df['FamilySize'].plot(kind='hist')


# In[ ]:


df.loc[df.FamilySize==df.FamilySize.max(),['Name','Survived','FamilySize','Ticket']]


# In[ ]:


pd.crosstab(df[df.Survived !=-888].Survived,df[df.Survived!=-888].FamilySize)


# In[ ]:


df['IsMother']=np.where(((df.Sex=='female')&(df.Parch>0)&(df.Age>18)&(df.Title!='Miss')),1,0)


# In[ ]:


pd.crosstab(df[df.Survived !=-888].Survived,df[df.Survived!=-888].IsMother)


# In[ ]:


df.Cabin


# In[ ]:


df.Cabin.unique()


# In[ ]:


df.loc[df.Cabin=='T']


# In[ ]:


df.loc[df.Cabin=='T','Cabin']=np.NaN


# In[ ]:


def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')
df['Deck']= df['Cabin'].map(lambda x: get_deck(x))


# In[ ]:


df.Deck.value_counts()


# In[ ]:


pd.crosstab(df[df.Survived !=-888].Survived, df[df.Survived !=-888].Deck)


# ## Categorical Fetaure Enconding

# In[ ]:


df['IsMale']=np.where(df.Sex=='Male',1,0)


# In[ ]:


df=pd.get_dummies(df,columns=['Deck','Pclass','Title','Fare_Bin','Embarked','AgeState'])


# In[ ]:


print(df.info())


# In[ ]:


# drop and reorder columns
df.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis=1,inplace=True)


# In[ ]:


# reorder columns
columns = [column for column in df.columns if column != 'Survived']
columns = ['Survived'] + columns
df = df[columns]


# In[ ]:


# check info again
df.info()


# In[ ]:


# train data
train_df= df[df.Survived != -888]
# test data
columns = [column for column in df.columns if column != 'Survived']
test_df= df[df.Survived == -888][columns]


# # building predictive models

# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# # data preparation

# In[ ]:


X = train_df.loc[:,'Age':].as_matrix().astype('float')
y = train_df['Survived'].ravel()


# In[ ]:


print(X.shape,Y.shape)


# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


# average survival in train and test
print('mean survival in train : {0:.3f}'.format(np.mean(y_train)))
print('mean survival in test : {0:.3f}'.format(np.mean(y_test)))


# ## Building basline Model

# In[ ]:


import sklearn
from sklearn.dummy import DummyClassifier


# In[ ]:


# create model
model_dummy = DummyClassifier(strategy='most_frequent', random_state=0)


# In[ ]:


# train model
model_dummy.fit(X_train, y_train)


# In[ ]:


print('score for baseline model : {0:.2f}'.format(model_dummy.score(X_test, y_test)))


# In[ ]:


# peformance metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


# In[ ]:


print('Accuracy for baseline model {:.2f}'.format(accuracy_score(Y_test,model_dummy.predict(X_test))))


# In[ ]:


print('Confusion Matrix baseline model\n {}'.format(confusion_matrix(Y_test,model_dummy.predict(X_test))))


# In[ ]:


print('precision for basleline model :{:.2f}'.format(precision_score(Y_test,model_dummy.predict(X_test))))


# ****Logistic Regression Model****

# In[ ]:


# import function
from sklearn.linear_model import LogisticRegression


# In[ ]:


# create model
model_lr_1= LogisticRegression(random_state=0)


# In[ ]:


# train model
model_lr_1.fit(X_train, y_train)


# In[ ]:


#evaluate model
print('score for logistic regression - version 1 : {0:.2f}'.format(model_lr_1.score(X_test,Y_test)))


# In[ ]:


# performance metrics
# accuracy
print('accuracy for logistic regression - version 1 : {0:.2f}'.format(accuracy_score(Y_test, model_lr_1.predict(X_test))))
# confusion matrix
print('confusion matrix for logistic regression - version 1: \n {0}'.format(confusion_matrix(Y_test, model_lr_1.predict(X_test))))
# precision 
print('precision for logistic regression - version 1 : {0:.2f}'.format(precision_score(Y_test, model_lr_1.predict(X_test))))
# precision 
print('recall for logistic regression - version 1 : {0:.2f}'.format(recall_score(Y_test, model_lr_1.predict(X_test))))


# In[ ]:


# model coefficients
model_lr_1.coef_


# ### Hyperparameter Optimization

# In[ ]:


# base model 
model_lr = LogisticRegression(random_state=0)


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


parameters = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1','l2']}
clf = GridSearchCV(model_lr, param_grid=parameters, cv=3)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


clf.best_params_


# In[ ]:


print('best score : {0:.2f}'.format(clf.best_score_))


# In[ ]:


# evaluate model
print('score for logistic regression - version 2 : {0:.2f}'.format(clf.score(X_test, y_test)))


# ### Feature Normalization and Standardization

# In[ ]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[ ]:


# feature normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)


# In[ ]:


X_train_scaled[:,0].min(),X_train_scaled[:,0].max()


# In[ ]:


# normalize test data
X_test_scaled = scaler.transform(X_test)


# In[ ]:


# feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# #### Create model after standardization

# In[ ]:


# base model 
model_lr = LogisticRegression(random_state=0)
parameters = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1','l2']}
clf = GridSearchCV(model_lr, param_grid=parameters, cv=3)
clf.fit(X_train_scaled, y_train)


# In[ ]:


clf.best_score_


# In[ ]:


# evaluate model
print('score for logistic regression - version 2 : {0:.2f}'.format(clf.score(X_test_scaled, y_test)))


# In[ ]:




