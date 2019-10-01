#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# # load data

# In[ ]:


data = pd.read_csv('../input/train.csv')


# # Data Summary

# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# # Filling null Values

# In[ ]:


plt.figure(figsize = (8,6))
sns.countplot(x='Embarked',data=data)


# In[ ]:


data['Embarked'][data['Embarked'].isnull()]='S'


# In[ ]:


data['Embarked'].isnull().sum()


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=data)


# In[ ]:


data['Age'][data['Age'].isnull()]=data['Age'].mean()


# In[ ]:


data['Age'].isnull().sum()


# In[ ]:


data['Has Cabin']= data['Cabin'].apply(lambda x:0 if isinstance(x,float) else 1)


# In[ ]:


data['Survived'].isnull().sum()


# In[ ]:


data['Survived'].fillna(value=0,inplace=True)


# In[ ]:


data['Fare'][data['Fare'].isnull()]= data['Fare'].mean()


# In[ ]:


data.isnull().sum()


# # Visualisation

# In[ ]:


sns.countplot(x='Survived',data=data,hue='Sex')
print('Percentage of Male Survived: ',data['Survived'][data['Sex']=='male'].value_counts(normalize=True)[1]*100)
print('Percentage of Female Survived: ',data['Survived'][data['Sex']=='female'].value_counts(normalize=True)[1]*100)


# In[ ]:


sns.barplot(x='Pclass',y='Survived',data=data)
print('Percentage of class 1 passenger Survived: ',data['Survived'][data['Pclass']== 1].value_counts(normalize=True)[1]*100)
print('Percentage of class 2 passenger Survived: ',data['Survived'][data['Pclass']== 2].value_counts(normalize=True)[1]*100)
print('Percentage of class 3 passenger Survived: ',data['Survived'][data['Pclass']== 3].value_counts(normalize=True)[1]*100)


# In[ ]:


sns.barplot(x='Has Cabin',y='Survived',data=data)
print('Percentage of passengers have Cabin & Survived: ',data['Survived'][data['Has Cabin']== 1].value_counts(normalize=True)[1]*100)
print('Percentage of passengers dont have Cabin & Survived: ',data['Survived'][data['Has Cabin']== 0].value_counts(normalize=True)[1]*100)


# In[ ]:


sns.countplot(x='Embarked',data=data)
print('percentage of P embarked from S: ', ((data['Embarked']=='S').value_counts())/(data['Embarked'].count())*100)
print('percentage of P embarked from C: ', ((data['Embarked']=='C').value_counts())/(data['Embarked'].count())*100)
print('percentage of P embarked from Q: ', ((data['Embarked']=='Q').value_counts())/(data['Embarked'].count())*100)


# In[ ]:


sns.barplot(x='Embarked',y='Survived',data=data)
print('Percentage of S embarked survived: ',data['Survived'][data['Embarked']=='S'].value_counts(normalize=True)[1]*100)
print('Percentage of Q embarked survived: ',data['Survived'][data['Embarked']=='Q'].value_counts(normalize=True)[1]*100)
print('Percentage of C embarked survived: ',data['Survived'][data['Embarked']=='C'].value_counts(normalize=True)[1]*100)


# # Feature Engineering

# In[ ]:


import random
random.sample(list(data['Name'].values),10)
data['Title']=Titles=data['Name'].apply(lambda x: x.split(',')[1].split('.')[0] if ',' in x else x)
data['Title'].value_counts()


# In[ ]:


def map_marriage(Title):
    Title = Title.strip()
    if Title in ['Dr', 'Col', 'Capt','Major','Don','Rev','Dona','Jonkheer']:
        return 0
    if Title in ['the Countess', 'Lady', 'Sir']:
        return 1
    if Title in ['Mlle','Ms','Miss']:
        return 2
    if Title in ['Mrs']:
        return 3
    if Title in ['Mr','Master','Mme']:
        return 4


# In[ ]:


data['Title']=data['Title'].apply(map_marriage)


# In[ ]:


data.head()


# In[ ]:


data['Male']=data['Sex'].map({'male':1,'female':0})


# In[ ]:


data['FamSize']=data['SibSp'] + data['Parch']+1


# In[ ]:


def map_age(Age):
    if Age <=12:
        return 'Child'
    elif 12 < Age <=18:
        return 'Teenager'
    elif 18 < Age <=50:
        return 'Adult'
    else:
        return 'Old'


# In[ ]:


data['Age']=data['Age'].apply(map_age)


# In[ ]:


data['Age'] = data['Age'].map({'Child':1,'Teenager':2,'Adult':3,'Old':4})


# In[ ]:


def impute_fare(Fare):
    if Fare <=20:
        return 1
    if 20 < Fare <=40:
        return 2
    if Fare > 40:
        return 3


# In[ ]:


data['Fare'] = data['Fare'].apply(impute_fare)


# In[ ]:


data['Embarked'] = data['Embarked'].map({'S':1,'C':2,'Q':3})


# In[ ]:


data.head()


# In[ ]:


df_data = data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Has Cabin','SibSp','Parch'],axis=1)


# In[ ]:


df_data.head()


# # Machine Learning Algorithm 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=df_data.drop(['Survived'],axis=1)
y= df_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,random_state=50)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


pred_logmodel = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report
acc_logmodel = round(accuracy_score(pred_logmodel,y_test)*100,2)
print('Accuracy of Logmodel is: ',acc_logmodel)
print(classification_report(pred_logmodel,y_test))


# # Support Vector Machine (SVM)

# In[ ]:


from sklearn.svm import SVC
SVM_model = SVC()
SVM_model.fit(X_train,y_train)


# In[ ]:


pred_SVM = SVM_model.predict(X_test)


# In[ ]:


acc_SVM_model = round(accuracy_score(pred_SVM,y_test)*100,2)
print('Accuracy of SVM_model is: ',acc_SVM_model)
print(classification_report(pred_logmodel,y_test))


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train,y_train)


# In[ ]:


pred_DT = DT_model.predict(X_test)


# In[ ]:


acc_DT_model = round(accuracy_score(pred_DT,y_test)*100,2)
print('Accuracy of DT_model is: ',acc_DT_model)
print(classification_report(pred_DT,y_test))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier()
RF_model.fit(X_train,y_train)


# In[ ]:


pred_RF = RF_model.predict(X_test)


# In[ ]:


acc_RF_model = round(accuracy_score(pred_RF,y_test)*100,2)
print('Accuracy of RF_model is: ',acc_RF_model)
print(classification_report(pred_RF,y_test))


# # K Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train,y_train)
pred_KNN = KNN_model.predict(X_test)
acc_KNN_model = round(accuracy_score(pred_KNN,y_test)*100,2)
print('Accuracy of KNN_model is: ',acc_KNN_model)
print(classification_report(pred_KNN,y_test))


# # Model Evaluation 

# In[ ]:


models = pd.DataFrame({'Model':['Support Vector Machines','Random Forest','Decision Tree','Logistic Regression','K Nearest'],
                      'Score':[acc_SVM_model,acc_RF_model,acc_DT_model,acc_logmodel,acc_KNN_model]
                      }).sort_values(by='Score',ascending=False)


# In[ ]:


models


# # Import Test Data and Do all the Feature Engg. done on Training Data

# In[ ]:


test_data = pd.read_csv('../input/test.csv')


# In[ ]:


test_data.head()


# In[ ]:


test_data.info()


# In[ ]:


test_data['Age'][test_data['Age'].isnull()] = test_data['Age'].mean()
test_data['Fare'][test_data['Fare'].isnull()] = test_data['Fare'].mean()


# In[ ]:


test_data.info()


# In[ ]:


import random
random.sample(list(test_data['Name'].values),10)
test_data['Title']=Titles=test_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0] if ',' in x else x)
test_data['Title'].value_counts()


# In[ ]:


def map_marriage(Title):
    Title = Title.strip()
    if Title in ['Dr', 'Col', 'Capt','Major','Don','Rev','Dona','Jonkheer']:
        return 0
    if Title in ['the Countess', 'Lady', 'Sir']:
        return 1
    if Title in ['Mlle','Ms','Miss']:
        return 2
    if Title in ['Mrs']:
        return 3
    if Title in ['Mr','Master','Mme']:
        return 4


# In[ ]:


test_data['Title'] = test_data['Title'].apply(map_marriage)


# In[ ]:


test_data['Male']=test_data['Sex'].map({'male':1,'female':0})


# In[ ]:


def map_age(Age):
    if Age <=12:
        return 'Child'
    if 12 < Age <=18:
        return 'Teenager'
    if 18 < Age <=50:
        return 'Adult'
    if Age >50:
        return 'Old'


# In[ ]:


test_data['Age'] = test_data['Age'].apply(map_age)
test_data['Age'] = test_data['Age'].map({'Child':1,'Teenager':2,'Adult':3,'Old':4})


# In[ ]:


test_data.head()


# In[ ]:


test_data['Embarked'] = test_data['Embarked'].map({'S':1,'Q':2,'C':3})


# In[ ]:


def impute_fare(Fare):
    if Fare <=20:
        return 1
    if 20 < Fare <=40:
        return 2
    if Fare > 40:
        return 3


# In[ ]:


test_data['Fare'] = test_data['Fare'].apply(impute_fare)


# In[ ]:


test_data['Fare'].unique()


# In[ ]:


test_data['FamSize'] = test_data['SibSp'] + test_data['Parch'] +1


# In[ ]:


test_data['Has Cabin'] = test_data['Cabin'].apply(lambda x: 0 if isinstance(x,float) else 1)


# In[ ]:


test_data['Has Cabin'][test_data['Has Cabin']==1].count()


# In[ ]:


df_test = test_data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Has Cabin','SibSp','Parch'],axis=1)


# In[ ]:


test_data['Survived'] = RF_model.predict(df_test)


# In[ ]:


test_data.head(15)


# In[ ]:


submission = test_data[['PassengerId','Survived']]


# In[ ]:


submission.head(15)


# In[ ]:


submission.to_csv('New Submission.csv',index=False)


# In[ ]:




