#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.countplot(data=train,x='Survived',hue='Pclass')


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[ ]:


def calculate_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train[['Age','Pclass']].head()


# In[ ]:


train['Age']=train[['Age','Pclass']].apply(calculate_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),cmap='viridis',yticklabels=False,cbar=False)


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


sns.heatmap(train.isnull(),cmap='viridis',yticklabels=False,cbar=False)


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


sns.heatmap(train.isnull(),cmap='viridis',yticklabels=False,cbar=False)


# In[ ]:


train.head()


# In[ ]:


pd.get_dummies(train['Sex']).head(2)


# In[ ]:


sex=pd.get_dummies(train['Sex'],drop_first=True)


# In[ ]:


embark=pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train=pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head(2)


# In[ ]:





# In[ ]:


train.drop('PassengerId',axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


PC=pd.get_dummies(train['Pclass'],drop_first=True)


# In[ ]:


train=pd.concat([train,PC],axis=1)


# In[ ]:


train.head()


# In[ ]:


#train.drop([2,3],axis=1,inplace=True)


# In[ ]:


train.drop(['Embarked'],axis=1,inplace=True)


# In[ ]:


train.drop(['Name','Sex','Ticket','Pclass'],axis=1,inplace=True)


# In[ ]:


X=train.drop('Survived',axis=1)
y=train['Survived']


# In[ ]:





# In[ ]:


X.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions=logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,predictions)


# In[ ]:


test_data = pd.read_csv('../input/test.csv')


# In[ ]:


test_data.head()


# In[ ]:


sns.boxplot(x=test_data['Pclass'],y=test_data['Age'],data=test_data)


# In[ ]:


test_data['Age']=test_data[['Age','Pclass']].apply(calculate_age,axis=1)


# In[ ]:


test_data.head()


# In[ ]:


sns.heatmap(test_data.isnull(),cmap='viridis',yticklabels=False,cbar=False)


# In[ ]:


test_data.columns


# In[ ]:


test_data.info()


# In[ ]:



sns.boxplot(x='Pclass',y='Fare',data=test_data)


# In[ ]:


test_data[test_data['Pclass'] == 3]['Fare'].median()


# In[ ]:


def calculate_Fare(cols):
    Fare=cols[0]
    Pclass=cols[1]
    if pd.isnull(Fare):
        if Pclass == 1:
            return 60
        elif Pclass==2:
            return 15.75
        else:
            return 7.8958
    else:
        return Fare


# In[ ]:


test_data['Fare'] = test_data[['Fare','Pclass']].apply(calculate_Fare,axis=1)


# In[ ]:





# In[ ]:





# In[ ]:


sex = pd.get_dummies(test_data['Sex'],drop_first=True)
embark = pd.get_dummies(test_data['Embarked'],drop_first=True)
PC = pd.get_dummies(test_data['Pclass'],drop_first=True)


# In[ ]:


test_data = pd.concat([test_data,sex,embark,PC],axis=1)


# In[ ]:


test_data.head()


# In[ ]:


test_data.drop(['Sex','Embarked','Pclass','Name','Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


test_data.head(2)


# In[ ]:





# In[ ]:


test_data.info()


# In[ ]:





# In[ ]:





# In[ ]:


idx.head()


# In[ ]:


idx.rename(columns={'idx':'PassengerId'},inplace = True)


# In[ ]:


idx.head()


# In[ ]:


test_data.drop('PassengerId',inplace=True,axis=1)


# In[ ]:


predictions_new = logmodel.predict(test_data)


# In[ ]:


survived = pd.DataFrame({'Survived':predictions_new})


# In[ ]:


survived.head()


# In[ ]:


submission=pd.concat([idx,survived],axis=1)


# In[ ]:


submission.head()


# In[ ]:


OUTPUT_RESULT="submission1.csv"
submission.to_csv(OUTPUT_RESULT,index=False)


# In[ ]:




