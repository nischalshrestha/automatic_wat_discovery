#!/usr/bin/env python
# coding: utf-8

# **Using Logistic Regression**

# In[ ]:


# Importing Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#Reading data
train = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv") 


# In[ ]:


#explore train data
train.head()


# In[ ]:


#EDA using Seaborn
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[ ]:


train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# **Data Cleaning**

# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


#Now let's check that heat map again!
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


#drop the Cabin column and the row in Embarked that is NaN.
train.drop('Cabin',axis=1,inplace=True)
test_data.drop('Cabin',axis=1,inplace=True)


# **Converting Categorical Features**

# In[ ]:


#for train data
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)




# In[ ]:


#for test data
test_sex = pd.get_dummies(test_data['Sex'],drop_first=True)
test_embark = pd.get_dummies(test_data['Embarked'],drop_first=True)
test_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test_data = pd.concat([test_data,test_sex,test_embark],axis=1)


# In[ ]:


#fill null value of fare column with 0
test_data.Fare.fillna(0 ,inplace = True)


# **Building a Logistic Regression model**

# In[ ]:


#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived','PassengerId'],axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


#Training and Predicting
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


#Evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:


id = test_data['PassengerId']
predictions = logmodel.predict(test_data.drop('PassengerId', axis=1))


result = pd.DataFrame({ 'PassengerId' : id, 'Survived': predictions })
# output.to_csv('titanic-predictions.csv', index = False)
result.head()


# In[ ]:


#writing the output in csv 
result.to_csv('predictions_final.csv', index = False)


# In[ ]:




