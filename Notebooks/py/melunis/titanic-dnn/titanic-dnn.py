#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This is a logistic regression model to predict survival of passengers on the Titanic. Data is from Kaggle: https://www.kaggle.com/c/titanic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis') #we see a lot of missing data in terms of age and cabin


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis') #pattern also exists in the test dataset


# In[ ]:


#remove cabin column, too much missing data
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


#We must next fix the missing data from the "Age" category
sns.boxplot(x='Pclass',y='Age',data=train) #here we can see that the age is very dependent on the passanger's class. Due to this, we want to fill in the passanger's age due to it's class. 


# In[ ]:


sns.boxplot(x='Sex',y='Age',data=train) #the age is not that dependent on Sex


# In[ ]:


A1 = train[train['Pclass']==1]['Age'].mean()
A2 = train[train['Pclass']==2]['Age'].mean()
A3 = train[train['Pclass']==3]['Age'].mean()
def correct_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return A1

        elif Pclass == 2:
            return A2

        else:
            return A3

    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(correct_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(correct_age,axis=1)


# In[ ]:


train.info() #we are showing 2 rows where Embarked is null, lets drop these 2 columns
test.info() #we are showing 1 row where Fare is null. We cannot simply drop this column


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


train.info() #data cleaned


# In[ ]:


#lets replace missing fares with the median fare cost
median_fare = train['Fare'].median()
test['Fare'] = test['Fare'].replace(np.NaN, median_fare)
test.info()


# In[ ]:


#now lets convert some categorical features {Sex, PClass, Embarked}
sns.countplot(x='SibSp',data=train) #this looks like we can divide it to be 0,1, or 2 and above
sns.countplot(x='Parch',data=train) #same divide, 0,1, 2 and above


# In[ ]:





# In[ ]:


train.head()


# In[ ]:


#Lets consider the passanger's full family size and if the passanger is alone
train['FamilySize']=train['SibSp'] + train['Parch'];
train['Alone'] = 1;
train['Alone'].loc[train['FamilySize'] > 0]=0;

test['FamilySize']=test['SibSp'] + test['Parch'];
test['Alone'] = 1;
test['Alone'].loc[test['FamilySize'] > 1]=0;
test.head()


# In[ ]:


#Lets also gather the title of the passanger
train['title']=train['Name'].str.split(",", expand = True)[1].str.split(".",expand = True)[0]
test['title']=test['Name'].str.split(",", expand = True)[1].str.split(".",expand = True)[0]
train.head()

#We do not want titles with low counts, lets say 10 in the training data
count_min=10
title_names =train['title'].value_counts() >= count_min
Title_List = list(title_names[title_names==True].index)
train['title'] = train['title'].apply(lambda x: x if x in Title_List else 'Misc')
test['title'] = test['title'].apply(lambda x: x if x in Title_List else 'Misc')


# In[ ]:


#train['Ticket'].unique  #Here we can see that each ticket code is unique, lets avoid trying to use this for now


# In[ ]:


test.tail()


# In[ ]:


print(train['SibSp'].loc[train['SibSp']>4].count()) #shows that there is not a large number of values over 4 (only 12)
print(train['Parch'].loc[train['Parch']>2].count()) #shows not a large amount of values over 2 (only 15)
print(train['FamilySize'].loc[train['FamilySize']>6].count()) #This should be cut off at 6 (only 13 above)


# In[ ]:


#lets check the count of each value
print('SibSp')
print(train['SibSp'].loc[train['SibSp']==0].count())
print(train['SibSp'].loc[train['SibSp']==1].count())
print(train['SibSp'].loc[train['SibSp']==2].count())
print(train['SibSp'].loc[train['SibSp']==3].count())
print(train['SibSp'].loc[train['SibSp']==4].count())
print(train['SibSp'].loc[train['SibSp']>4].count())
print('parch')
print(train['Parch'].loc[train['Parch']==0].count())
print(train['Parch'].loc[train['Parch']==1].count())
print(train['Parch'].loc[train['Parch']==2].count())
print(train['Parch'].loc[train['Parch']>2].count())
print('FamilySize')
print(train['FamilySize'].loc[train['FamilySize']==0].count())
print(train['FamilySize'].loc[train['FamilySize']==1].count())
print(train['FamilySize'].loc[train['FamilySize']==2].count())
print(train['FamilySize'].loc[train['FamilySize']==3].count())
print(train['FamilySize'].loc[train['FamilySize']==4].count())
print(train['FamilySize'].loc[train['FamilySize']==5].count())
print(train['FamilySize'].loc[train['FamilySize']==6].count())
print(train['FamilySize'].loc[train['FamilySize']>6].count())


# In[ ]:


#to reduce small numbers in different family size categories, we make the last category include that value and up
def replace3(col):
    if col > 3:
        return 3
    else:
        return col
    
def replace4(col):
    if col > 4:
        return 4
    else:
        return col
    
def replace2(col):
    if col > 2:
        return 2
    else:
        return col


# In[ ]:


#I feel there was not enough of the counts 
train['SibSp']=train['SibSp'].apply(replace2)
train['Parch']=train['Parch'].apply(replace2)
test['SibSp']=test['SibSp'].apply(replace2)
test['Parch']=test['Parch'].apply(replace2)

train['FamilySize']=train['FamilySize'].apply(replace3)
test['FamilySize']=test['FamilySize'].apply(replace3)


# In[ ]:


sns.jointplot(data=train,x='SibSp',y='Survived',kind='kde') 
sns.jointplot(data=train,x='FamilySize',y='Survived',kind='kde') 


# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
pclass = pd.get_dummies(train['Pclass'],drop_first=True)

sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test = pd.get_dummies(test['Embarked'],drop_first=True)
pclass_test = pd.get_dummies(test['Pclass'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)
train = pd.concat([train,sex,embark,pclass],axis=1)
test.drop(['Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)
test = pd.concat([test,sex_test,embark_test,pclass_test],axis=1)

Sib = pd.get_dummies(train['SibSp'],drop_first=True)
parch = pd.get_dummies(train['Parch'],drop_first=True)
Title=pd.get_dummies(train['title'],drop_first=True)
Family=pd.get_dummies(train['FamilySize'],drop_first=True)

Sib_test = pd.get_dummies(test['SibSp'],drop_first=True)
parch_test = pd.get_dummies(test['Parch'],drop_first=True)
Title_test=pd.get_dummies(test['title'],drop_first=True)
Family_test=pd.get_dummies(test['FamilySize'],drop_first=True)

Sib.columns = ['S1','S2']
parch.columns = ['P1','P2']
Family.columns = ['F1','F2','F3']
Sib_test.columns = ['S1','S2']
parch_test.columns = ['P1','P2']
Family_test.columns = ['F1','F2','F3']

train.drop(['SibSp','Parch','title','FamilySize'],axis=1,inplace=True)
test.drop(['SibSp','Parch','title','FamilySize'],axis=1,inplace=True)
train = pd.concat([train,Sib,parch,Title,Family],axis=1)
test = pd.concat([test,Sib_test,parch_test,Title_test,Family_test],axis=1)


# In[ ]:


train.head()


# In[ ]:


def young_kid(col):
    if col < 13:
        return 1
    elif col <20:
        return 2
    elif col <26:
        return 3
    elif col <35:
        return 4
    else:
        return 5


# In[ ]:


train['Age_Cat']=train['Age'].apply(young_kid)
test['Age_Cat']=test['Age'].apply(young_kid)
print(train['Age_Cat'][train['Age_Cat']==1].count())
print(train['Age_Cat'][train['Age_Cat']==2].count())
print(train['Age_Cat'][train['Age_Cat']==3].count())
print(train['Age_Cat'][train['Age_Cat']==4].count())
print(train['Age_Cat'][train['Age_Cat']==5].count())


# In[ ]:


sns.jointplot(data=train,x='Age_Cat',y='Survived',kind='kde') #big takeaway is that category 3 does not survive well


# In[ ]:


AC = pd.get_dummies(train['Age_Cat'],drop_first=True)
AC_test = pd.get_dummies(test['Age_Cat'],drop_first=True)
AC.columns=['teen','young_adult','adult','adult_eld']
AC_test.columns = ['teen','young_adult','adult','adult_eld']
train.drop(['Age_Cat'],axis=1,inplace=True)
test.drop(['Age_Cat'],axis=1,inplace=True)
train = pd.concat([train,AC],axis=1)
test = pd.concat([test,AC_test],axis=1)


# In[ ]:


train.drop('PassengerId',axis=1,inplace=True)
train.head()


# In[ ]:


train['Alone']=train['Alone'].astype('bool')
train.head()


# In[ ]:


import tensorflow.contrib.learn as learn


# In[ ]:


feature_columns = learn.infer_real_valued_columns_from_input(train.drop('Survived',axis=1))


# In[ ]:


classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2, feature_columns = feature_columns)


# In[ ]:


feature_columns


# In[ ]:


classifier.fit(train.drop('Survived',axis=1),train['Survived'],max_steps=1000000)


# In[ ]:


predictions = classifier.predict_classes(test.drop('PassengerId',axis=1))


# In[ ]:


a=list(predictions)


# In[ ]:


a


# In[ ]:


test['Survived']=a


# In[ ]:


#test.drop('Survived',axis=1,inplace=True)


# In[ ]:


out = test[['PassengerId','Survived']]


# In[ ]:


out.head()


# In[ ]:


out.to_csv('Predictions.csv',index=False)


# In[ ]:




