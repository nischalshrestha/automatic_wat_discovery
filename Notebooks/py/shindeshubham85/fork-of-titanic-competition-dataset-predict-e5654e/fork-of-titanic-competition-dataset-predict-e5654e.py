#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


url_train = '../input/train.csv'
url_test = '../input/test.csv'
titanic_train = pd.read_csv(url_train)
titanic_test = pd.read_csv(url_test)
titanic = pd.concat([titanic_train, titanic_test],axis=0,sort=False)
titanic.head()


# In[ ]:


#Checking if our target variable is binary or not
sb.countplot(x='Survived',data=titanic)


# In[ ]:


#Checking Null values
titanic.isnull().sum()


# In[ ]:


titanic_data = titanic.drop(['Ticket'],1)
titanic_data.head()


# Now need to take care of the missing data for **Age**,** Embarked** and **Fare** variables. 
# Need to approximate- one way, to take mean age for all the missing values. Or, find if age is related to Pclass, and assign respective means.
# For Embarked, use the mode.
# For Fare, use medians for respective classes.

# In[ ]:


#sb.boxplot(x='Pclass',y='Age',data=titanic_data)
titanic_data.Embarked.mode()


# If Passenger belongs to Pclass 3, age assigned is 24, if 2, age is assigned 29, if 1 then 37.

# In[ ]:


def age_approx(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age
def cabin_approx(cols):
    cabin = cols[0]
    pclass = cols[1]
    if pd.isnull(cabin):
        return 0
    elif cabin[0] == ('C' or 'B'):
        return 3
    elif cabin[0] == ('A' or 'D' or 'E' or 'T'):
        return 2
    elif cabin[0] == ('F' or 'G'):
        return 1
    else:
        return 0
def embarked_approx(cols):
    emb = cols[0]
    if pd.isnull(emb):
        return 'S'
    else:
        return emb
def fare_approx(cols):
    fare = cols[0]
    pclass = cols[1]
    if pd.isnull(fare):
        if pclass == 1:
            return 55
        elif pclass == 2:
            return 20
        else:
            return 10
    else:
        return fare


# In[ ]:


titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(age_approx, axis=1)
titanic_data['Cabin'] = titanic_data[['Cabin', 'Pclass']].apply(cabin_approx, axis=1)
titanic_data['Embarked'] = titanic_data[['Embarked', 'Pclass']].apply(embarked_approx, axis=1)
titanic_data['Fare'] = titanic_data[['Fare','Pclass']].apply(fare_approx, axis=1)
titanic_data.isnull().sum()


# Creating a new variable, **FamilySize** comprising of SibSp (Siblings, Spouse) and Parch (Parents, Children).
# 
# Extracting Titles from Names.

# In[ ]:


titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']+1

titanic_data['IsAlone'] = 0
titanic_data.loc[titanic_data['FamilySize']== 1, 'IsAlone'] = 1

titanic_data['Title'] = titanic_data.Name.str.extract('([A-Za-z]+)\.',expand=False)

titanic_data['Title'] = titanic_data['Title'].replace(['Lady','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
titanic_data['Title'] = titanic_data['Title'].replace(['Mlle','Ms'],'Miss')
titanic_data['Title'] = titanic_data['Title'].replace('Mme','Mrs')
titanic_data.FamilySize.describe()


# In[ ]:


#Buckets of Family
titanic_data['FamilySizeGroup'] = 'Small'
titanic_data.loc[titanic_data['FamilySize']>=5,'FamilySizeGroup'] = 'Big'
titanic_data.loc[titanic_data['FamilySize']==1,'FamilySizeGroup'] = 'Alone'

#Buckets of Age
titanic_data.loc[titanic_data['Age'] <=14,'Age'] = 0
titanic_data.loc[(titanic_data['Age'] >14)&(titanic_data['Age'] <= 32),'Age'] = 1
titanic_data.loc[(titanic_data['Age'] >32)&(titanic_data['Age'] <= 48),'Age'] = 2
titanic_data.loc[(titanic_data['Age'] >48)&(titanic_data['Age'] <= 64),'Age'] = 3
titanic_data.loc[titanic_data['Age'] >64,'Age'] = 4

#Buckets of Fare
titanic_data.loc[titanic_data['Fare'] <=7.91,'Fare'] = 0
titanic_data.loc[(titanic_data['Fare'] >7.91)&(titanic_data['Fare'] <= 14.454),'Fare'] = 1
titanic_data.loc[(titanic_data['Fare'] >14.454)&(titanic_data['Fare'] <= 31),'Fare'] = 2
titanic_data.loc[titanic_data['Fare'] >31,'Fare'] = 3
titanic_data['Fare'] = titanic_data['Fare'].astype(int)

titanic_data.Age.isnull().sum()


# Converting Categorical Variables into Dummies.

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
family_mapping = {"Small": 1, "Alone": 0, "Big": 2}
titanic_data['Title'] = titanic_data['Title'].map(title_mapping)
titanic_data['FamilySizeGroup'] = titanic_data['FamilySizeGroup'].map(family_mapping)
titanic_data.FamilySizeGroup.describe()


# In[ ]:


gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark_location = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
titanic_data.drop(['Sex','Embarked','Name','SibSp','Parch','FamilySize'],axis=1,inplace=True)
titanic_dmy = pd.concat([titanic_data, gender, embark_location],axis=1)


# In[ ]:


#Checking for correlation between variables.
sb.heatmap(abs(titanic_dmy.corr()),square=True)
#print(titanic_dmy['male'].corr(titanic_dmy['Title']))


# In[ ]:


titanic_dmy.drop(['Cabin'], axis=1, inplace=True)
titanic_dmy.head()
#titanic_dmy.Survived.isnull().sum()


# In[ ]:


train_data = titanic_dmy.loc[titanic_dmy.Survived.notnull()]
n = train_data.shape[1]
test_data = titanic_dmy.loc[titanic_dmy.Survived.isnull()]
test_data.head()


# In[ ]:


#X = train_data.ix[:,(1,2,3,4,5,6,7,8,9)].values
length = np.arange(2,n,1)
X_train = train_data.iloc[:,length]
y_train = train_data.iloc[:,1]
X_test = test_data.iloc[:,length]
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
#from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier

clf1 = SVC(kernel='linear',C=1.0,random_state=3)
clf2 = XGBClassifier(random_state=5)
clf3 = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=3)
eclf = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2),('clf3',clf3)], voting='hard')

eclf.fit(X_train, y_train)
print('Training Score:', eclf.score(X_train, y_train))
print('Cross Val Score Mean: ', cross_val_score(eclf, X_train, y_train, cv = 5).mean())


# In[ ]:


y_pred = eclf.predict(X_test).astype(int)


# In[ ]:


submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':y_pred})
submission.head()


# In[ ]:


filename = 'titanic_submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:





# In[ ]:




