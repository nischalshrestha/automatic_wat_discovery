#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# First let us load the data from the train.csv and test.csv into two dataframes

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# Lets us explore the data in the two dataframes train and test

# In[ ]:


train.shape


# In[ ]:


test.shape


# As you can see there are 891 rows and 12 columns in the train dataframe.The test dataframe has 418 rows and eleven columns

# Next lets look at the columns of both the dataframes

# In[ ]:


print(train.columns)
print(test.columns)


# The aim is to predict the Survived label for the test data.
# 

# Let us look at the datatypes for the columns in the train dataframe

# In[ ]:


train.dtypes


# As you can see there are 5 integer columns and two float columns and 5 object columns

# Lets have a look at the data in detail

# In[ ]:


train.head()


# The first column is just a serial id that does not provide any useful information.The second column is the Survived column which indicates whether the particular person survived or not.The survived column is the label and the other columns can be used to predict whether the person survived or not.
# 

# The third colmn is Pclass.Let us look at the values and the counts of Pclass

# In[ ]:


train['Pclass'].value_counts()


# As you can see above the Pclass is catogoerical variables which takes values 1,2 and 3.You can see that most number of passengers are in class 3

# Let us look at the fraction of people who survived in each class

# In[ ]:


train[['Survived','Pclass']].groupby(['Pclass']).mean().sort_values('Survived',ascending=False)


# You can see almost 63% of 1 class survived.47% of 2nd class and 24 % of the third class survived

# In[ ]:


train[['Survived','Pclass']].groupby(['Pclass']).mean().sort_values('Survived',ascending=False).plot.bar()


# Lets next look at Name column and extract the title out of the name column

# In[ ]:


train['Title']=train['Name'].str.extract('([A-Za-z]+)\.')


# In[ ]:


train['Title'].value_counts()


# In[ ]:


pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')


# In[ ]:


train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train['Title'] = train['Title'].map(title_mapping)
train['Title'] = train['Title'].fillna(0)


# In[ ]:


train.head()


# Now lets look at the Age column

# In[ ]:


train.Age.isnull().sum()


# There are 177 null values lets fill them with mean value of age

# In[ ]:


train['Age']=train.Age.fillna(train.Age.mean())


# In[ ]:


sns.distplot(train['Age'])


# In[ ]:


train['Age-band']=pd.cut(train['Age'],5)


# In[ ]:


train['Age-band'].value_counts()


# In[ ]:


train.loc[train['Age']<16,'Age']=1
train.loc[(train['Age']>=16)&(train['Age']<32),'Age']=2
train.loc[(train['Age']>=32)&(train['Age']<48),'Age']=3
train.loc[(train['Age']>=48)&(train['Age']<64),'Age']=4
train.loc[(train['Age']>=64),'Age']  =5                           


# In[ ]:


train.Age.value_counts()


# In[ ]:


train['Fare-band']=pd.qcut(train['Fare'],4)


# In[ ]:


train['Fare-band'].value_counts()


# In[ ]:


train.loc[train['Fare']<7,'Fare']=1
train.loc[(train['Fare']>=7)&(train['Fare']<14),'Fare']=2
train.loc[(train['Fare']>=14)&(train['Fare']<31),'Fare']=3
train.loc[(train['Fare']>=31),'Fare']=4


# In[ ]:


train['Fare'].value_counts()


# In[ ]:


train[['Survived','Fare']].groupby('Fare').mean().sort_values('Survived',ascending=False)


# In[ ]:


train[['Survived','Fare']].groupby('Fare').mean().sort_values('Survived',ascending=False).plot.bar()


# It is clear that people with higher fare had a greater chance of survival

# In[ ]:


train['FamilySize']=train['SibSp']+train['Parch']+1
train['FamilySize'].value_counts()


# In[ ]:


train[['Survived','FamilySize']].groupby('FamilySize').mean().sort_values('Survived',ascending=False)


# In[ ]:





# In[ ]:


train['Cabin'].isnull().sum()


# This column has many null values so will drop it

# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train['Embarked'].isnull().sum()


# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


train.Embarked.fillna('S',inplace=True)


# In[ ]:


train[['Survived','Embarked']].groupby('Embarked').mean().sort_values('Survived',ascending=False)


# In[ ]:


train.head()


# In[ ]:


drop_columns=['Name','SibSp','Parch','Ticket','Age-band','Fare-band','PassengerId']
train.drop(drop_columns,axis=1,inplace=True)


# In[ ]:


train.head()
train['Sex']=train.Sex.map({'male':0,'female':1})
train['Embarked']=train.Embarked.map({'S':0,'C':1,'Q':2})
train.head()


# In[ ]:


cat_cols = ['Pclass', 'Age', 'Fare', 'Embarked', 'Title','FamilySize']
train= pd.get_dummies(train, columns = cat_cols,drop_first=True)
train.head()


# In[ ]:


X=train.iloc[:,1:].values
y=train.iloc[:,0].values
X.shape
y.shape


# In[ ]:


#split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {'Logistic Regression': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(n_estimators=10), 
          'K-Nearest Neighbors':KNeighborsClassifier(n_neighbors=1),
            'Linear SVM':SVC(kernel='rbf', gamma=.10, C=1.0)}
accuracy={}
for descr,model in models.items():
    mod=model
    mod.fit(X_train,y_train)
    prediction=mod.predict(X_test)    
    accuracy[descr]=((prediction==y_test).mean())
print(accuracy)



# In[ ]:




