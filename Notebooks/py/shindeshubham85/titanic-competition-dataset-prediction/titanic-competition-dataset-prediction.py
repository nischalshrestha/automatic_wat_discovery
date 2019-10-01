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
titanic = pd.read_csv(url_train)
titanic.head()


# In[ ]:


#Checking if our target variable is binary or not
#sb.countplot(x='Survived',data=titanic)


# In[ ]:


#Checking Null values
titanic.isnull().sum()


# Dropping PassengerId, Name and Ticket because they are unique.
# Dropping Cabin because of too many null values.

# In[ ]:


titanic_data = titanic.drop(['PassengerId','Name','Ticket'],1)
titanic_data.head()


# Now need to take care of the missing data for Age variable. Need to approximate- one way, to take mean age for all the missing values.
# Or, find if age is related to Pclass, and assign respective means.

# In[ ]:


sb.boxplot(x='Pclass',y='Age',data=titanic_data)


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


# In[ ]:


titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(age_approx, axis=1)
titanic_data.isnull().sum()


# In[ ]:


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


# In[ ]:


titanic_data['Cabin'] = titanic_data[['Cabin', 'Pclass']].apply(cabin_approx, axis=1)
#titanic_data.isnull().sum()
sb.boxplot(x='Cabin',y='Fare',data=titanic_data)


# There are two null values in Embarked, we can just drop them.

# In[ ]:


titanic_data.dropna(inplace=True)
titanic_data.isnull().sum()


# Getting dummy variables from categorical ones.

# In[ ]:


gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
gender.head()


# In[ ]:


embark_location = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
embark_location.head()


# In[ ]:


titanic_data.drop(['Sex','Embarked'],axis=1,inplace=True)
titanic_data.head()


# In[ ]:


titanic_dmy = pd.concat([titanic_data, gender, embark_location],axis=1)
titanic_dmy.tail()


# In[ ]:


#Checking for correlation between variables.
sb.heatmap(titanic_dmy.corr(),square=True)
#print(titanic_dmy.corr())


# In[ ]:


X = titanic_dmy.ix[:,(1,2,3,4,5,6,7,8,9)].values
y = titanic_dmy.ix[:,0].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=2)


# The train test split is done for parameter tuning.
# We now deploy the models.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier

clf1 = SVC(kernel='linear',C=1.0,random_state=3)
clf2 = XGBClassifier(random_state=3)
clf3 = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=300)
eclf = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2),('clf3',clf3)], voting='hard')

eclf.fit(X_train, y_train)
y_pred = eclf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(eclf.score(X_test, y_test))


# **Now taking in Competition Data.**

# In[ ]:


url = '../input/test.csv'
test = pd.read_csv(url)
test.head()


# In[ ]:


test.isnull().sum()


# There are 86 null values in Age, so we approximate them like we did earlier.
# There are 327 null values in Cabin, so we drop it altogether.
# There is 1 null value in Fare, so we approximate it according to the median of each class of the null position.

# In[ ]:


test.describe()


# In[ ]:


sb.set(rc={'figure.figsize':(11.7,8.27)})
ax = sb.boxplot(x='Pclass',y='Fare',data=test,width=0.9)


# In[ ]:


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


# **Cleaning up the test data:**
# Dropping variables, approximating age and fare, dummy variables.

# In[ ]:


test_data = test.drop(['Name','Ticket'],1)
test_data['Age'] = test_data[['Age', 'Pclass']].apply(age_approx, axis=1)
test_data['Fare'] = test_data[['Fare','Pclass']].apply(fare_approx, axis=1)
test_data['Cabin'] = test_data[['Cabin','Pclass']].apply(cabin_approx, axis=1)
#
gender_test = pd.get_dummies(test_data['Sex'],drop_first=True)
embark_location_test = pd.get_dummies(test_data['Embarked'],drop_first=True)
test_data.drop(['Sex','Embarked'],axis=1,inplace=True)
test_dmy = pd.concat([test_data, gender_test, embark_location_test],axis=1)

#test_dmy.describe()
test_data.dropna(inplace=True)
test_dmy.isnull().sum()


# In[ ]:


test_dmy.head()


# In[ ]:


X_competition = test_dmy.ix[:,(1,2,3,4,5,6,7,8,9)].values


# **Prediction for Competition Data**

# In[ ]:


y_comp = eclf.predict(X_competition)


# In[ ]:


submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':y_comp})
submission.head()


# In[ ]:


filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




