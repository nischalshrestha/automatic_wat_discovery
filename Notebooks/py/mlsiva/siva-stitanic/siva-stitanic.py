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


# In[ ]:


inputds=pd.read_csv('../input/train.csv')


# In[ ]:


inputds.fillna(inputds.mean(),inplace=True)


# In[ ]:


inputds.fillna('S',inplace=True)


# In[ ]:


#removing the column Cabin as it has more than 60% Null values
inputds=inputds.drop('Cabin',axis=1)


# In[ ]:


inputds=inputds.drop('Name',axis=1)


# In[ ]:


inputds[inputds['PassengerId']==62]


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le=LabelEncoder()
#we can use one hot encoding when we want to encode for multiple columns at once
#but One hot encoding wont work when the in case of many categories in a same column(Eg:happy, extremely happy, okay,poor)


# In[ ]:


inputds['Sex']=le.fit_transform(inputds['Sex'])


# In[ ]:


inputds['Embarked']=le.fit_transform(inputds['Embarked'])


# In[ ]:


inputds['Ticket']=le.fit_transform(inputds['Ticket'])


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


etc=ExtraTreesClassifier()


# In[ ]:


y=inputds['Survived'].tolist()


# In[ ]:


inputds1=inputds


# In[ ]:


inputds2=inputds1.drop('Survived',axis=1)


# In[ ]:


x=inputds2.values


# In[ ]:


etc.fit(x,y)
#inputds2.head


# In[ ]:


etc.feature_importances_


# In[ ]:


inputds2.columns


# In[ ]:


x1=inputds2.drop('Sex',axis=1).drop('Age',axis=1).drop('SibSp',axis=1).drop('Parch',axis=1).drop('Embarked',axis=1).values


# In[ ]:


# have add some code to visualize the feature selection output in a graph


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


mnb=MultinomialNB()


# In[ ]:


mnb.fit(inputds2,y)


# In[ ]:


testds=pd.read_csv('../input/test.csv')


# In[ ]:


testds.fillna(testds.mean(),inplace=True)


# In[ ]:


testds.fillna('S',inplace=True)


# In[ ]:


testds=testds.drop('Cabin',axis=1)


# In[ ]:


testds=testds.drop('Name',axis=1)


# In[ ]:


testds['Sex']=le.fit_transform(testds['Sex'])


# In[ ]:


inputds[inputds['PassengerId']==62]


# In[ ]:


testds['Embarked']=le.fit_transform(testds['Embarked'])


# In[ ]:


testds['Ticket']=le.fit_transform(testds['Ticket'])


# In[ ]:


#testds['Name']=le.fit_transform(testds['Name'])
testds


# In[ ]:


predictt=mnb.predict(testds)


# In[ ]:


finalseries=pd.Series(predictt)


# In[ ]:


finalseries.to_csv('finalprediction.csv')


# In[ ]:


round( mnb.score(x,y)* 100, 2)


# In[ ]:


y


# In[ ]:


testrst=pd.read_csv('../input/gender_submission.csv')


# In[ ]:


testrst1=testrst['Survived']


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(testrst1,predictt)


# In[ ]:


#inputds.select('PassengerId')


# In[ ]:


for i in predictt:
    print(predictt[i])


# In[ ]:


testd=testds.PassengerId.tolist()
testrst1=testrst.Survived.tolist()
#spam_df.head(10).text.tolist() + spam_df.tail(10).text.tolist()


# In[ ]:


i=0
for a,b in zip(testd,testrst1):
    print(str(testd[i][:50])+" actual value is ("+str(testrst1[i]) +") predicted value is "+str(predictt[i]))
    i+=1


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y, predictt)


# In[ ]:


finalseries


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


inputds.select[PassengerId]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




