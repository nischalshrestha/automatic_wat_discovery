#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#data visualization
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Import data

# In[ ]:


traindf=pd.read_csv('../input/train.csv')
testdf=pd.read_csv('../input/test.csv')
dfgs=pd.read_csv('../input/gender_submission.csv')


# In[ ]:


traindf.head()


# In[ ]:


testdf.head()


# In[ ]:


dfgs.head()


# # EDA

# In[ ]:


traindf.info()


# In[ ]:


traindf.describe()


# In[ ]:


traindf.isnull().head()


# In[ ]:


sns.heatmap(traindf.isnull(),yticklabels=False,cmap='viridis',cbar=False)


# In[ ]:


sns.countplot(x='Survived',hue='Sex', data=traindf)


# In[ ]:


sns.countplot(x='Survived',hue='Pclass', data=traindf)


# In[ ]:


sns.countplot(x='Pclass',data=traindf)


# In[ ]:


sns.distplot(traindf['Age'].dropna(),kde=False,bins=40)
sns.set_style('whitegrid')


# In[ ]:


sns.countplot(x='SibSp',data=traindf)


# # Eliminate missing values

# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x='Pclass',y='Age',data=traindf)


# In[ ]:


traindf.groupby('Pclass').mean()['Age']


# In[ ]:


def get_age(cols):
    age=cols[0]
    pclass=cols[1]
    if pd.isnull(age):
        if pclass==1:
            return 38
        elif pclass==2:
            return 29
        else:
            return 25
    else:
        return age


# In[ ]:


traindf['Age']=traindf[['Age','Pclass']].apply(get_age,axis='columns')


# In[ ]:


sns.heatmap(traindf.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


traindf1=traindf.drop('Cabin',axis='columns')


# In[ ]:


traindf1.isnull().sum()


# In[ ]:


traindf1.dropna(inplace=True)


# In[ ]:


traindf1.isnull().sum()


# In[ ]:


traindf=traindf1


# In[ ]:


sns.heatmap(traindf.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# # Get dummies

# In[ ]:


traindf.head()


# In[ ]:


sexdummies=pd.get_dummies(traindf['Sex'],drop_first=True)
embarkeddummies=pd.get_dummies(traindf['Embarked'],drop_first=True)


# In[ ]:


traindf1=pd.concat([traindf,sexdummies,embarkeddummies],axis='columns')


# In[ ]:


traindf1.head(3)


# In[ ]:


traindf=traindf1.drop(['Name','Sex','Ticket','Embarked','PassengerId'],axis='columns')


# In[ ]:


traindf.head()


# # Model selection

# In[ ]:


X=traindf.drop('Survived',axis='columns')
y=traindf['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[ ]:


X_train.head(2)


# In[ ]:


y_train.head(3)


# In[ ]:


x_test.head()


# In[ ]:


y_test.head(3)


# # Training model

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model=LogisticRegression()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


prediction=model.predict(x_test)


# In[ ]:


prediction


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,prediction))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,prediction)

