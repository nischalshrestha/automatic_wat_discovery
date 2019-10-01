#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split

# Figures inline and set visualization style
get_ipython().magic(u'matplotlib inline')
sns.set()


# In[ ]:


os.listdir('../input')


# In[ ]:


# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


print(df_train.shape)
df_train.head()


# In[ ]:


print(df_test.shape)
df_test.head()


# In[ ]:


df = df_train.append(df_test, sort=False)
df.info()


# In[ ]:


# Dealing with missing numerical variables
df['Age'] = df.Age.fillna(df.Age.median())
df['Fare'] = df.Fare.fillna(df.Fare.median())
df.info()


# In[ ]:


#df = pd.get_dummies(df, columns=['Sex'], drop_first=True)


# In[ ]:


df['Surname'] = df['Name'].str.split(',').str[0]


# In[ ]:


df['Title'] = df['Name'].str.split(',').str[1].str.split().str[0]  


# In[ ]:


#df['Cabin Len'] = df.Cabin.str.split().str.len()


# In[ ]:


df['Cabin Letter'] = df['Cabin'].str[0]


# In[ ]:


df['Family_Size'] = df['SibSp'] + df['Parch']


# In[ ]:


df['Fare Per Person'] = df['Fare'] / (df['Family_Size'] + 1)


# In[ ]:


df['Number of Ticket Uses'] = df.groupby('Ticket', as_index=False)['Ticket'].transform(lambda s: s.count())


# In[ ]:


df['Average Fare per Person'] = df['Fare'] / df['Number of Ticket Uses'] 


# In[ ]:


for col in df.columns:  
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')  # change text to category
        df[col] = df[col].cat.codes  # save code as column value


# In[ ]:


# RandomForest/Decision Tree it is interesting to replace NA by a value less then the minimum or greater then the maximum
#df.fillna(-1, inplace=True)


# In[ ]:


data_train = df.iloc[:891].copy()
data_test = df.iloc[891:].copy()


# In[ ]:


train, test = train_test_split(data_train, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_features=.5, random_state=42)


# In[ ]:


remove = ['Survived', 'PassengerId', 'Name', 'Cabin', 'Embarked']
feats = [col for col in df.columns if col not in remove]


# In[ ]:


rf.fit(train[feats], train['Survived'])


# In[ ]:


preds_train = rf.predict(train[feats])


# In[ ]:


preds = rf.predict(test[feats])


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(train['Survived'], preds_train)


# In[ ]:


accuracy_score(test['Survived'], preds)


# In[ ]:


rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=.5, random_state=42)


# In[ ]:


# train with training and test dataset
rf.fit(data_train[feats],data_train['Survived'])


# In[ ]:


preds_kaggle = rf.predict(data_test[feats])


# In[ ]:


submission = pd.DataFrame({ 'PassengerId': data_test['PassengerId'],
                            'Survived': preds_kaggle }, dtype=int)
submission.to_csv("submission.csv",index=False)


# In[ ]:




