#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df = pd.concat([df_train, df_test])
df.sample(5)


# In[ ]:


df.info()


# In[ ]:


title_dict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

df['title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip()).map(title_dict)
df.head()


# In[ ]:


grouped_age = df.groupby(['Sex', 'Pclass', 'title']).median().reset_index()[['Sex', 'Pclass', 'title', 'Age']]
grouped_age


# In[ ]:


def fill_age(row):
    condition = (
        (grouped_age['Sex'] == row['Sex']) & 
        (grouped_age['title'] == row['title']) & 
        (grouped_age['Pclass'] == row['Pclass'])
    ) 
    return grouped_age[condition]['Age'].values[0]


def process_age():
    df['Age'] = df.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    return df

df = process_age()
df['Age'].isnull().values.any()


# In[ ]:


df['Family_Size'] = df['Parch'] + df['SibSp'] + 1 # including the person in the row
df.head()


# In[ ]:


df['Last_Name'] = df['Name'].apply(lambda x: str.split(x, ",")[0])
df['Fare'].fillna(df['Fare'].mean(), inplace=True)


# In[ ]:


df['Family_Survival'] = 0.5

for grp, grp_df in df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      df.loc[df['Family_Survival']!=0.5].shape[0])


# In[ ]:


for _, grp_df in df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(df[df['Family_Survival']!=0.5].shape[0]))


# In[ ]:


df.sample(10)


# In[ ]:


df['Fare'].fillna(df['Fare'].median(), inplace = True)

df['FareBin'] = pd.qcut(df['Fare'], 5)

label = LabelEncoder()
df['FareBin_Code'] = label.fit_transform(df['FareBin'])


# In[ ]:


df['AgeBin'] = pd.qcut(df['Age'], 4)

label = LabelEncoder()
df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])


# In[ ]:


df['Sex'] = df['Sex'].map({'male':0, 'female':1})


# In[ ]:


df.head()


# In[ ]:


df.drop(['Age', 'Cabin', 'Embarked', 'Fare', 'Name', 'Parch', 'PassengerId', 'SibSp', 'Ticket', 'title', 'Last_Name', 'FareBin', 'AgeBin'], axis=1, inplace=True)
df.head()


# In[ ]:


df_train = df[:891]
df_test = df[891:]


# In[ ]:


X = df_train.drop('Survived', axis=1)
y = df_train['Survived']
X_test = df_test.copy()


# In[ ]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test.drop('Survived', axis=1, inplace=True)
X_test = scaler.transform(X_test)


# In[ ]:


knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 
                           weights='uniform')
knn.fit(X, y)
y_pred = knn.predict(X_test).astype(int)


# In[ ]:


temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = y_pred
temp.to_csv("../working/submission.csv", index = False)


# In[ ]:




