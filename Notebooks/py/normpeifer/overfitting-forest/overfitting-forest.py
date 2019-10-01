#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2


# In[ ]:


# Import the data and separate the target 

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
y = train_df['Survived'].values
identity = test_df["PassengerId"]
train_df = train_df.drop(["Survived"], axis=1)

train_df.head()


# In[ ]:


# Find the missing data

def findna(df, colname):
    name = [x for x in globals() if globals()[x] is df][0]
    fnl = df.loc[df[colname].isnull(), colname]
    if (fnl.shape[0] != 0):
        print(name, "/", colname, ":  ", fnl.shape)


for df in train_df, test_df:
    for colname in df.columns.values:
        findna(df, colname)


# In[ ]:


# Use the most commun value for missing embarked

fillembark = train_df["Embarked"].mode()[0]
train_df["Embarked"] = train_df["Embarked"].fillna(fillembark)


# In[ ]:


'''
1. Mark people with missing data for Age or Cabin
2. Extract regular informations from Name, Ticket and Cabin.
'N' will mark missing data and 'R' or 'Rare' unusual ones
'''

for df in [train_df, test_df]:
        df["WithAge"] = 1
        df["WithCabin"] = 1
        df.loc[df['Age'].isnull(), 'WithAge'] = 0
        df.loc[df['Cabin'].isnull(), 'WithCabin'] = 0
        df["Title"] = df.Name.str.extract(' ([A-Za-z]+)\.',
                                          expand=False)
        df["Title"] = df["Title"].replace('Mlle', 'Miss')
        df["Title"] = df["Title"].replace('Ms', 'Miss')
        df["Title"] = df["Title"].replace('Mme', 'Mrs')
        df.loc[~df.Title.isin(['Mr', 'Miss', 'Master',
                               'Mrs']), 'Title'] = 'Rare'
        df["TyTicket"] = df.Ticket.str.extract('([A-Za-z])', expand=False)
        df["TyCabin"] = df.Cabin.str.extract('([A-Za-z])', expand=False)
        df.loc[df['TyTicket'].isnull(), 'TyTicket'] = 'N'
        df.loc[~df.TyTicket.isin(['S', 'P', 'C', 'N']), 'TyTicket'] = 'R'
        df.loc[df['TyCabin'].isnull(), 'TyCabin'] = 'N'
        df.loc[~df.TyCabin.isin(['C', 'B', 'N']), 'TyCabin'] = 'R'


# In[ ]:


'''
Fill missing age and fare values based on Sex and Class
Median are obtain with x_all that contains both training and test data
'''

x_all = pd.concat([train_df, test_df], axis=0)

for clss in list(x_all["Pclass"].unique()):
    for gender in list(x_all["Sex"].unique()):
        stdfare = x_all.loc[(x_all["Pclass"] == clss) &
                            (x_all["Sex"] == gender),
                            'Fare'].median()
        stdage = x_all.loc[(x_all["Pclass"] == clss) &
                           (x_all["Sex"] == gender),
                           'Age'].median()
        
        for df in [train_df, test_df]:
            df.loc[(df["Pclass"] == clss) &
                   (df["Sex"] == gender),
                   'Age'] = df.loc[(df["Pclass"] == clss) &
                                   (df["Sex"] == gender),
                                   'Age'].fillna(stdage)
            df.loc[(df["Pclass"] == clss) &
                   (df["Sex"] == gender),
                   'Fare'] = df.loc[(df["Pclass"] == clss) &
                                    (df["Sex"] == gender),
                                    'Fare'].fillna(stdfare)
            
        print("Class ", clss, "  &  ", "Gender ", gender, "Fare ", stdfare,
              "Age", stdage)


# In[ ]:


# Drop "useless" columns

train_df = train_df.drop(["Cabin", "PassengerId",
                          "Ticket", "Name"], axis=1)
test_df = test_df.drop(["Cabin", "PassengerId", "Ticket",
                        "Name"], axis=1)
train_df.head()


# In[ ]:


# Encode categorical values with LabelEncoder

lbl_enc = preprocessing.LabelEncoder()
for col in ["Title", "TyTicket", "TyCabin", "Sex", "Embarked"]:
    lbl_enc.fit(train_df[col])
    train_df[col] = lbl_enc.transform(train_df[col])
    test_df[col] = lbl_enc.transform(test_df[col])


# In[ ]:


# Use OneHotEncoder on non binary categorical values

hot = preprocessing.OneHotEncoder(categorical_features=[True, False, False,
                                                        False, False, False,
                                                        True, False, False,
                                                        True, True, True],
                                  sparse=False)

hot.fit(np.concatenate((train_df, test_df)))
train_df = hot.transform(train_df)
test_df = hot.transform(test_df)


# Prepare for feature selection (22 seemed to work well)

skb = SelectKBest(chi2, k = 22)


# In[ ]:


# Define the number of validation run and the data frame with the results

nbloop = 5
accuracy_df = pd.DataFrame({'BaseSv': np.zeros(nbloop),
                            'Param1': np.zeros(nbloop),
                            'Param2': np.zeros(nbloop)})

nbseed = np.random.randint(1, 100)


# In[ ]:


'''
For Each validation run:
    1. Split data with train_test_split 
    2. Select best features with SelectKBest and chi2
    3. Perfrom cross validation with GridSearchCV to select best 
       hyperparameters of random forest
'''

for k in range(nbloop):
    
    x_fit, x_val, y_fit, y_val = train_test_split(train_df, y,
                                                  test_size=0.3,
                                                  stratify=y,
                                                  random_state=k + nbseed)
    
    skb.fit(x_fit, y_fit)
    x_fit = skb.transform(x_fit)
    x_val = skb.transform(x_val)   
    
    param_grid = {'n_estimators': [90, 120],
                  'max_depth': [11, 15],
                  'min_samples_split': [10],
                  'min_samples_leaf': [1],
                  'max_features': ['log2']}
    
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, param_grid, cv=5)
    
    
    clf.fit(x_fit, y_fit)
    y_pred = clf.predict(x_val)
    acc = accuracy_score(y_pred, y_val)
    accuracy_df.loc[k, 'BaseSv'] = acc
    accuracy_df.loc[k, 'Param1'] = clf.best_params_['n_estimators']
    accuracy_df.loc[k, 'Param2'] = clf.best_params_['max_depth']


# In[ ]:


# Check results

print(accuracy_df)
print("\n")
print(accuracy_df.describe())
print("\n END")

