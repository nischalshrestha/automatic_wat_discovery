#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


df_gender = pd.read_csv('../input/gender_submission.csv')
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train.shape, df_test.shape, df_gender.shape


# ## Data Overview

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.columns


# In[ ]:


df_gender.head()


# ### Feature engineering on travelling class (Pclass)

# In[ ]:


le_pclass = preprocessing.LabelEncoder()
le_pclass.fit(pd.concat([df_train['Pclass'], df_test['Pclass']], ignore_index=True))
df_train['PclassEncoded'] = le_pclass.transform(df_train['Pclass'])
df_test['PclassEncoded'] = le_pclass.transform(df_test['Pclass'])
df_train.head()


# ### Feature engineering on passenger names

# In[ ]:


name_parts = df_train['Name'].str.split(',', n=1, expand=True)
df_train['LastName'] = name_parts[0]
df_train['FirstName'] = name_parts[1]
df_train.drop(['Name'], axis=1, inplace=True)

name_parts = df_test['Name'].str.split(',', n=1, expand=True)
df_test['LastName'] = name_parts[0]
df_test['FirstName'] = name_parts[1]
df_test.drop(['Name'], axis=1, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


df_train['LastName'].unique().shape, df_test['LastName'].unique().shape


# Lets have a look at the last name of passengers regarding the survival count

# In[ ]:


df_train[df_train['Survived'] == 1].groupby('LastName').agg({'Survived': 'count'}).max()


# In[ ]:


df_train[df_train['Survived'] == 0].groupby('LastName').agg({'Survived': 'count'}).max()


# Convert last names to categories

# In[ ]:


le_lastname = preprocessing.LabelEncoder()
le_lastname.fit(pd.concat([df_train['LastName'], df_test['LastName']], ignore_index=True))
df_train['LastNameEncoded'] = le_lastname.transform(df_train['LastName'])
df_test['LastNameEncoded'] = le_lastname.transform(df_test['LastName'])
df_train.head()


# Get the titles from first name

# In[ ]:


name_parts = df_train['FirstName'].str.strip().str.split(' ', n=1, expand=True)
df_train['Title'] = name_parts[0].str.strip()

name_parts = df_test['FirstName'].str.strip().str.split(' ', n=1, expand=True)
df_test['Title'] = name_parts[0].str.strip()

df_train.head()


# In[ ]:


df_train['Title'].unique()


# In[ ]:


df_train[df_train['Survived'] == 1].groupby('Title').agg({'Survived': 'count'}).max()


# In[ ]:


df_train[df_train['Survived'] == 0].groupby('Title').agg({'Survived': 'count'}).max()


# Convert the Titls into category

# In[ ]:


le_title = preprocessing.LabelEncoder()
le_title.fit(pd.concat([df_train['Title'], df_test['Title']], ignore_index=True))
df_train['TitleEncoded'] = le_title.transform(df_train['Title'])
df_test['TitleEncoded'] = le_title.transform(df_test['Title'])
df_train.head()


# ### Feature engineering on Sex

# In[ ]:


le_sex = preprocessing.LabelEncoder()
le_sex.fit(df_train['Sex'])
df_train['SexEncoded'] = le_sex.transform(df_train['Sex'])
df_test['SexEncoded'] = le_sex.transform(df_test['Sex'])
df_train.head()


# ### Feature engineering on Age
# There are many missing data in age.

# In[ ]:


df_train['Age'].isnull().sum(), df_test['Age'].isnull().sum()


# In[ ]:


df_train['Age'].max(), df_train['Age'].min(), df_test['Age'].max(), df_test['Age'].min()


# In[ ]:


df_train['Age'].fillna(0, inplace=True)
df_test['Age'].fillna(0, inplace=True)
df_train['Age'].max(), df_train['Age'].min(), df_test['Age'].max(), df_test['Age'].min()


# In[ ]:


df_train['AgeEncoded'] = pd.cut(df_train['Age'], 5, labels=[1, 2, 3, 4, 5])
df_test['AgeEncoded'] = pd.cut(df_test['Age'], 5, labels=[1, 2, 3, 4, 5])


# In[ ]:


df_train['AgeEncoded'].max(), df_train['AgeEncoded'].min(), df_test['AgeEncoded'].max(), df_test['AgeEncoded'].min()


# Lets see if we still have NaNs in Age

# In[ ]:


df_train['Age'].isnull().sum(), df_test['Age'].isnull().sum()


# ### Feature engineering on SibSp and Parch

# In[ ]:


df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_train.loc[df_train['FamilySize'] == 0, 'FamilySize'] = 1

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']
df_test.loc[df_test['FamilySize'] == 0, 'FamilySize'] = 1
df_train.head()


# ### Feature engineering on Ticket

# In[ ]:


ticket_parts = df_train['Ticket'].str.strip().str.split(' ', n=1, expand=True)
df_train['TicketEncoded'] = ticket_parts[0]
df_train['TicketEncoded'] = df_train['TicketEncoded'].apply(lambda x: 'Unknown' if x.isnumeric() else x)

ticket_parts = df_test['Ticket'].str.strip().str.split(' ', n=1, expand=True)
df_test['TicketEncoded'] = ticket_parts[0]
df_test['TicketEncoded'] = df_test['TicketEncoded'].apply(lambda x: 'Unknown' if x.isnumeric() else x)

le_ticket = preprocessing.LabelEncoder()
le_ticket = le_ticket.fit(pd.concat([df_train['TicketEncoded'], df_test['TicketEncoded']], ignore_index=True))
df_train['TicketEncoded'] = le_ticket.transform(df_train['TicketEncoded'])
df_test['TicketEncoded'] = le_ticket.transform(df_test['TicketEncoded'])
df_train.head()


# In[ ]:


df_train[df_train['Survived'] == 1].groupby('TicketEncoded').agg({'Survived': 'count'}).max()


# In[ ]:


df_train[df_train['Survived'] == 0].groupby('TicketEncoded').agg({'Survived': 'count'}).max()


# ### Feature engineering on Fare

# In[ ]:


df_train['FareAvg'] = df_train['Fare'] / df_train['FamilySize']
df_test['FareAvg'] = df_test['Fare'] / df_test['FamilySize']


# In[ ]:


df_train['Fare'].max(), df_test['Fare'].max()


# In[ ]:


# Now binning the FareAvg
df_train['FareAvg'] = pd.cut(df_train['FareAvg'], 5, labels=[1, 2, 3, 4, 5])
df_test['FareAvg'] = pd.cut(df_test['FareAvg'], 5, labels=[1, 2, 3, 4, 5])


# In[ ]:


df_train['FareAvg'].max(), df_test['FareAvg'].max()


# ### Feature engineering on Cabin
# We see many visible NaN values in Cabin column. Lets see how many NaN are there in Cabin columns.

# In[ ]:


df_train['Cabin'].isnull().sum(), df_test['Cabin'].isnull().sum()


# Lets have a look at the values for Cabin just to check if using Cabin as feature will be useful or not.

# In[ ]:


df_train['Cabin'].unique(), df_test['Cabin'].unique()


# Find the number of people survived who has `Cabin` assigned to them vs. people survived without `Cabin` assigned.

# In[ ]:


len(df_train['Cabin'].unique()), len(df_test['Cabin'].unique())


# In[ ]:


survived_without_cabin = len(df_train[df_train['Cabin'].isnull() & df_train['Survived'] == 1])
not_survived_without_cabin = len(df_train[df_train['Cabin'].isnull() & df_train['Survived'] == 0])

survived_with_cabin = len(df_train[df_train['Cabin'].notnull() & df_train['Survived'] == 1])
not_survived_with_cabin = len(df_train[df_train['Cabin'].notnull() & df_train['Survived'] == 0])

print(survived_without_cabin, not_survived_without_cabin, survived_with_cabin, not_survived_with_cabin)

pd.DataFrame({
    'without cabin': [survived_without_cabin, not_survived_without_cabin],
    'with cabin': [survived_with_cabin, not_survived_with_cabin]
}, index=['survived', 'not survived']).plot.bar(rot=0)


# In[ ]:


df_train[df_train['Survived'] == 0].groupby('Cabin').agg({'Survived': 'count'}).max()


# In[ ]:


df_train[df_train['Survived'] == 1].groupby('Cabin').agg({'Survived': 'count'}).max()


# In[ ]:


df_train['Cabin'].fillna('Unknown', inplace=True)
df_test['Cabin'].fillna('Unknown', inplace=True)

le_cabin = preprocessing.LabelEncoder()
le_cabin.fit(pd.concat([df_train['Cabin'], df_test['Cabin']], ignore_index=True))
df_train['CabinEncoded'] = le_cabin.transform(df_train['Cabin'])
df_test['CabinEncoded'] = le_cabin.transform(df_test['Cabin'])
df_train.head()


# Definitely its worth to use Cabin info

# ### Feature engineering on Embarked

# In[ ]:


df_train['Embarked'].fillna('Unknown', inplace=True)
df_test['Embarked'].fillna('Unknown', inplace=True)

le_embarked = preprocessing.LabelEncoder()
le_embarked.fit(pd.concat([df_train['Embarked'], df_test['Embarked']], ignore_index=True))
df_train['EmbarkedEncoded'] = le_embarked.transform(df_train['Embarked'])
df_test['EmbarkedEncoded'] = le_embarked.transform(df_test['Embarked'])
df_train.head()


# In[ ]:


df_train.columns


# In[ ]:


df_test.columns


# In[ ]:


columns_train = ['AgeEncoded', 'SibSp', 'Parch',
       'PclassEncoded', 'LastNameEncoded', 'TitleEncoded', 'SexEncoded',
       'FamilySize', 'TicketEncoded', 'FareAvg', 'CabinEncoded', 'EmbarkedEncoded']


# In[ ]:


df_train.isnull().sum(), df_test.isnull().sum()


# Still have one NaN in Fare and FareAvg. Just replace it with 0.

# In[ ]:


df_train.fillna(1, inplace=True)
df_test.fillna(1, inplace=True)

df_train.isnull().sum(), df_test.isnull().sum()


# In[ ]:


X = df_train[columns_train]
y = df_train['Survived']
X_pred = df_test[columns_train]
X.head()


# In[ ]:


scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)
X_pred = scaler.transform(X_pred)


# In[ ]:


X.shape, y.shape, X_pred.shape


# ### Preparing for training

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ### Predict with Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
# y_pred = model.predict(X_test)
# score = log_loss(y_test, predict_proba)

# result_df = pd.DataFrame(data={'PassengerId': X_test.index, 'Survived': y_predict})
# result_df.columns = ['PassengerId', 'Survived']

# print(result_df.head(100))

# plot_feature_imp(model, X_test)

# print(roc_auc_score(y_pred, predict_proba))

# file = open('submission_rf.csv', 'w')
# file.write(result_df.to_csv(index=False))
# file.close()

# result_df.head()
print(score)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto')
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)
# y_pred = clf.predict(X_pred)

# result_df = pd.DataFrame(data={'PassengerId': df_test['PassengerId'], 'Survived': y_pred})
# result_df.columns = ['PassengerId', 'Survived']

# print(result_df.head(20))

# # print(roc_auc_score(y_pred, predict_proba))

# file = open('submission_rf.csv', 'w')
# file.write(result_df.to_csv(index=False))
# file.close()

# # result_df.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

y_pred = clf.predict(X_pred)
result_df = pd.DataFrame(data={'PassengerId': df_test['PassengerId'], 'Survived': y_pred})

file = open('submission_rfc.csv', 'w')
file.write(result_df.to_csv(index=False))
file.close()

# print(clf)
print(score)
result_df.head()


# In[ ]:


from sklearn.svm import SVC
clf = SVC(kernel='linear', C=0.0001, probability=True)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)


# ### Predict with Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
score = clf.score(X_test, y_test)
# print(clf)
print(score)


# In[ ]:





# In[ ]:





# In[ ]:




