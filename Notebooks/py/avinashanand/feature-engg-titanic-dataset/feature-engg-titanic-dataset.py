#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,OneHotEncoder,LabelEncoder
import seaborn as sns
from time import strftime,gmtime
import re


# In[ ]:


df_train = pd.read_csv('../input/titanic/train.csv')
df_gs = pd.read_csv('../input/titanic/gender_submission.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
df_train_tmp = pd.read_csv('../input/dataset2/train_temp.csv')
df_test_tmp = pd.read_csv('../input/dataset2/test_tmp.csv')


# In[ ]:


df_train_tmp.drop('Ticket',axis=1,inplace=True)
df_test_tmp.drop('Ticket',axis=1,inplace=True)


# In[ ]:


print('Test shape {}'.format(df_test.shape))
print('Train shape {}'.format(df_train.shape))


# In[ ]:


y = df_train['Survived']
df_train = df_train.merge(df_train_tmp,on='PassengerId',how='left')
df_test = df_test.merge(df_test_tmp,on='PassengerId',how='left')
del df_train_tmp
del df_test_tmp
df_train.drop(['Survived','PassengerId','Ticket'],axis=1,inplace=True)
df_test.drop(['PassengerId','Ticket'],axis=1,inplace=True)
print(df_train.info())
print(df_test.info())


# In[ ]:


def null_graph(X_train,X_test,show_graph=False):
#     print(X_train.isna().sum())
#     print(X_test.isna().sum())
    if show_graph:
        plt.bar(range(len(X_train.columns)),X_train.isna().sum().values)
        plt.bar(range(len(X_test.columns)),X_test.isna().sum().values)
        plt.ylabel('Count')
        plt.xlabel('Columns')
        plt.title('Count of null vs columns')
        plt.xticks
        plt.legend(('Train', 'Test'))
        plt.show()


# In[ ]:


null_graph(df_train,df_test,show_graph=True)


# In[ ]:


df_train['Embarked'].fillna('S',inplace=True)
df_test['Embarked'].fillna('S',inplace=True)


# In[ ]:


def age_feature_engg(X):
    age_avg = X['Age'].mean()
    age_std = X['Age'].std()
    age_null_count = X['Age'].isna().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    X.loc[np.isnan(X['Age']), 'Age'] = age_null_random_list
    X['Age'] = X['Age'].astype(int)
    X.loc[ X['Age'] <= 16, 'Age'] 					 = 0
    X.loc[(X['Age'] > 16) & (X['Age'] <= 32), 'Age'] = 1
    X.loc[(X['Age'] > 32) & (X['Age'] <= 48), 'Age'] = 2
    X.loc[(X['Age'] > 48) & (X['Age'] <= 64), 'Age'] = 3
    X.loc[ X['Age'] > 64, 'Age'] ;


# In[ ]:


age_feature_engg(df_train)
age_feature_engg(df_test)


# In[ ]:


def fare_feature_engg(X):
    X['Fare'].fillna(X['Fare'].median(),inplace=True)
    X.loc[ X['Fare'] <= 7.91, 'Fare'] 						        = 0
    X.loc[(X['Fare'] > 7.91) & (X['Fare'] <= 14.454), 'Fare'] = 1
    X.loc[(X['Fare'] > 14.454) & (X['Fare'] <= 31), 'Fare']   = 2
    X.loc[ X['Fare'] > 31, 'Fare'] 							        = 3
    X['Fare'] = X['Fare'].astype(int)
    X['Fare'].fillna(df_train['Fare'].median(),inplace=True)


# In[ ]:


fare_feature_engg(df_train)
fare_feature_engg(df_test)


# In[ ]:


def name_title_feature_engg(X):
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""
    X
    X['Title'] = X['Name'].apply(get_title)
    X['Title'] = X['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    X['Title'] = X['Title'].replace('Mlle', 'Miss')
    X['Title'] = X['Title'].replace('Ms', 'Miss')
    X['Title'] = X['Title'].replace('Mme', 'Mrs')
    X['Title'] = LabelEncoder().fit_transform(X['Title'])
    X['Name_length'] = X['Name'].apply(len)
    X.drop('Name',axis=1,inplace=True)


# In[ ]:


name_title_feature_engg(df_train)
name_title_feature_engg(df_test)


# In[ ]:


def family_feature_engg(X):
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    X.drop('SibSp',axis=1,inplace=True)


# In[ ]:


family_feature_engg(df_train)
family_feature_engg(df_test)


# In[ ]:


def cabin_feature_engg(X):  
    arr_temp = np.zeros(len(X))
    arr_temp[X.Cabin[(X.Cabin.isna())].index] = 1
    df_temp = pd.DataFrame(arr_temp,columns=['Cabin_null'])
    X = pd.concat([X,df_temp],axis=1,ignore_index=False)
    X.Cabin.fillna(-1,inplace=True)
    X.Cabin.fillna(-1,inplace=True)
    arr = {'A':[],'B':[],'C':[],'D':[],'E':[],'F':[],'G':[],'H':[]}
    for row in range(0,len(X)):
        if X.loc[row,'Cabin'] != -1:
            if X.loc[row,'Cabin'].startswith('A'):
                arr['A'].append(row)
            elif X.loc[row,'Cabin'].startswith('B'):
                arr['B'].append(row)
            elif X.loc[row,'Cabin'].startswith('C'):
                arr['C'].append(row)
            elif X.loc[row,'Cabin'].startswith('D'):
                arr['D'].append(row)
            elif X.loc[row,'Cabin'].startswith('E'):
                arr['E'].append(row)
            elif X.loc[row,'Cabin'].startswith('F'):
                arr['F'].append(row)
            elif X.loc[row,'Cabin'].startswith('G'):
                arr['G'].append(row)
            elif X.loc[row,'Cabin'].startswith('T'):
                arr['H'].append(row)
    for key in arr.keys():
        X['Cabin_{}'.format(key)] = 0
#         for row in arr[key]:
        X.loc[arr[key],'Cabin_{}'.format(key)] = 1
    X.drop('Cabin',inplace=True,axis=1)
    return X


# In[ ]:


df_train = cabin_feature_engg(df_train)
df_test = cabin_feature_engg(df_test)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.info()


# In[ ]:


def one_hot_encoder(X,usecol,columns):
    le = LabelEncoder()
    integer_encoded = le.fit_transform(X[usecol])
    ohe = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(X), 1)
    integer_encoded = ohe.fit_transform(integer_encoded)
    df_ohe_tmp = pd.DataFrame(integer_encoded,columns=columns)
    X = pd.concat([X,df_ohe_tmp],axis=1)
    X.drop(usecol,axis=1,inplace=True)
    return X


# In[ ]:


cols_Embarked = ['Embarked_C','Embarked_Q','Embarked_S']
cols_Sex = ['Male','Female']
cols_Pclass = ['Pclass_1','Pclass_2','Pclass_3']
df_train = one_hot_encoder(df_train,'Embarked',cols_Embarked)
df_train = one_hot_encoder(df_train,'Sex',cols_Sex)
df_train = one_hot_encoder(df_train,'Pclass',cols_Pclass)
df_test = one_hot_encoder(df_test,'Embarked',cols_Embarked)
df_test = one_hot_encoder(df_test,'Sex',cols_Sex)
df_test = one_hot_encoder(df_test,'Pclass',cols_Pclass)


# In[ ]:


df_train.fillna(0,axis=1,inplace=True)
df_test.fillna(0,axis=1,inplace=True)


# In[ ]:


df_train


# In[ ]:


#null_graph(df_train,df_test,show_graph=True)


# In[ ]:


# df_test.to_csv('Dataset/feature_engg_test.csv',index=False)
# pd.concat([df_train,y],axis=1).to_csv('Dataset/feature_engg_train.csv',index=False)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# In[ ]:


for col in df_train.columns:
    print("{} has {} unique values".format(col,df_train[col].nunique()))


# In[ ]:


class CategoryTransformer(TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        for col in X.columns:
            if X[col].dtype.name == 'object':
                X[col] = X[col].astype('category')
                X[col] = X[col].cat.codes
        return X


# In[ ]:


class NullTransformer(TransformerMixin):
    def transform(self,X):
        imputer = Imputer(axis=1)
        X = pd.DataFrame(imputer.fit_transform(X),columns=X.columns)
        return X


# In[ ]:


df_train = df_train[['Age', 'Parch', 'Fare', 'Title', 'Name_length', 'FamilySize',
       'Cabin_null', 'Cabin_B', 'Embarked_C', 'Embarked_S', 'Male', 'Female',
       'Pclass_1', 'Pclass_2', 'Pclass_3']]
df_test = df_test[['Age', 'Parch', 'Fare', 'Title', 'Name_length', 'FamilySize',
       'Cabin_null', 'Cabin_B', 'Embarked_C', 'Embarked_S', 'Male', 'Female',
       'Pclass_1', 'Pclass_2', 'Pclass_3']]


# In[ ]:


kfold = StratifiedKFold(n_splits=10,random_state=42)
fig, ax = plt.subplots(5,2,figsize=(10,30))
for i,(train_idx,valid_idx) in enumerate(kfold.split(df_train,y)):
    X_train,y_train = df_train.iloc[train_idx],y[train_idx]
    X_valid,y_valid = df_train.iloc[valid_idx],y[valid_idx]
    lr = LogisticRegression(penalty='l1',random_state=42)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_valid)
#     print(accuracy_score(y_pred=y_pred,y_true=y_valid))
    row = i//2
    col = i%2
    ax[row,col].plot(df_train.columns,lr.coef_.reshape(len(df_train.columns),1))
    ax[row,col]
    ax[row,col].set_title(accuracy_score(y_pred=y_pred,y_true=y_valid))
plt.show()


# In[ ]:


df_train.info()


# In[ ]:


kfold = StratifiedKFold(n_splits=10,random_state=42)
fig, ax = plt.subplots(5,2,figsize=(20,40))
max_accr = 0
for i,(train_idx,valid_idx) in enumerate(kfold.split(df_train,y)):
    X_train,y_train = df_train.iloc[train_idx],y[train_idx]
    X_valid,y_valid = df_train.iloc[valid_idx],y[valid_idx]
    rf = ExtraTreesClassifier(20,min_samples_split=10,random_state=23,max_leaf_nodes=30)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_valid)
    accr = accuracy_score(y_pred=y_pred,y_true=y_valid)
    print(accr)
    row = i//2
    col = i%2
    if(accr>0):
        max_accr = accr
        model = rf
    ax[row,col].bar(range(0,len(rf.feature_importances_)),rf.feature_importances_)
    ax[row,col].set_xticklabels(df_train.columns)
    ax[row,col].set_xlim(0, len(rf.feature_importances_))
    ax[row,col].set_title(accuracy_score(y_pred=y_pred,y_true=y_valid))
plt.show()


# In[ ]:


df_train.columns[(model.feature_importances_>0.015)]


# In[ ]:


df_train.columns


# In[ ]:


y_pred = model.predict(df_test)


# In[ ]:


df_gs['Survived'] = y_pred


# In[ ]:


df_gs['Survived'].value_counts()


# In[ ]:


curr_time = strftime("%Y-%m-%d-%H-%M-%S")
df_gs.to_csv('output{}.csv'.format(curr_time),index=False)

