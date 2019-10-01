#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train_id = train['PassengerId']
test_id = test['PassengerId']


# In[ ]:


train.drop('PassengerId', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)


# In[ ]:


y = train['Survived']


# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]


# In[ ]:


all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop('Survived', axis=1, inplace=True)


# In[ ]:


all_data.shape


# In[ ]:


all_data.head()


# In[ ]:


all_data.isnull().sum().sort_values(ascending=False)/len(all_data)


# In[ ]:


missing_df = pd.DataFrame(all_data.isnull().sum().sort_values(ascending=False)/len(all_data))
missing_df.reset_index(inplace=True)
missing_df.columns = ['Feature', 'Missing Value Ratio']


# In[ ]:


missing_df.head()


# In[ ]:


all_data.Cabin.fillna('U', inplace=True)


# In[ ]:


age_df = all_data.groupby('Sex').mean().loc[:, 'Age']


# In[ ]:


age_df


# In[ ]:


mean_age_male = age_df['male']


# In[ ]:


mean_age_female = age_df['female']


# In[ ]:


all_data.loc[(all_data['Age'].isnull()) & (all_data['Sex'] == 'male'), 'Age'] = mean_age_male


# In[ ]:


all_data.loc[(all_data['Age'].isnull()) & (all_data['Sex'] == 'female'), 'Age'] = mean_age_female


# In[ ]:


all_data.Embarked.fillna(all_data.Embarked.value_counts()[0], inplace=True)


# In[ ]:


all_data.Fare.fillna(all_data.Fare.mean(), inplace=True)


# In[ ]:


all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1


# In[ ]:


all_data.head()


# In[ ]:


all_data['IsAlone'] = 0
all_data.loc[all_data['FamilySize'] == 1, 'IsAlone'] = 1


# In[ ]:


all_data['CategoricalFare'] = (pd.qcut(all_data['Fare'], 5, labels = [0, 1, 2, 3, 4])).astype(int)


# In[ ]:


# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# In[ ]:


all_data['Title'] = all_data['Name'].apply(get_title)


# In[ ]:


all_data.head()


# In[ ]:


cols_to_be_dropped = ['Name', 'Parch', 'SibSp', 'FamilySize','Ticket', 'Fare']
all_data.drop(cols_to_be_dropped, axis=1, inplace=True)


# In[ ]:


all_data.shape


# In[ ]:


all_data.dtypes


# In[ ]:


all_data = pd.get_dummies(all_data)


# In[ ]:


ntrain


# In[ ]:


x = all_data[:ntrain]


# In[ ]:


x_test = all_data[ntrain:]


# In[ ]:


from sklearn.cross_validation import StratifiedKFold
eval_size = 0.20
kf = StratifiedKFold(y, round(1./eval_size))
train_indices, valid_indices = next(iter(kf))

x_train, y_train = x.loc[train_indices], y.loc[train_indices]
x_valid, y_valid = x.loc[valid_indices], y.loc[valid_indices]


# In[ ]:


y_train.value_counts()[0]/y_train.value_counts()[1]


# In[ ]:


y_valid.value_counts()[0]/y_valid.value_counts()[1]


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    SVC(),
    KNeighborsClassifier(),
    GaussianProcessClassifier(),
    MLPClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier()]


# In[ ]:


# Logging for Visual Comparison
log_cols=["Classifier", "Training Accuracy", "Validation Accuracy"]
log = pd.DataFrame(columns=log_cols)


# In[ ]:


for clf in classifiers:
    print("="*30)
    
    clf_name = clf.__class__.__name__
    print(clf_name)

    clf.fit(x_train, y_train)
    
    #Training Accuracy
    y_train_pred = clf.predict(x_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    #Validation Accuracy
    y_valid_pred = clf.predict(x_valid)
    valid_acc = accuracy_score(y_valid, y_valid_pred)
    
    print("Validation Accuracy: {:.4%}".format(valid_acc))
    
    log_entry = pd.DataFrame([[clf_name, train_acc, valid_acc]], columns=log_cols)
    log = log.append(log_entry)


# In[ ]:


log.sort_values('Validation Accuracy', ascending=True).plot.barh(x='Classifier', y='Validation Accuracy', figsize=(16,7))


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


clf_1 = GradientBoostingClassifier()
param_grid_clf_1 = { 
           "n_estimators" : [9, 18, 27, 36, 45, 54, 63],
           "learning_rate" : [0.001, 0.01, 0.1],
           "min_samples_leaf" : [1, 2, 4, 6, 8, 10],
            "min_samples_split": range(200,1001,200)
            }
grid_search_1 = GridSearchCV(estimator=clf_1, cv=10, param_grid=param_grid_clf_1)
# grid_search_1.fit(x,y)

# results['GBC'] = grid_search_1.predict(x_test)


# In[ ]:


clf_2 = RandomForestClassifier()
param_grid_clf_2 = { 
           "n_estimators" : [9, 18, 27, 36, 45, 54, 63],
           "max_depth" : [1, 5, 10, 15, 20, 25, 30],
           "min_samples_leaf" : [1, 2, 4, 6, 8, 10]}
grid_search_2 = GridSearchCV(estimator=clf_2, cv=10, param_grid=param_grid_clf_2)
# grid_search_2.fit(x,y)

# results['RFC'] = grid_search_2.predict(x_test)


# In[ ]:


clf_3 = LogisticRegression()
# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
param_grid_clf_3 = dict(C=C, penalty=penalty)
grid_search_3 = GridSearchCV(estimator=clf_3, cv=10, param_grid=param_grid_clf_3)
# grid_search_3.fit(x,y)

#results['LRC'] = grid_search_3.predict(x_test)


# In[ ]:


from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[('lr', grid_search_1), ('rf', grid_search_2), ('gnb', grid_search_3)], voting='hard')
eclf1.fit(x,y)


# In[ ]:


y_predict = eclf1.predict(x_test)


# In[ ]:


sub = pd.DataFrame()
sub['PassengerId'] = test_id
sub['Survived'] = y_predict
sub.to_csv('submission.csv',index=False)


# Feature Engineering Credits : https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
