#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# helper functions and variable initialization
train_path = '../input/train.csv'
test_path  = '../input/test.csv'
# function to read csv file
def fn_read(path):
    df = pd.read_csv(path)
    return df

# function to find total missing values and missing value percentage
def fn_miss_val(df):
    miss_tot = df.isnull().sum()
    miss_per = 100 * df.isnull().sum() / len(df)
    miss_tab = pd.concat([miss_tot, miss_per], axis=1)
    miss_tab = miss_tab.rename(columns={0: "Missing values", 1: "Missing value %"})
    return miss_tab

# group by function 
def fn_grouping(df, var1=None, mode='numeric'):
    if var1[0] != 'Age':
        df = df.groupby(var1[0])[var1[1]].agg(['sum', 'count']).reset_index()
    else:
        bins = [1, 10, 20, 30, 40, 50, 60, 70, 80]
        df = df.groupby(pd.cut(df['Age'], bins))['Survived'].agg(['sum', 'count']).reset_index()
    
    df = df.rename(columns={'sum': "survived", 'count': "Total_Passengers"})
    df['survived_per'] = df.apply(lambda x: fn_survival_per(x), axis=1)
        
    return df.sort_values('survived_per', ascending=False)

# function to calculate survival %
def fn_survival_per(df):
    return (df['survived'])* 100/df['Total_Passengers']

# function to plot kde
def fn_kde_plot(df, target, features):
    j = len(features)
    plt.figure(figsize=(15,12))
    for i, feature in enumerate(features):
        # create a new subplot for each source
        plt.subplot(j, 1, i + 1)
        sns.kdeplot(df.loc[df[target] == 0, feature], label = 'survived == 0')
        sns.kdeplot(df.loc[df[target] == 1, feature], label = 'survived == 1')
        plt.title('Distribution of %s by Survival Value' % feature)
        plt.xlabel('%s' % feature); plt.ylabel('Density');
        plt.legend();
    plt.tight_layout(h_pad = 2.5)

# function to plot confusion matrix
def fn_confusion(y_true, y_pred, score, model=None):
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    print(classification_report(y_true, y_pred))
    
# function to plot roc curve
def fn_plot_auc(roc_auc, model, X_test, y_true, model_name=None):
    fpr, tpr, thresholds = roc_curve(y_true, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='%s (area = %0.4f)' % (model_name, roc_auc))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    
# model function
def fn_model(X_train, y_train, X_test, y_true, model_name=None):
    if model_name == 'logistic':
        model = LogisticRegression()
    elif model_name == 'tree':
        model = tree.DecisionTreeClassifier(criterion='entropy') 
    elif model_name == 'randomforest':
        model= RandomForestClassifier(n_estimators=1000)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # plot confusion matrix using seaborn
    score = model.score(X_test , y_true)
    fn_confusion(y_true, y_pred, score, model)
    roc_auc = roc_auc_score(y_true, y_pred)
    # plot the roc curve
    fn_plot_auc(roc_auc, model, X_test, y_true, model_name)
    return  model


# In[ ]:


train = fn_read(train_path)
train.head()


# In[ ]:


train.info()


# In[ ]:


# cahnge the datatype of P Class to object (its a categorical variable)
train['Pclass'] = train['Pclass'].astype(object)
train.info()


# In[ ]:


miss_tab = fn_miss_val(train)
miss_tab
# age has 177 missing values, will use imputer to fill mean values


# In[ ]:


train_grp = fn_grouping(train,['Pclass', 'Survived'])
# Pclass 3 has a minimum survival %. There is a -ve correlation between survival and Pclass
train_grp


# In[ ]:


train_grp = fn_grouping(train,['SibSp', 'Survived'])
train_grp
# more the SibSp count lesser the rate of survival. 5, 8 can be outliers will see


# In[ ]:


train_grp = fn_grouping(train,['Age', 'Survived'])
train_grp
# survival rate is going down as the age of the individual increases
# the data still has 177 missing age values which we will impute and calculate the survival % again


# In[ ]:


train_grp = fn_grouping(train,['Parch', 'Survived'])
train_grp


# In[ ]:


train_grp = fn_grouping(train,['Embarked', 'Survived'])
train_grp


# In[ ]:


train_grp = fn_grouping(train,['Sex', 'Survived'])
train_grp
# females had more chance of survival


# In[ ]:


train_grp = fn_grouping(train,['Age', 'Survived'])
train_grp
# Chance of survival goes down with Age, exception 10-20, 20-30. There are still 177 NAN values


# In[ ]:


# kde plots for Age, SibSp, Parch and Pclass
fn_kde_plot(train,target='Survived', features=['Age','SibSp','Parch', 'Pclass','Fare'])


# In[ ]:


# describe the training set
train.describe()


# In[ ]:


# describe test set
test = fn_read(test_path)
test.describe()


# In[ ]:


# impute missing age values in test and train sets
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
train_X = train[['Age']]
test_X  = test[['Age']]
imputer.fit(train_X)
train['Age'] = imputer.transform(train_X)
test['Age'] = imputer.transform(test_X)
train.describe()


# In[ ]:


train_grp = fn_grouping(train,['Age', 'Survived'])
train_grp


# In[ ]:


train.info()


# In[ ]:


# check for collinearities
train_corr = train.corr()
train_corr
sns.heatmap(train_corr, annot=True)
# Features to include for modelling
# Age, Sibsp, Pclass (Fare and Pclass seems to be correlated), Parch, Sex. 


# In[ ]:


# one hot encoding for categorical variables
features = ['PassengerId','Survived','Pclass','Sex','Age','Parch']
#features = ['PassengerId','Survived','Pclass','Sex','Age']
train_X = pd.get_dummies(train[features])
#create test and train sets from the test set
train_X, test_X = train_test_split(train_X, test_size=0.2, random_state=10)
train_y1 = train_X['Survived']
test_y1  = test_X['Survived']
train_X1 = train_X.drop(columns=['PassengerId', 'Survived'])
test_X1  = test_X.drop(columns=['PassengerId', 'Survived'])

# test set
test = pd.get_dummies(test[features])
test = test.drop(columns=['PassengerId'])

#list(test_X1.columns)


# In[ ]:


# logistic regression
model = fn_model(train_X1, train_y1, test_X1, test_y1, model_name='logistic')


# In[ ]:


# Decision trees
model = fn_model(train_X1, train_y1, test_X1, test_y1, model_name='tree')


# In[ ]:


# Decision trees
model = fn_model(train_X1, train_y1, test_X1, test_y1, model_name='randomforest')


# In[ ]:


model.feature_importances_


# In[ ]:


model


# In[ ]:


# test set
test['Pclass'] = test['Pclass'].astype(object)
test_y = pd.get_dummies(test[['Pclass','Sex','Age','Parch']])
test_y.head()


# In[ ]:


test_y.describe()


# In[ ]:


y_pred_final = model.predict(test_y)
len(y_pred_final)


# In[ ]:


test_y['survived'] = y_pred_final
test_y['passenger_Id'] = test['PassengerId']
submission = test_y[['passenger_Id', 'survived']]
submission.head()


# In[ ]:


submission.to_csv('titanic_submission.csv', index = False)

