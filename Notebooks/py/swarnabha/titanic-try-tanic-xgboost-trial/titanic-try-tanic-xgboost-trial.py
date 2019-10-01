#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#import data 
path = ('../input/train.csv')
data = pd.read_csv(path)


# In[48]:


#explore datatypes and no of rows (N/A values)
data.info()

#check coloumns
#data.columns


# In[2]:


data.head(5)
#data.Cabin.value_counts()


# In[2]:


#data feature checks
data_2 = data
data_1 = data.drop(labels = ['Survived'], axis = 1)
data_2 = data_1.drop(labels = ['PassengerId','Ticket','Cabin','Name'], axis = 1)

data_target = data.Survived
data_train = data_2

data_train.head()


# In[5]:


#model checks
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#getting the cross validation score with different models passed as method
def get_mae(X, y, method):
    return  cross_val_score(method, X, y, scoring = 'accuracy', cv = 4)

#calulating the baseline score
def baseline(y):
    a = len(y)
    b = [0]*a
    return accuracy_score(y, b)


# In[7]:


#model checks basic
from sklearn.ensemble import RandomForestClassifier


# dropiing columns  with NaN values to get minimal data 
lowest_data = data_train.dropna(axis = 'columns')

#Dropping all catagorical data columns, to be treated later with one hot encoding
lowest_data = lowest_data.select_dtypes(exclude = ['object'])
lowest_data.isna().sum()

base = baseline(data_target)
RFC_lowest_data = get_mae(lowest_data, data_target, RandomForestClassifier()).mean()
SV_lowest_data = get_mae(lowest_data, data_target, svm.SVC(gamma = 'auto')).mean()
LR_lowest_data = get_mae(lowest_data, data_target, LogisticRegression()).mean()

print('baseline score = ', base)
print('RandomForestClassifier = ', RFC_lowest_data)
print('SVM = ', SV_lowest_data)
print('LogisticRegression = ', LR_lowest_data)


# In[105]:


#regular imputation
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()

data_imp = data_train.select_dtypes(exclude = ['object'])
imputed_data = my_imputer.fit_transform(data_imp)

imputed_data_train = pd.DataFrame(imputed_data, columns = data_imp.columns)

imputed_data_train.isna().sum()

base = baseline(data_target)
RFC_lowest_data = get_mae(imputed_data_train, data_target, RandomForestClassifier()).mean()
SV_lowest_data = get_mae(imputed_data_train, data_target, svm.SVC(gamma = 'auto')).mean()
LR_lowest_data = get_mae(imputed_data_train, data_target, LogisticRegression()).mean()

print('baseline score = ', base)
print('RandomForestClassifier = ', RFC_lowest_data)
print('SVM = ', SV_lowest_data)
print('LogisticRegression = ', LR_lowest_data)


# In[8]:


#to check if we can impute male and female age seperately 

data_male = data.drop(data[data.Sex == 'female'].index)
data_female = data.drop(data[data.Sex == 'male'].index)
#data_male.head(15)
print("male mean age:",data_male['Age'].mean())
print("female mean age:",data_female['Age'].mean())


# In[9]:


#impute age with Sex based mean age
import math

def data_age_imp(d):
    data_imp = d
    a=len(d.Sex)
    f = data_female['Age'].mean()
    m = data_male['Age'].mean()
    for i in range(0, a):
        if math.isnan(data_imp.Age[i]):
            if d.Sex[i] == 'female':
                data_imp.Age[i] = f
            else:
                data_imp.Age[i] = m
    return data_imp;


# In[12]:


data_one = data_age_imp(data_train)
data_one_hot = pd.get_dummies(data_one)

data_one_hot.head()

base = baseline(data_target)
RFC = get_mae(data_one_hot, data_target, RandomForestClassifier()).mean()
SV = get_mae(data_one_hot, data_target, svm.SVC(gamma = 'auto')).mean()
LR = get_mae(data_one_hot, data_target, LogisticRegression()).mean()

print('baseline score = ', base)
print('RandomForestClassifier = ', RFC)
print('SVM = ', SV)
print('LogisticRegression = ', LR)


# In[13]:



data_one_hot.head()
data_one_hot_d1 = data_one_hot.drop(labels = ['Sex_female'], axis = 1)

base = baseline(data_target)
RFC = get_mae(data_one_hot_d1, data_target, RandomForestClassifier()).mean()
SV = get_mae(data_one_hot_d1, data_target, svm.SVC(gamma = 'auto')).mean()
LR = get_mae(data_one_hot_d1, data_target, LogisticRegression()).mean()

print('baseline score = ', base)
print('RandomForestClassifier = ', RFC)
print('SVM = ', SV)
print('LogisticRegression = ', LR)


# In[14]:


from sklearn.model_selection import train_test_split

#latest model for training : data_one_hot_d1
#target : data_target

X_train, X_test, y_train, y_test = train_test_split(data_one_hot_d1, data_target, random_state=6)



# In[98]:


#Checking random forest classifier

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


RF_classifier = RandomForestClassifier()
RF_classifier.fit(data_one_hot_d1, data_target)
pred_RFC = RF_classifier.predict(X_test)

score = accuracy_score(pred_RFC, y_test)
print('Random forest Classifier', score)

Confusion_matrix = metrics.confusion_matrix(y_test, pred_RFC)
print(Confusion_matrix)




# In[85]:


#Checking XGB classifier
from xgboost import XGBClassifier

score_test = []
score_train = []
base = []

for i in range(0,20):
    l=1+i
    XGB_classf = XGBClassifier(n_estimators=l, learning_rate=0.1)
    XGB_classf.fit(X_train, y_train)
    pred_test = XGB_classf.predict(X_test)
    pred_train = XGB_classf.predict(X_train)
    score_test.append(1-accuracy_score(pred_test, y_test))
    score_train.append(1-accuracy_score(pred_train, y_train))
    base.append(l)

print('testing score', accuracy_score(pred_test, y_test))
print('training score', accuracy_score(pred_train, y_train))
print('fin',l)



# In[96]:


#Checking random forest classifier
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(base,score_test)

plt.scatter(base,score_train)

plt.show()


# In[90]:



path = ('../input/test.csv')
data_test = pd.read_csv(path)
new_test_data = data_test.drop(labels = ['Name', 'Ticket', 'Cabin'], axis = 1)
new_test_data_d1 = data_age_imp(new_test_data)



print(1)


# In[91]:


new_test_data_d2 = pd.get_dummies(new_test_data_d1)


final_train, final_test = data_one_hot_d1.align(new_test_data_d2, join='left', axis=1)


# In[92]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()

final_test_1 = my_imputer.fit_transform(final_test)
final_test_2 = pd.DataFrame(final_test_1, columns = final_test.columns)


# In[97]:


#Submissions based on XGBClassifier
from xgboost import XGBClassifier

XGB_classf_f = XGBClassifier(n_estimators=8, learning_rate=0.1)
XGB_classf_f.fit(data_one_hot_d1, data_target)


# In[103]:


#Submissions based RandomForestClassifier
pred_final_RF = RF_classifier.predict(final_test_2)

my_submission = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': pred_final_RF})

my_submission.to_csv('submission_RFC.csv', index=False)


# In[104]:


#Submissions based XGB boost
pred_final_XGB = XGB_classf_f.predict(final_test_2)

my_submission_XGB = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': pred_final_XGB})

my_submission_XGB.to_csv('submission_XGB.csv', index=False)

