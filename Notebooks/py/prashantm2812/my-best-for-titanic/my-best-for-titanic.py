#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../input/train.csv')
y_train = data['Survived']
data = data.drop(['Survived'], axis = 1)
data.head()


# In[ ]:


y_train.head()


# In[ ]:


data = data.append(pd.read_csv('../input/test.csv'))
data.tail()


# In[ ]:


data = data.drop(['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare'], axis = 1)
data['FamMem'] = data['SibSp'] + data['Parch']


# In[ ]:


data.head()


# In[ ]:


check = data.describe()
check.head()


# In[ ]:


for i in list(check):#list(check)- column names in check
    data[i] = data[i].fillna(data[i].mean())
check_again = data.describe(include = 'all')
data['Embarked'] = data['Embarked'].fillna('S')
data


# In[ ]:


categorical = data.select_dtypes(exclude = ['number'])
columns_categorical = list(categorical)
        
col_ind = []
for i in columns_categorical:
    col_ind.append(data.columns.get_loc(i))
col_ind


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for i in columns_categorical:
    labenc = LabelEncoder()
    data[i] = labenc.fit_transform(data[i])
ohenc = OneHotEncoder(categorical_features = col_ind)
data = ohenc.fit_transform(data).toarray()


# In[ ]:


X_train = data[0:891, :]
X_test = data[891:, :]


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
sc_test = StandardScaler()
X_test = sc_test.fit_transform(X_test)


# In[ ]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 118, gamma = 0.013, random_state = 0)#79.904%
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)


# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [118, 119, 120, 121, 122], 'kernel' : ['poly', 'sigmoid', 'rbf'], 'gamma' : [0.01, 0.012, 0.013, 0.014, 0.02], 'random_state' : [0, 11, 42]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 20,
                           verbose = 60, n_jobs = 1)
grid_search = grid_search.fit(X_train, y_train)


# In[ ]:


best_score = grid_search.best_score_
best_params = grid_search.best_params_


# In[ ]:


best_score


# In[ ]:


best_params


# In[ ]:


y_pred_test = grid_search.predict(X_test)
# any suggestions are welcomed


# In[ ]:




