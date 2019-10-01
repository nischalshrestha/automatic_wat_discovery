#!/usr/bin/env python
# coding: utf-8

# # Домашняя работа 2

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#источник данных https://www.kaggle.com/c/titanic/data 


# In[ ]:


data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


data.head()


# ## Feature engineering

# In[ ]:


num_cols = [col for col in data.columns if data[col].dtype == 'float64' 
            or data[col].dtype == 'int64']
num_cols.remove('Survived')


# ### Работа с пропущенными значениями

# In[ ]:


sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()


# In[ ]:


data.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


# видим, что в столбце "Cabin" большинство пропусков, значит можем его удалить


# In[ ]:


data.drop('Cabin',inplace=True, axis = 1)
test.drop('Cabin',inplace=True, axis = 1)


# In[ ]:


#data.loc[data['Embarked'].isnull(), 'Embarked'] = 'NoneType'
#data.loc[data['Age'].isnull(), 'Age'] = 0
#test.loc[data['Age'].isnull(), 'Age'] = 0


# In[ ]:


data = data.fillna(data.mean())
test = test.fillna(test.mean())
data.loc[data['Embarked'].isnull(), 'Embarked'] = 'NoneType'


# ### Преобразование категориальных признаков

# In[ ]:


data.info()


# In[ ]:


test.info()


# In[ ]:


# все имена в столбце Ticket уникальны, значит этот столбец не принесет доп пользы и мы можем его удалить
# проверила на уникальность- data['Name'].value_counts().plot.barh()
# все остальные столбцы с категориальными значениями имеют повторения, будем их преобразовывать


# In[ ]:


data.drop('Name',inplace=True, axis = 1)


# In[ ]:


test.drop('Name',inplace=True, axis = 1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['Sex'] = encoder.fit_transform(data['Sex'])
test['Sex'] = encoder.fit_transform(test['Sex'])
data['Embarked'] = encoder.fit_transform(data['Embarked'])
test['Embarked'] = encoder.fit_transform(test['Embarked'])
data['Ticket'] = encoder.fit_transform(data['Ticket'])
test['Ticket'] = encoder.fit_transform(test['Ticket'])


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold 
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
import time


# In[ ]:


cv = KFold(data.shape[0], shuffle=True, 
           random_state=42, n_folds=5)


# In[ ]:


gs = GridSearchCV(RandomForestClassifier(
    n_jobs=-1,
    random_state=42),
                  param_grid={'max_features': 
                              [None, 'log2', 'sqrt'], 
                              'max_depth': 
                              range(6, 10),
                              'n_estimators':
                              range(50, 150, 30),
                              'criterion': 
                              ['gini', 'entropy'],
                              'min_samples_leaf':
                              range(1, 6),
                              'warm_start':
                              [True, False]},
                  cv=cv,
                  scoring='accuracy',
                  n_jobs=-1,
                  verbose=1)


# In[ ]:


start = time.time()
gs.fit(data[num_cols],  data['Survived'])
print((time.time() - start)/60)


# In[ ]:


gs.best_score_


# In[ ]:


gs.best_params_


# In[ ]:


rfc = RandomForestClassifier(n_jobs=-1,
                             random_state=42,
                             n_estimators=gs.best_params_['n_estimators'], 
                             max_depth=gs.best_params_['max_depth'], 
                             max_features=gs.best_params_['max_features'],
                             criterion=gs.best_params_['criterion'],
                             min_samples_leaf=gs.best_params_['min_samples_leaf'],
                             warm_start=gs.best_params_['warm_start'])
rfc.fit(data[num_cols], data['Survived'])


# ### Финальный результат

# In[ ]:




