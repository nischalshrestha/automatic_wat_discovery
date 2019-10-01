#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
train.info()


# In[ ]:


train.Survived.value_counts()


# In[ ]:


float(train.Survived.value_counts()[1]) / train.Survived.value_counts()[0]


# In[ ]:


train.head()


# In[ ]:


# directly fed into sklearn cross validation.

data = pd.get_dummies(train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1))

X = data.drop('Survived',axis = 1)
y = data['Survived']


# In[ ]:


from sklearn.preprocessing import Imputer

X_i = Imputer(strategy = 'median').fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_i, y, test_size = .3, random_state = 233
)


# In[ ]:


from xgboost import XGBClassifier

xlf = XGBClassifier(
 scale_pos_weight = .62,
 objective= 'binary:logistic',
 seed=27)


# In[ ]:


xlf.fit(X_train, y_train)


# In[ ]:


xlf_prob = xlf.predict_proba(X_test)[:,1]
xlf_pred = xlf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, auc

print(confusion_matrix(xlf_pred, y_test))
print(accuracy_score(xlf_pred, y_test))


# In[ ]:


importance = pd.DataFrame()
importance['Feature'] = X.columns
importance['importance'] = xlf.feature_importances_


# In[ ]:


importance.sort_values(ascending = False, by = 'importance')


# In[ ]:


X_2 = pd.get_dummies(pd.read_csv('../input/test.csv').drop(['PassengerId',
                                                            'Name', 'Ticket', 'Cabin'], axis = 1))
id_2 = pd.read_csv('../input/test.csv')['PassengerId']


# In[ ]:


submit = pd.DataFrame()

submit['PassengerId'] = id_2
submit['Survived'] = xlf.predict(np.array(X_2))

submit.to_csv('plain_xgb_2.csv', index = False)

