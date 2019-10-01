#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import describe
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')

df_train = pd.read_csv('../input/train.csv')

df_train.columns


# In[ ]:


df_train.head


# In[ ]:


# % missing data
df_train.isnull().mean()


# In[ ]:


# name, ticket, cabin --> vielleicht später interessantl, PassengerId auch raus
X_data = df_train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X_data.iloc[0:10, :]


# In[ ]:


# Datentypen
X_data.dtypes


# In[ ]:


describe(df_train.Survived)


# In[ ]:


describe(df_train.Survived[df_train.Sex =='male'])


# In[ ]:


describe(df_train.Survived[df_train.Sex =='female'])


# In[ ]:


describe(df_train.Survived[(df_train.Sex =='female') & (df_train.Pclass == 1)])


# In[ ]:


describe(df_train.Survived[(df_train.Sex =='male') & (df_train.Pclass == 1)])


# In[ ]:


describe(df_train.Survived[(df_train.Sex =='male') & (df_train.Pclass == 3)])


# In[ ]:


# Dummi vars encoden
X_data = pd.get_dummies(X_data, drop_first=True)
X_data


# In[ ]:


corr = X_data.corr()
corr


# In[ ]:


hm = sns.heatmap(corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})
plt.show()


# In[ ]:


X_data.columns


# In[ ]:


X_data = X_data.drop(columns=['Embarked_Q', 'Embarked_S'])
X_data.iloc[0:10, :]


# In[ ]:


sns.distplot(X_data.Age[X_data.Age.isnull() == False])


# In[ ]:


# daten als numpy arrays für Regressionen splitten:
X = X_data.iloc[:, 1:6] 
y = X_data.iloc[:, 0]

# impute age as mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(X)
X = imputer.transform(X)
X


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y)

y_pred = classifier.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
cm


# In[ ]:


X_data = df_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
X_data = pd.get_dummies(X_data, drop_first=True)
X_data.columns


# In[ ]:


X_data = pd.get_dummies(X_data, drop_first=True)
# daten als numpy arrays für Regressionen splitten:
X = X_data.iloc[:, 1:]
y = X_data.iloc[:, 0]

# impute age as mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(X)
X = imputer.transform(X)

classifier.fit(X, y)

y_pred_2 = classifier.predict(X)

from sklearn.metrics import confusion_matrix
cm_2 = confusion_matrix(y, y_pred_2)
cm_2


# In[ ]:


# read in test data - Logistic Regression
df_test = pd.read_csv('../input/test.csv')
df_test.iloc[0:10, :]


# In[ ]:


# Prepare data
X_test = df_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
X_test = pd.get_dummies(X_test, drop_first=True)
X_test.columns


# In[ ]:


#X = X_test.iloc[:, 1:]
imputer.fit(X_test)
X_test = imputer.transform(X_test)
y_pred_test = classifier.predict(X_test)
len(y_pred_test)


# In[ ]:


#predictions = pd.DataFrame(df_test['PassengerId'])
#predictions['Survived'] = y_pred_test

# predictions.to_csv('predictions_test_2.csv', sep=",", header=True, index=False)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_knn.fit(X, y)
y_pred_knn = classifier_knn.predict(X)
cm_knn = confusion_matrix(y, y_pred_knn)
cm_knn


# In[ ]:


from sklearn.svm import SVC
classifier_SVC = SVC(kernel='rbf', random_state = 0)
classifier_SVC.fit(X, y)
y_pred_SVC = classifier_SVC.predict(X)
cm_SVC = confusion_matrix(y, y_pred_SVC)
cm_SVC


# In[ ]:


y_pred_SVC = classifier_SVC.predict(X_test)


# In[ ]:


# predict test data and write file for submit:

predictions = pd.DataFrame(df_test['PassengerId'])
predictions['Survived'] = y_pred_SVC

#predictions.to_csv('predictions_test_SVC.csv', sep=",", header=True, index=False)
# Overfitting!


# In[ ]:




