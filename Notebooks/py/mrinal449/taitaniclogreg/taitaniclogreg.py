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
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, model_selection, metrics


# ### Defining data preparation module

# In[ ]:


def _get_categoricals(df):
    return list(set(df.columns) - set(df._get_numeric_data().columns))

def preprocess_data(df):
    cat_cols = _get_categoricals(df)
    num_cols = list(df._get_numeric_data().columns)
    
    print("One-hot-encoding will be applied on the columns: {cols}"
         .format(cols=cat_cols))
    cat_df_filled = pd.get_dummies(df[cat_cols].fillna(""))
    
    # Filling missing values for numerical columns with the mean of all values for that column
    num_df = df[list(df._get_numeric_data().columns)]
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean')
    num_df_filled = pd.DataFrame(imp.fit_transform(num_df), columns=num_df.columns)
    
    # Concatenate num and cat dfs
    concat_df = pd.concat([cat_df_filled, num_df_filled], axis=1)
    
    # Scaling features and returning
    return preprocessing.StandardScaler().fit_transform(concat_df)


# ### Reading training data for input

# In[ ]:


# Reading the training data
train_df_actual = pd.read_csv("../input/train.csv")    .drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Embarked'], axis=1)


# ### Splitting label and features

# In[ ]:


train_y = train_df_actual['Survived']
train_x = preprocess_data(train_df_actual.drop('Survived', axis=1))


# ### Splitting data for training, testing and evaluation[](http://)

# In[ ]:


train_x, test_x, train_y, test_y = model_selection.train_test_split(train_x, train_y, test_size=0.3, random_state=69)
test_x, eval_x, test_y, eval_y = model_selection.train_test_split(test_x, test_y, test_size=0.5, random_state=69)


# * ### **Setting up LogReg model (benchmark)**

# In[ ]:


model = LogisticRegression()
model.fit(train_x, train_y)


# In[ ]:


preds = model.predict_proba(test_x)[:,1]


# In[ ]:


auc = metrics.roc_auc_score(test_y, preds)
logloss = metrics.log_loss(test_y, preds)

print ("The Log Loss for this model on the test set is {0}".format(logloss))
print ("The AUC for this model on the test set is {0} ".format(auc))


# #### **Actual Model**

# In[ ]:


parameters = {'C':[0.1, 0.3, 0.6, 0.9], 'penalty' : ['l1']}
mod = LogisticRegression()
clsf = model_selection.GridSearchCV(mod, parameters, cv=2, n_jobs=2, verbose=3, scoring ='roc_auc')
clsf.fit(train_x, train_y)
preds = clsf.predict_proba(test_x)[:, 1]


# In[ ]:


rocauc = metrics.roc_auc_score(test_y, preds)
logloss = metrics.log_loss(test_y, preds)


# In[ ]:


print('Log Loss: {0:.4f} '.format(logloss))
print('AUC: {0:.4f} '.format(rocauc))
print('Best performing C value: {}'.format(clsf.best_params_ ))


# In[ ]:


parameters2 = {'C':[0.1, 0.3, 0.6, 0.9], 'penalty' : ['l2']}
mod2 = LogisticRegression()
clsf2 = model_selection.GridSearchCV(mod2, parameters2, cv=2, n_jobs=2, verbose=3, scoring ='roc_auc')
clsf2.fit(train_x, train_y)
preds2 = clsf2.predict_proba(test_x)[:, 1]


# In[ ]:


rocauc2 = metrics.roc_auc_score(test_y, preds2)
logloss2 = metrics.log_loss(test_y, preds2)


# In[ ]:


print('Log Loss: {0:.4f} '.format(logloss2))
print('AUC: {0:.4f} '.format(rocauc2))
print('Best performing C value: {}'.format(clsf2.best_params_ ))


# #### Seems lasso (l1) is doing better

# #### **Trying Polynomial (with interaction)**

# In[ ]:


poly2 = preprocessing.PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
train_x_poly2 = poly2.fit_transform(train_x)
test_x_poly2 = poly2.transform(test_x)
eval_x_poly2 = poly2.transform(eval_x)


# In[ ]:


parameters3 = {'C':[0.1, 0.3, 0.6, 0.9], 'penalty' : ['l1']}
mod3 = LogisticRegression()
clsf3 = model_selection.GridSearchCV(mod3, parameters3, cv=2, n_jobs=2, verbose=3, scoring ='roc_auc')
clsf3.fit(train_x_poly2, train_y)
preds3 = clsf3.predict_proba(test_x_poly2)[:, 1]


# In[ ]:


rocauc3 = metrics.roc_auc_score(test_y, preds3)
logloss3 = metrics.log_loss(test_y, preds3)


# In[ ]:


print('Log Loss: {0:.4f} '.format(logloss3))
print('AUC: {0:.4f} '.format(rocauc3))
print('Best performing C value: {}'.format(clsf3.best_params_ ))


# #### Polynomial with degree 2 is giving a better result. Let's check with degree 3

# In[ ]:


poly3 = preprocessing.PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
train_x_poly3 = poly3.fit_transform(train_x)
test_x_poly3 = poly3.transform(test_x)
eval_x_poly3 = poly3.transform(eval_x)


# In[ ]:


parameters4 = {'C':[0.1, 0.3, 0.6, 0.9], 'penalty' : ['l1']}
mod4 = LogisticRegression()
clsf4 = model_selection.GridSearchCV(mod4, parameters4, cv=2, n_jobs=2, verbose=3, scoring ='roc_auc')
clsf4.fit(train_x_poly3, train_y)
preds4 = clsf4.predict_proba(test_x_poly3)[:, 1]


# In[ ]:


rocauc4 = metrics.roc_auc_score(test_y, preds4)
logloss4 = metrics.log_loss(test_y, preds4)


# In[ ]:


print('Log Loss: {0:.4f} '.format(logloss4))
print('AUC: {0:.4f} '.format(rocauc4))
print('Best performing C value: {}'.format(clsf4.best_params_ ))


# #### So Polynomial with degree 2 is working better than even degree 2 (and keeps itself as far away from overfitting as possible)

# #### Let's evaluate our final model. Criteria:
# * C = 0.9
# * Penalty = L1 (Lasso)

# In[ ]:


preds_eval2 = clsf3.predict_proba(eval_x_poly2)[:,1]


# In[ ]:


rocauc_eval2 = metrics.roc_auc_score(eval_y, preds_eval2)
logloss_eval2 = metrics.log_loss(eval_y, preds_eval2)


# In[ ]:


print('Log Loss: {0:.4f} '.format(logloss_eval2))
print('AUC: {0:.4f} '.format(rocauc_eval2))


# ### **Now we will predict the survival status for the actual test data**

# In[ ]:


test_df_read = pd.read_csv("../input/test.csv")
test_df_actual = test_df_read.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Embarked'], axis=1)


# In[ ]:


test_df_read.head()


# In[ ]:


test_x_actual = preprocess_data(test_df_actual)


# In[ ]:


test_x_actual_poly2 = poly2.transform(test_x_actual)


# In[ ]:


predict_test_actual = clsf3.predict(test_x_actual_poly2)


# In[ ]:


result_prediction = pd.concat([test_df_read['PassengerId'], pd.DataFrame(predict_test_actual, columns=['Survived'])], axis=1).set_index('PassengerId')


# In[ ]:


result_prediction.to_csv('titanic_prediction.csv')


# In[ ]:


print(check_output(["ls", "./"]).decode("utf8"))


# In[ ]:




