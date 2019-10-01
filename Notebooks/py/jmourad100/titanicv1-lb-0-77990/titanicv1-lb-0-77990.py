#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV

get_ipython().magic(u'matplotlib inline')

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.head()


# ## Data Cleaning

# In[ ]:


survived_train = df_train.Survived
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])
data.info()


# In[ ]:


data.Name.tail()


# <style>
# .output_png {
#     display: table-cell;
#     text-align: center;
#     vertical-align: middle;
# }
# </style>

# In[ ]:


## Extracting a new feature
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);


# In[ ]:


data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr','Major', 'Lady', 'Sir',
                                       'Col', 'Capt', 'Countess','Jonkheer'],'Special')
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);


# In[ ]:


data['Has_Cabin'] = ~data.Cabin.isnull()


# In[ ]:


data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
data.head()


# In[ ]:


## Handling missing values
# data.info()

# Impute missing values for Age, Fare, Embarked
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data['Embarked'] = data['Embarked'].fillna('S')
data.info()


# In[ ]:


# Binning numerical columns

data['CatAge'] = pd.qcut(data.Age, q=4, labels=False)
data['CatFare'] = pd.qcut(data.Fare, q=4, labels=False)
data = data.drop(['Age', 'Fare'], axis=1)


# In[ ]:


# Create column of number of Family members onboard
data['Fam_Size'] = data.Parch + data.SibSp
data = data.drop(['SibSp','Parch'], axis=1)
data.head()


# In[ ]:


# Transform into binary variables
data_dum = pd.get_dummies(data, drop_first=True)


# ## Building a predictive model with the new dataset

# In[ ]:


data_train = data_dum.iloc[:891]
data_test = data_dum.iloc[891:]

# Transform into arrays for sklearn
X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


param_grid = {
    'max_depth' : np.arange(1, 9)
}

clf = tree.DecisionTreeClassifier()
clf_cv = GridSearchCV(clf, param_grid=param_grid)
clf_cv.fit(X, y)

print("Tuned Decision Tree Parameters : {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))


# In[ ]:


Y_pred = clf_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('predictions.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




