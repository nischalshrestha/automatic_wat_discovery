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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train.columns


# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


test.columns


# In[ ]:


train.info()


# In[ ]:


#https://www.kaggle.com/tanetboss/starter-guide-preprocessing-randomforest
train['Survived'].value_counts(sort = False)


# In[ ]:


#https://www.kaggle.com/tanetboss/starter-guide-preprocessing-randomforest
#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)


# In[ ]:


#https://stackoverflow.com/questions/30503321/finding-count-of-distinct-elements-in-dataframe-in-each-column
train.nunique()


# In[ ]:


y_train = train.Survived


# In[ ]:


train = train.drop('Name', axis=1)


# In[ ]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
train.groupby('Ticket')['Age'].agg(['size', 'count', 'mean'])


# In[ ]:


train = train.drop('Ticket', axis=1)


# In[ ]:


train = train.drop('PassengerId', axis=1)


# In[ ]:


train.corr()


# In[ ]:


train = train.drop('Survived', axis=1)


# In[ ]:


train.info()


# In[ ]:


train.Embarked.value_counts()


# In[ ]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
const_cols = [c for c in train.columns if train[c].nunique(dropna=False)==1 ]
const_cols


# In[ ]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
train.groupby('Pclass')['Age'].agg(['size', 'count', 'mean'])


# In[ ]:


train.describe()


# In[ ]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
train.groupby('Sex')['Age'].agg(['size', 'count', 'mean'])


# In[ ]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
train.groupby('Embarked')['Age'].agg(['size', 'count', 'mean'])


# In[ ]:


test.info()


# In[ ]:


test.describe()


# In[ ]:


train.info()


# In[ ]:


train[['Pclass','Sex','Cabin','Embarked']]


# In[ ]:


#https://www.kaggle.com/moghazy/eda-for-iris-dataset-with-svm-and-pca
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = train[['Age','SibSp','Parch', 'Fare']].copy()
train_scaled = scaler.fit_transform(train_scaled)
train_scaled = pd.DataFrame(train_scaled,columns=['Age','SibSp','Parch', 'Fare'])
train_scaled = pd.concat([train_scaled,train[['Pclass','Sex','Cabin','Embarked']]],axis=1)
print(train_scaled.info())


# In[ ]:


train_scaled.describe()


# In[ ]:


#train_one_hot_encoded = pd.get_dummies(train)
train_one_hot_encoded = pd.get_dummies(train_scaled)
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
data_with_imputed_values = my_imputer.fit_transform(train_one_hot_encoded)


# In[ ]:


train_one_hot_encoded.shape


# In[ ]:


train_one_hot_encoded.columns


# In[ ]:


#https://www.kaggle.com/tanetboss/starter-guide-preprocessing-randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  cross_val_score,GridSearchCV


param_grid  = { 
                'n_estimators' : [500,1200],
               # 'min_samples_split': [2,5,10,15,100],
               # 'min_samples_leaf': [1,2,5,10],
                'max_depth': range(1,5,2),
                'max_features' : ('log2', 'sqrt'),
                'class_weight':[{1: w} for w in [1,1.5]]
              }

GridRF = GridSearchCV(RandomForestClassifier(random_state=15), param_grid)

GridRF.fit(data_with_imputed_values, y_train)
#RF_preds = GridRF.predict_proba(X_test)[:, 1]
#RF_performance = roc_auc_score(Y_test, RF_preds)

print(
    #'DecisionTree: Area under the ROC curve = {}'.format(RF_performance)
     "\nBest parameters \n" + str(GridRF.best_params_))


rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)
rf.fit(data_with_imputed_values, y_train)

Rfclf_fea = pd.DataFrame(rf.feature_importances_)
#print(Rfclf_fea)


# In[ ]:


Rfclf_fea["Feature"] = list(train_one_hot_encoded.columns) 
Rfclf_fea.sort_values(by=0, ascending=False).head(10)


# In[ ]:


#https://www.kaggle.com/moghazy/eda-for-iris-dataset-with-svm-and-pca
import seaborn as sns
import matplotlib.pyplot as plt

df_imp = pd.DataFrame(data_with_imputed_values, columns = train_one_hot_encoded.columns)
print(df_imp.columns)
df_imp = pd.concat([df_imp, pd.DataFrame(y_train)], axis=1)
print(df_imp.columns)
corr = df_imp.iloc[:,[0,3,4,5,6,157]].corr()
f, ax = plt.subplots(figsize=(15, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)


# In[ ]:


print(type(data_with_imputed_values))
print(data_with_imputed_values.shape)


# In[ ]:


#https://www.kaggle.com/sibelkcansu/machine-learning-classification
color_list1 = ['red' if i=='male' else 'blue' for i in train.Sex]
plt.subplots(figsize=(10,10))
plt.scatter(train.Fare,train.Age,color=color_list1, alpha=0.8)
plt.xlabel("Fare")
plt.ylabel("Age")
plt.grid()
plt.title("Fare vs Age Scatter Plot",color="black",fontsize=15)
plt.show()


# In[ ]:


#https://www.kaggle.com/sibelkcansu/machine-learning-classification
color_list1 = ['red' if i==0 else 'blue' for i in y_train]
plt.subplots(figsize=(10,10))
plt.scatter(train.Fare,train.Age,color=color_list1, alpha=0.8)
plt.xlabel("Fare")
plt.ylabel("Age")
plt.grid()
plt.title("Fare vs Age Scatter Plot",color="black",fontsize=15)
plt.show()

