#!/usr/bin/env python
# coding: utf-8

# # Combination of Clustering and Classification
# 
# The purpose of this notebook is to practice the pandas library, plus K-Means and XGBoost.
# 
# I assume that there is groups of the passengers by any relationships such as party, dining, or class, so I create eight groups by K-means clustering before predicting survivors from the disaster.
# 
# ## - Category
# 
# ### 1. Feature Engineering
#     1) Embarked
#     2) Fare
#     3) Pclass
#     4) Sex
#     5) Parch & SibSp
#     6) Age
#     7) Cabin
#     
# ### 2. Data Prediction
#     1) Clustering
#     2) Prediction
# 
# 

# ## 1. Feature Engineering
# 
# ** - Define Libraries and Read Files **

# In[ ]:


# Data Manipulation
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Data Prediction
from sklearn.preprocessing import StandardScaler
import xgboost
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Data Set
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_all = df_train.append(df_test)

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'], 'Survived': np.nan})


#  ** In order to manipulate the data easily and conveniently, I merge the train and test sets and I create the submission file in advance. **

# In[ ]:


df_all.info()


# ** In the table above, I recognize 'Age', 'Cabin', 'Embarked', 'Fare', and 'Survived' sections have null values.('Survived' is what I have to predict.) **
# 
# ** I decide **
# 
# ** not to consider 'Ticket' and 'Name', **
# 
# ** to predict 'Age',  'Embarked' and  'Fare', **
# 
#  ** to categorize 'Cabin'. **

# ## 1) Embarked

# In[ ]:


# Embarked
df_all[df_all.Embarked.isnull() == True]


# In[ ]:


df_all.Embarked[(df_all.Pclass == 1) & (abs(df_all.Fare - df_train.Fare.mean()) < df_all.Fare.std()) & (df_all.Cabin.str.contains('B') == True)].value_counts()


# In[ ]:


df_all.Embarked.value_counts()


# In[ ]:


df_all.Embarked = df_all.Embarked.fillna('S')


# ** Through the data exploration, I replace these missing values with 'S'. **

# In[ ]:


plt.figure(figsize = (10, 5))
sns.barplot(x='Embarked', y='Survived', data=df_all)
plt.show()


# In[ ]:


# Convert str to numeric
df_all.Embarked = df_all.Embarked.replace('S',2)
df_all.Embarked = df_all.Embarked.replace('C',0)
df_all.Embarked = df_all.Embarked.replace('Q',1)


# ** I convert all string values to numeric to predict the data mathematically. **

# ## 2) Fare

# In[ ]:


df_all[df_all.Fare.isnull() == True]


# In[ ]:


df_all.Fare = df_all.Fare.fillna(df_all.Fare[df_all.Pclass == 3].median())


# ** The median of the 3rd class is put into the null space. **

# In[ ]:


plt.figure(figsize = (10, 5))
df_all.Fare.plot(kind = 'kde')
plt.show()
df_all.Fare.skew()


# In[ ]:


plt.figure(figsize = (10, 5))
df_all.Fare = np.log(df_all.Fare + 1)
df_all.Fare.plot(kind = 'kde')
plt.show()
df_all.Fare.skew()


# ** The original distribution of the Fare is right-skewed so I use log-transformation. **

# ## 3) Pclass

# In[ ]:


plt.figure(figsize = (10, 5))
sns.barplot(x='Pclass', y='Survived', data=df_all)
plt.title('Survival rate by Pclass')
plt.show()


# ## 4) Sex

# In[ ]:


plt.figure(figsize = (10, 5))
sns.barplot(x='Sex', y='Survived', palette='coolwarm', data=df_all)
plt.title('Survival rate by Sex')
plt.show()


# In[ ]:


# Convert
df_all.Sex = df_all.Sex.replace('male',0)
df_all.Sex = df_all.Sex.replace('female',1)


# ## 5. Parch & SibSp

# In[ ]:


df_all['Family'] = df_all.Parch + df_all.SibSp + 1
# 1: Single one


# ** I consider the number of family. **

# In[ ]:


plt.figure(figsize = (10, 5))
sns.barplot(x='Family', y='Survived', data=df_all)
plt.title('Survival rate by Family')
plt.show()


# In[ ]:


plt.figure(figsize = (10, 5))
sns.barplot(x='Family', y='Survived', hue='Sex', palette='coolwarm', data=df_all)
plt.title('Survival rate by Family')
plt.show()


# ## 6. Age

# In[ ]:


plt.figure(figsize = (10, 5))
df_all.Age.plot(kind = 'kde')
plt.show()
df_all.Age.skew()


# In[ ]:


plt.figure(figsize = (10, 5))
plt.subplot2grid((1,2),(0,0))
age_bins = [0, 1, 8, 15, 20, 30, 40, 50, 65, 80]
df_train.Age[df_train.Survived == 1].hist(bins=age_bins, alpha=0.5, color='pink')
df_train.Age[df_train.Survived == 0].hist(bins=age_bins, alpha=0.2)
plt.ylim(0,150)
plt.legend(('Survived', 'Died'))
plt.show()


# In[ ]:


print(df_train.Survived[df_train.Age < 2].value_counts(normalize=True),'\n',
    df_train.Survived[(df_train.Age >= 2) & (df_train.Age < 8)].value_counts(normalize=True),'\n',
    df_train.Survived[(df_train.Age >= 8) & (df_train.Age < 15)].value_counts(normalize=True),'\n',
    df_train.Survived[(df_train.Age >= 15) & (df_train.Age < 30)].value_counts(normalize=True),'\n',
    df_train.Survived[(df_train.Age >= 30) & (df_train.Age < 40)].value_counts(normalize=True),'\n',
    df_train.Survived[(df_train.Age >= 40) & (df_train.Age < 50)].value_counts(normalize=True),'\n',
    df_train.Survived[(df_train.Age >= 50) & (df_train.Age < 65)].value_counts(normalize=True),'\n',
    df_train.Survived[df_train.Age >= 65].value_counts(normalize=True))


# ** - Age Categorization **
# 
# ** I think that it is hard to predict itself because the values are spread widely and continuously, so I try to predict category values of Age instead of the original Age. **

# In[ ]:


df_all.Age[df_all.Age < 2] = 0
df_all.Age[(df_all.Age >= 2) & (df_all.Age < 8)] = 1
df_all.Age[(df_all.Age >= 8) & (df_all.Age < 15)] = 4
df_all.Age[(df_all.Age >= 15) & (df_all.Age < 30)] = 6
df_all.Age[(df_all.Age >= 30) & (df_all.Age < 40)] = 2
df_all.Age[(df_all.Age >= 40) & (df_all.Age < 50)] = 5
df_all.Age[(df_all.Age >= 50) & (df_all.Age < 65)] = 3
df_all.Age[df_all.Age >= 65] = 7


# ** - Age Prediction **

# In[ ]:


CF_set = ['Embarked','Fare','Parch', 'SibSp', 'Family', 'Pclass', 'Sex', 'Age']
df_sub = df_all[CF_set]
unknown_set = df_sub[df_sub.Age.isnull() == True].drop('Age', axis=1)
known_set = df_sub[df_sub.Age.isnull() == False]


# In[ ]:


x_test = unknown_set.values.reshape((263,7))

y_train = known_set.Age.values.reshape((1046,1))
x_train = known_set.drop('Age', axis=1).values.reshape((1046, 7))


# In[ ]:


xgb = xgboost.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
xgb.fit(x_train,y_train)
age_pred_xgb = xgb.predict(x_test)


# In[ ]:


unknown_set['Age'] = age_pred_xgb


# In[ ]:


df_all.Age[df_all.Age.isnull() == True] = unknown_set.Age


# ## 7. Cabin

# In[ ]:


df_all.Cabin[df_all.Pclass == 1] = df_all.Cabin[df_all.Pclass == 1].fillna('X')
df_all.Cabin[df_all.Pclass == 2] = df_all.Cabin[df_all.Pclass == 2].fillna('Y')
df_all.Cabin[df_all.Pclass == 3] = df_all.Cabin[df_all.Pclass == 3].fillna('Z')


# ** 'Cabin' and 'Pclass' have a strong relationship. **

# In[ ]:


df_all['Cabin_Series'] = df_all['Cabin'].str[:1]
df_all.Cabin_Series.value_counts()


# In[ ]:


plt.figure(figsize = (10, 5))
sns.barplot(x='Cabin_Series', y='Survived', data=df_all)
plt.title('Survival rate by Cabin')
plt.show()


# In[ ]:


# Categorization
df_all.Cabin_Series[df_all.Cabin_Series == 'A'] = 7
df_all.Cabin_Series[df_all.Cabin_Series == 'B'] = 2
df_all.Cabin_Series[df_all.Cabin_Series == 'C'] = 4
df_all.Cabin_Series[df_all.Cabin_Series == 'D'] = 0
df_all.Cabin_Series[df_all.Cabin_Series == 'E'] = 1
df_all.Cabin_Series[df_all.Cabin_Series == 'F'] = 3
df_all.Cabin_Series[df_all.Cabin_Series == 'G'] = 5
df_all.Cabin_Series[df_all.Cabin_Series == 'T'] = 7
df_all.Cabin_Series[df_all.Cabin_Series == 'X'] = 6
df_all.Cabin_Series[df_all.Cabin_Series == 'Y'] = 8
df_all.Cabin_Series[df_all.Cabin_Series == 'Z'] = 9


# ## 2. Data Prediction

# ## 1) Clustering

# ** I think that there exist some groups in the data so I divide into eight groups by K-Means Clustering before predicting 'Survived'. **

# In[ ]:


cl_set = df_all[CF_set]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cl_set)
kmeans = KMeans(n_clusters=8, random_state=0).fit(scaled_data)
df_all['Cluster_label'] = kmeans.labels_


# In[ ]:


df_all.Cluster_label.value_counts()


# ## 2) Prediction

# In[ ]:


data_set = ['Embarked','Fare','Parch', 'SibSp', 'Family', 'Pclass', 'Sex', 'Age', 'Cluster_label','Cabin_Series', 'Survived']
df = df_all[data_set]
df_train = df[df.Survived.isnull() == False]
df_train.shape


# In[ ]:


df_test = df[df.Survived.isnull() == True]
df_test = df_test.drop('Survived', axis=1)
df_test.shape


# In[ ]:


x_test = df_test.values.reshape((418,10))

y_train = df_train.Survived.values.reshape((891,1))
x_train = df_train.drop('Survived', axis=1).values.reshape((891, 10))


# ** - 'Survived' Prediction with XGBoost **

# In[ ]:


xgb1 = xgboost.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
xgb1.fit(x_train,y_train)
y_pred_xgb1 = xgb1.predict(x_test)


# In[ ]:


xgb2 = xgboost.XGBClassifier(max_depth=2, n_estimators=300, learning_rate=0.05)
xgb2.fit(x_train,y_train)
y_pred_xgb2 = xgb2.predict(x_test)


# ** - 'Survived' Prediction with Random Forest **

# In[ ]:


rf = RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)


# ** - 'Survived' Prediction with Decision Tree **

# In[ ]:


decision_tree = tree.DecisionTreeClassifier(random_state=1)
decision_tree.fit(x_train,y_train)
y_pred_dt = decision_tree.predict(x_test)


# ** I choose the predicted values voted unanimously.(y_pred_tmp = 0 or 1) **
# 
# ** The other values are predicted by XGBoost once again.**

# In[ ]:


y_pred_tmp = (y_pred_xgb1 + y_pred_rf + y_pred_xgb2 + y_pred_dt)/4


# In[ ]:


df_test['Survived'] = y_pred_tmp
re_test = df_test[(df_test.Survived == 0.25) | (df_test.Survived == 0.5) | (df_test.Survived == 0.75)] # re-test x-set
re_test = re_test.drop('Survived', axis=1)

re_test.shape


# In[ ]:


re_train_temp = df_test[(df_test.Survived == 0) | (df_test.Survived == 1)]
re_train = df_train.append(re_train_temp)
re_train.shape


# In[ ]:


x_test = re_test.values.reshape((re_test.shape[0],10))


# In[ ]:


y_train = re_train.Survived.values.reshape((re_train.shape[0],1))
x_train = re_train.drop('Survived', axis=1).values.reshape((re_train.shape[0], 10))


# In[ ]:


xgb = xgboost.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.02)
xgb.fit(x_train,y_train)
y_pred = xgb.predict(x_test)


# In[ ]:


n = int(len(y_pred_tmp))
count = 0
for i in np.arange(n):
    
    if (y_pred_tmp[i] == 0.25)|(y_pred_tmp[i] == 0.5)|(y_pred_tmp[i] == 0.75):
        y_pred_tmp[i] = y_pred[count]
        count = count + 1


# In[ ]:


submission['Survived'] = y_pred_tmp.astype('int')


# In[ ]:


#submission.to_csv('./submission.csv', index=False)


# Thank you for reading this notebook. I hope that this is useful to you. Please upvote!
