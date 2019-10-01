#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


print('Reading Data...\n')
train_raw  = pd.read_csv('../input/train.csv')
test_raw   = pd.read_csv('../input/test.csv')
train  = train_raw.copy(deep = True)
test = test_raw.copy(deep = True)
data = pd.concat([train, test], ignore_index=True)
train_len = len(train)
test_len = len(test)

print('Shape of train : {}'.format(train.shape))
print('Shape of test  : {}'.format(test.shape))
print('Shape of data  : {}'.format(data.shape))


# In[ ]:


data.head()


# In[ ]:


print('data with null values:\n', data.isnull().sum())


# 
# ## データ分析
# 
# 史実によれば、
# 
# 1.  **ウィメン・アンド・チルドレン・ファースト**。女性と子供が優先的に救助された。*(Sex, Age)*
# 2.  **優先的に救助されたのは一等、二等船室客**。三等船室客はアメリカ合衆国移民法の理由により物理的にも隔離された状態になっていた。*(Pclass)*
# 
# とのこと。ちなみに、三等船室には英語が話せる人がそもそも少なかったため、助かった人は英語が話せる人が多かったとも。
# 念のためデータで確認してみる。

# In[ ]:


plt.figure(figsize=[10, 10])
plt.subplot(221)
sns.barplot('Sex','Survived',data=data)
plt.subplot(222)
sns.barplot('Pclass','Survived',data=data, hue='Sex')

sns.FacetGrid(data=data, hue='Survived', aspect=3).map(sns.kdeplot, 'Age', shade=True)
plt.ylabel('Passenger Density')
plt.legend()
plt.show()


# ## データ分析(Cabin, Fare, Embarked)
# 
# ### Cabin :
# 
# 　全データ1,309件のうち、欠落データが1,014件(77.46%)もある。
# 
# 　有効な分析ができないと思われるため、**除外することにする**。
# 
# ### Fare :
# 
# 　船賃は船室のクラスと同一乗船人数で決まる（同じチケット番号の人は同じ船賃になっている）。
# 
# 　結果的にPclass次第となるため、**除外することにする**。
# 
# 
# ### Embarked :
# 
# 　常識的に考えると、乗船地自体が生死に直接関係したとは考え難い。
# 
# 　なお、タイタニックの航路は以下の通り（Wikipediaより）。
# 
# 　![Wikipediaより](https://upload.wikimedia.org/wikipedia/commons/5/51/Titanic_voyage_map.png)
# 
# 　一方で、乗船地毎の生存率を取るとC(herbourg)が最も高く見えるが、
#  
# 　これはCherbougから乗船した客に一等、二等船室客の割合が多かったことが原因と思われる。
#  
# 　また、S(outhampton)が最も低いのは、Southamptonから乗船した客に男性の割合が多かったことが原因と思われる。
#   
# 　そのため、乗船地（Embarked）も**除外することにする**。
#  

# In[ ]:


plt.figure(figsize=[15,10])
plt.subplot(331)
sns.barplot('Embarked','Survived',data=data)
plt.subplot(332)
sns.countplot('Embarked',data=data, hue='Pclass')
plt.subplot(333)
sns.countplot('Embarked',data=data, hue='Sex')


# In[ ]:


data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['FamilySizeBin'] = pd.cut(data['FamilySize'], [0, 1, 4, 11])

plt.figure(figsize=[10,10])
plt.subplot(221)
sns.barplot('Parch', 'Survived', data=data)
plt.subplot(222)
sns.barplot('SibSp', 'Survived', data=data)
plt.subplot(223)
sns.barplot('FamilySize', 'Survived', data=data)
plt.subplot(224)
sns.barplot('FamilySizeBin', 'Survived', data=data)
plt.show()


# In[ ]:


data['Title']  = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False) 

# 特徴量のうち、"Mr", "Master", "Mrs", "Miss"を残してその他を"Misc"とする
print(data['Title'].value_counts())
title_min = 20
title_names = (data['Title'].value_counts() < title_min)
data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

plt.figure(figsize=[10, 10])
plt.subplot(221)
sns.barplot('Title', 'Survived', data=data)
plt.subplot(222)
sns.barplot('Title', 'Age', data=data)


# In[ ]:


GroupByTitle = data.groupby(['Title']).mean()
AgeGroupLabel = GroupByTitle['Age']
AgeGroupLabel
data['Age'].fillna(data['Title'].apply(lambda x: AgeGroupLabel.loc[x]), inplace = True)

sns.FacetGrid(data=data, hue='Survived', aspect=3).map(sns.kdeplot, 'Age', shade=True)
plt.ylabel('Passenger Density')
plt.legend()
plt.show()

plt.figure(figsize=[5, 5])
data['AgeBin'] = pd.cut(data['Age'], [0, 4, 8, 16, 100])
sns.barplot('AgeBin', 'Survived', data=data)
plt.show


# In[ ]:


le = LabelEncoder()

data['Sex_Code'] = le.fit_transform(data['Sex'])
data['FamilySizeBin_Code'] = le.fit_transform(data['FamilySizeBin'])
data['Title_Code'] = le.fit_transform(data['Title'])
data['AgeBin_Code'] = le.fit_transform(data['AgeBin'])

drop_column = ['Age',
                        'AgeBin',
                        'Sex',
                        'FamilySizeBin',
                        'Title',
                        'Cabin',
                        'Embarked', 
                        'Fare', 
                        'Ticket',
                        'Name', 
                        'Parch',
                        'SibSp', 
                        'FamilySize',
                        'PassengerId']

data.drop(drop_column, axis=1, inplace = True)


# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


data = pd.get_dummies(data, columns=['Pclass','Sex_Code', 'FamilySizeBin_Code','Title_Code', 'AgeBin_Code'])


# In[ ]:


train_df = data.drop(['Survived'], axis=1)
test_df = data['Survived']
train_values = train_df.values
test_values = test_df.values

train_X_ALL = train_values[:train_len, :]
train_y_ALL = test_values[:train_len].astype(int)

pred_X = train_values[train_len:, :]

(train_X, test_X ,train_y, test_y) = train_test_split(train_X_ALL, train_y_ALL, test_size = 0.25, random_state=0)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV

parameters = {
        'n_estimators'      : [10,25,50,75,100],
        'random_state'      : [0],
        'n_jobs'            : [4],
        'min_samples_split' : [5, 10, 15, 20, 25, 30],
        'max_depth'         : [5, 10, 15, 20, 25, 30]
}

#clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
#clf.fit(train_X, train_y) 
#print(clf.best_estimator_)

rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=15, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=20,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=4,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
rfc.fit(train_X, train_y)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
stratifiedkfold = StratifiedKFold(n_splits=5)
print('Cross-validation scores: \n{}'.format(cross_val_score(rfc, train_X_ALL, train_y_ALL, cv=stratifiedkfold)))

train_size = np.arange(15, 446, step=30)

from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(rfc, train_X_ALL, train_y_ALL, train_sizes=train_size, cv=3)
plt.figure
plt.plot(train_sizes, train_scores, label='train')
plt.plot(train_sizes, valid_scores, label='valid')
plt.legend()
plt.show

features = train_df.columns
importances = rfc.feature_importances_
indices = np.argsort(importances)

plt.figure()
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.show()


# In[ ]:


Y_pred = rfc.predict(pred_X)

submission = pd.DataFrame({
        "PassengerId": test_raw["PassengerId"].astype(int),
        "Survived": Y_pred
    })

submission.to_csv('submission.csv', index=False)


# In[ ]:




