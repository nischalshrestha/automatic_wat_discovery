#!/usr/bin/env python
# coding: utf-8

# # 데이터 불러오기
# 
# import pandas as pd
# pd.read_csv("파일경로")

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.gender_submission.csv

input_file = ["../input/gender_submission.csv",
              "../input/train.csv",
              "../input/test.csv"]

gender = pd.read_csv(input_file[0], header = 0)
train = pd.read_csv(input_file[1], header = 0)
test = pd.read_csv(input_file[2], header = 0)


# # 데이터 전처리

# ## 1. 데이터 확인
# 
# ### 1-1. 함수로 확인 <br>
# <br>
# data.head() : data 앞부분 보기 <br>
# data.shape() : data dimension 보기 <br>
# data.info() : data 의 변수, 갯수, 자료형 보기 <br>
# data.isnull() : data 의 결측치 여부를 T / F 로 알려준다 <br>
# data.isnull().sum() : data 의 결측치의 수 를 변수별로 알 수 있다 <br>

# In[ ]:


train.info()


# In[ ]:


test.info()


# Age와 Cabin 변수의 결측치가 많이 존재함을 알 수 있다

# ### 1-2. chart로 확인
# <br>
# R처럼 자동으로 plot 을 그려주는 함수가 없으므로 직접 정의해 주어야 함 <br>
# 범주형 변수에 대해 barplot을 그려보겠다 <br>

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# In[ ]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts() # survived 라는 값에 대해 수를 세줌
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


bar_chart('Sex') # 생존자 중 여성이 많음을 알 수 있다.


# In[ ]:


bar_chart('Pclass') # 3등석의 사람들이 많이 죽음을 알 수 있다.


# In[ ]:


bar_chart('SibSp') # 동승자 중 형제나 배우자가 없는 사람이 훨씬 많이 죽음을 알 수 있다.


# In[ ]:


bar_chart('Parch') # 동승자 중 부모님이나 자식이 없는 사람이 훨씬 더 많이 죽음을 알 수 있다


# In[ ]:


bar_chart('Embarked') # S 자리에 있던 사람들이 많이 죽었음을 알 수 있다.


# ## 2. 데이터 전처리 
# 
# data['변수명'].value_counts() : 범주형 변수에 대한 정보, table과 비슷한 기능 <br>
# data.str.extract(' 정규표현식 ') : re.search() 와 동일한 기능 <br>
# data.drop(' 변수명 ') : 변수 열 삭제 <br>
# data.fillna(' 값 ') : 결측치 채우기 <br>
# data.groupby("기준변수")["선택변수"].transform("값") : 선택변수를 기준변수에 대한 값으로 채우기 <br>
# 
# ### 2-1. Name <br>

# In[ ]:


train['Sex'].value_counts()


# In[ ]:


train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train['Title'].value_counts()


# train set에 있는 Sex 와 Name 의 전처리 결과를 비교해 보니, <br>
# Mr 와 Male 의 수 가 거의 동일했고 <br>
# Mrs + Miss 와 Female 의 수가 거의 동일했다. <br>
# 그래서 Sex 와 Name 이 의미하는 바가 비슷하다고 판단, Name 변수를 제거함

# In[ ]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


#train.drop('Name', axis=1, inplace=True)
#test.drop('Name', axis=1, inplace=True)
#train.drop('Title', axis=1, inplace=True)
#test.drop('Title', axis=1, inplace=True)


# In[ ]:


train.info()


# #### 2-2. Age <br>
# info 로 확인했을때 Age의 결측치가 많은 것을 확인했음 <br>
# Age 의 분포를 보면 거의 정규분포를 따르는 것을 볼 수 있다. 따라서 mean 으로 결측치를 채워주겠다.

# In[ ]:


facet = sns.FacetGrid(train, aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.add_legend()
 
plt.show()


# In[ ]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


train.head(30)
train.groupby("Title")["Age"].transform("median")


# 또한 Age를 범주형 변수로 바꿔주는 작업이 필요<br>
# 그래프를 보면 0~10 세의 생존률이 높고, 20~30 세의 생존률이 낮으며 나머지는 비슷한 것을 알 수 있음
# 

# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'editAge',shade= True)
facet.set(xlim=(0, train['editAge'].max()))
facet.add_legend()
 
plt.show()


# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# In[ ]:


bar_chart('Age')


# ### 2-3. Embarked
# Embarked 는 결측치가 몇개 없으므로 가장 많은 'S'로 채워줌

# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')


# In[ ]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# ### 2-4. Fare
# PClass 와 Fare는 의미하는 바가 거의 비슷하고, PClass 는 결측치가 없으므로 Fare는 각 PClass의 중간값으로 채워줌  <br>
# 그래프를 그려보니 낮은 요금일 수록 생존률이 낮은것을 확인, 낮은 요금의 범주를 0으로 범주화 

# In[ ]:


train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
plt.show()


# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# ### 2-5. Cabin
# 그래프를 그려보면 1등급에는 F G T 가 거의 없음, F G T 는 2, 3등급의 Cabin number라고 간주 <br>
# 앞에서 우리가 범주화했던 숫자들은 0 ~ 1 , 0 ~ 4 등이 있었기 때문에 scaling해서 0 ~ 2.8 사이의 숫자로 범주화 <br>
# 또한 몇개 안되는 결측치는 범주화 후 median으로 진행

# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


cabin_mapping = {"A": 2.8, "B": 2.4, "C": 2, "D": 1.6, "E": 1.2, "F": 0.8, "G": 0.4, "T": 0}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# ### 2-6 Family size
# 혼자 탄 경우 많이 죽었고 (1명), 가족이 있으면 생존률이 높았음 <br>

# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


#train['FamilySize'] = list(map(lambda x: 1 if x >1 else 0, train['FamilySize']))


# In[ ]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[ ]:


#test['FamilySize'] = list(map(lambda x: 1 if x >1 else 0, test['FamilySize']))


# 앞에서 성별 mapping 안해준것을 처리하고, passenger id, age, ticket 열을 지워줌

# In[ ]:


train['Sex'] = list(map(lambda x: 1 if x=='female' else 0, train['Sex']))


# In[ ]:


test['Sex'] = list(map(lambda x: 1 if x=='female' else 0, test['Sex']))


# In[ ]:


features_drop = ['Ticket', 'SibSp', 'Parch', 'PassengerId']
features_drop1 = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop1, axis=1)


# In[ ]:


train.info()
train = train.drop('Name',axis=1)
test = test.drop('Name',axis=1)


# In[ ]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[ ]:


def classifier(clf):
    scoring = 'accuracy'
    score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    print(round(np.mean(score)*100,2))


# In[ ]:


clf = [KNeighborsClassifier(n_neighbors = 13),
       DecisionTreeClassifier(),
       RandomForestClassifier(n_estimators=13),
       SVC(),
       XGBClassifier(),
       XGBClassifier(
             learning_rate =0.01,
             n_estimators=1000,
             max_depth=3,
             min_child_weight=3,
             gamma=0,
             subsample=0.7,
             colsample_bytree=0.7,
             nthread=3,
             scale_pos_weight=1,
             seed=27
       )]


# In[ ]:


result = []
for clfs in clf:
    acc = classifier(clfs)
    result.append(result)
print(result)


# In[ ]:


clf1 = KNeighborsClassifier(n_neighbors = 13)
clf2 = DecisionTreeClassifier()
clf3 = RandomForestClassifier(n_estimators=13)
clf4 = SVC()
clf5 = XGBClassifier()
clf6 = XGBClassifier(
             learning_rate =0.01,
             n_estimators=1000,
             max_depth=3,
             min_child_weight=3,
             gamma=0,
             subsample=0.7,
             colsample_bytree=0.7,
             nthread=3,
             scale_pos_weight=1,
             seed=27
       )
eclf = VotingClassifier(estimators=[('knn', clf1), 
                              ('dt', clf2), 
                              ('rf', clf3), 
                              ('svc', clf4), 
                              ('xgb', clf5), 
                              ('xgb1', clf6)], voting='hard')


# In[ ]:


for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6 ,eclf], ['KNN','Decision Tree', 'Random Forest', 'SVM','xgboost','tuned_xgboost', 'Ensemble']):
    scores = cross_val_score(clf, train_data, target, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# In[ ]:


clf = XGBClassifier()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[ ]:



submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)


# In[ ]:



submission = pd.read_csv('submission.csv')
submission.head()


# In[ ]:




