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
os.chdir('../input')
# Any results you write to the current directory are saved as output.


# In[ ]:


# 한 셀의 결과 모두 출력
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ## **1. Titanic data 탐색**

# In[ ]:


train = pd.read_csv('train.csv')


# In[ ]:


test = pd.read_csv('test.csv')


# In[ ]:


train.head(20)


# In[ ]:


test.head(10)


# In[ ]:


train.shape


# In[ ]:


test.shape # 컬럼 한개가 적다 = survived


# In[ ]:


train.info


# In[ ]:


test.info


# In[ ]:


test.info()


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum() # 이렇게 NaN(null) 값을 표시할 수도 있다


# ## ** 1-2. 데이터 시각화**

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set() #setting seaborn default for plots - 디폴트 값으로 설정


# In[ ]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts() #feature에 따라 생존한 value(사람) 카운트
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead]) # 데이터프레임으로 묶고
    df.index = ['Survived','Dead'] # 인덱스 달아주고
    df.plot(kind='bar', stacked=True, figsize=(10,5)) # 차트그리기


# In[ ]:


bar_chart('Sex')


# 남자가 많이 죽었음을 알 수 있다. 여자들이 더 많이 살아남았다.

# In[ ]:


bar_chart('Pclass')


# 3등석(값 싼 좌석)일수록 많이 죽었다.

# In[ ]:


bar_chart('SibSp')


# 혼자 탄 경우 더 많이 죽었다.

# In[ ]:


bar_chart('Embarked') # 승선한 선착장에 따라서 죽었는지 살았는지


# 애매하긴 하지만 s에서 탔을 경우 더 많이 죽을 가능성이 있다.

# ## **2. 데이터 전처리**

# In[ ]:


train.head(5)


# 1. 일반적으로 머신러닝 알고리즘은 text를 잘 읽지 못한다. 그래서 이를 숫자로 바꿔주는 과정을 진행한다. - one hot encoding 등
# 2. NaN(null)을 채워줄 방법이 필요하다. 평균값으로 대체하거나 null값을 가지고 있는 row를 drop하는 등 다양한 방법이 있다.
# 
# ### ** 2-1. Name**
#  이름탭에서 이름 자체는 큰 영향이 없다. 하지만 Mr, Mrs, Miss 등 성별, 결혼 유무를 알 수 있는 타이틀은 매우 중요한 정보가 될 것이다. 따라서 해당 타이틀을 추출한 후 이름 칼럼을 삭제한다.

# In[ ]:


train_test_data = [train, test] # train과 test set 합침


# In[ ]:


train_test_data #train(891 rows) + test(418 rows) 합쳐진 것 확인


# In[ ]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)


# 위 코드에서는 정규표현식을 사용해서 Mr. Mrs. 등 Title을 추출하고 있다.
# 1. 대문자 A-Z로 시작하면서
# 2. 두번째 문자부터는 소문자이면서
# 3. 마지막 끝이 .로 끝나는 것
# 을 추출한다는 것이다.

# In[ ]:


train['Title'].value_counts()


# In[ ]:


test['Title'].value_counts()


# In[ ]:


title_mapping = {"Mr":0, "Mrs":2,"Miss":1,"Master":3,"Dr":3,"Rev":3,"Col":3,"Major":3,"Mlle":3,"Countess":3,"Ms":3,"Lady":3,"Jonkheer":3,"Don":3,"Dona":3,"Mme":3,"Capt":3,"Sir":3}

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

    # 위 title dictionary에 맞게 숫자를 mapping해준다. 숫자로 바꾸는 이유는 아까 말했던 것처럼 대부분 머신러닝 알고리즘들은 텍스트를 읽지 못하기 때문.


# In[ ]:


train.head(5)


# In[ ]:


test.head(10)


# In[ ]:


test.info()


# In[ ]:


train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[ ]:


train.head(5)


# In[ ]:


bar_chart('Title')


# 역시 남자가 많이 죽었다.

# ### ** 2-2. Sex**

# In[ ]:


sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


bar_chart('Sex')


# ### **2-3. Age**
# NaN값을 채워야 한다. 이 때 Title을 바탕으로 0은 0나이의 평균(남자 나이의 평균), 1은 1나이의 평균(기혼 여성이면 기혼 여성의 평균, 미혼여성은 미혼여성의 평균..등) Title별로 group을 지어서 해당 group에 속하는 평균을 NaN값에 넣는다. 

# In[ ]:


train.head()


# In[ ]:


train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)

# train의 age 칼럽의 nan값을 train의 title로 group을 지어서 해당 그룹의 age 칼럼의 median값으로 대체한다.
# 0 = Mr, 1 = Mrs, 2 = Miss, 3 = Others


# In[ ]:


test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.info()


# ## **2-4. Age_Binning**
# Age의 scale이 넓기 때문에 scale을 조절하기로 한다. 정교화 방법을 사용할 수도 있지만, 이번에는 십대 이십대 삼십대 등 구간을 정하고 그 구간별로 그룹화하도록 한다.

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[ (dataset['Age'] > 16)&(dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[ (dataset['Age'] > 26)&(dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[ (dataset['Age'] > 36)&(dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ (dataset['Age'] > 62), 'Age'] = 4


# In[ ]:


train.head()


# In[ ]:


train_test_data


# In[ ]:


bar_chart('Age')


# ## **2-5. Embarked**
# embarked는 탑승한 선착장에 관한 정보이다. 고소득 거주자 지역에서 탑승하였으면 1등석일 확률이 높고 생존할 확률이 높아지겠지만, 그 반대라면 낮아지지 않을 가능성이 있기 때문에 어느정도 유의미한 변수라 할 수 있다.
# 

# In[ ]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

# Embarked 칼럼에서 Pclass가 1인 인스턴스의 갯수를 카운트하여 Pclass1변수에 담는다.
# 2,3도 반복


# In[ ]:


df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class','3rd class']
df.plot(kind='bar', stacked=True, figsize =(10,5))


# In[ ]:


# 결과를 볼 때, 전체 탑승객 중 s의 비율이 높기 때문에 Embarked가 Nan이면 그냥 s라고 봐도 무방하다고 가정할 수 있다.

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


train.head(10)


# In[ ]:


embarked_mapping ={"S":0,"C":1,"Q":2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[ ]:


train.head(5)


# ## **2-6. Fare**

# In[ ]:


test[test['Fare'].isnull()==True]


# In[ ]:


# NaN인 인스턴스가 속한 Pclass의 median값을 해당 결측치를 가진 인스턴스에 넣어준다. 아까 Age랑 동일
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'))
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'))


# In[ ]:


test.groupby('Pclass')['Fare'].transform('median').head()


# In[ ]:


test.head()


# In[ ]:


test.iloc[150:155,:]


# In[ ]:


test[test['Fare'].isnull()==True]


# In[ ]:


test.loc[[test['PassengerId']==1044],: ]['Fare']= test[test['Pclass']==3]['Fare'].mean()


# In[ ]:


test.head()


# In[ ]:


test.info()


# In[ ]:


train.info()


# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare']<=17, 'Fare']=0,
    dataset.loc[(dataset['Fare']>17)&(dataset['Fare']<=30), 'Fare']=1,
    dataset.loc[(dataset['Fare']>30)&(dataset['Fare']<=100), 'Fare']=2,
    dataset.loc[(dataset['Fare']>100), 'Fare']=3


# In[ ]:


train.head()


# ## **2-7. Cabin**

# In[ ]:


train.Cabin.value_counts()


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].str[:1]


# In[ ]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()


# In[ ]:


df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class','3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))


# 1등석에는 abcde로 시작하는 cabin이 많지만 2등석, 3등석은 아예 없다.

# In[ ]:


cabin_mapping = {'A':0, 'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2,'G':2.4,'T':2.8}
for dataset in train_test_data:
    dataset['Cabin']= dataset['Cabin'].map(cabin_mapping)


# In[ ]:


#Pclass의 median으로 Cabin 결측치 대체
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'), inplace=True)


#  Cabin에 대해서 좀 더 보충하자면, Cabin은 객실을 뜻하는 것인데 알파벳과 숫자의 조합으로 이루어진다.
# 여기서 숫자까지 분류하기에는 조금 무리가 있기 때문에 우리는 제일 앞에 있는 알파벳만 추출하여 연관성을 보기 위해 시각화한 것이다.

# ## **2-8.FamilySize**

# In[ ]:


train['Familysize'] = train['SibSp']+train['Parch']+1 # sib = 형제자매, parch = 부모자식
test['Familysize'] = test['SibSp'] + test['Parch']+1 # 즉 형제자매 수 + 부모자식 수 + 나 = 우리가족수


# In[ ]:


train['Familysize'].max()


# In[ ]:


test['Familysize'].max()


# In[ ]:


#Familysize 의 범위는 1~11이다. 따라서 위에서 설명한 방식으로 정규화를 해준다.
family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4}
for dataset in train_test_data:
    dataset['Familysize']=dataset['Familysize'].map(family_mapping)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# 티켓번호, 형제자매수, 부모가족수 칼럼은 드랍하도록 한다
features_drop=['Ticket','SibSp','Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis = 1) # 인덱스 필요없음


# In[ ]:


train.head()


# In[ ]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape # survived를 떼서 target값으로 준다


# In[ ]:


target.head()


# ## 3. Modeling

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


train.info()


# ##  **3-1. Cross validation(K-fold) 교차검증 진행**

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


# 10 개의 fold로 나눈다.
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ## **3-1.1 kNN**

# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 13) #13개의 이웃을 기준으로 측정
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring = scoring)
print(score) # 교차 검증 스코어


# cross_val_score 파라미터
# cross_val_score(estimator, x, y=None, groups=None, scoring=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
# 
# estimator : 사용할 모델
# x : 학습할 모델
# y : 매개변수 = 예측할 값
# cv : 위에서 세팅한 kfold값을 사용
# 
# *리턴값 scores : array of float, shape=(len(list(cv)),)
# 각 횟수 수행시 추정기의 점수 = 높을 수록 좋다.

# In[ ]:


# kNN Score
round(np.mean(score)*100,2) # 10번 시행시 평균 정확도


# ## ** 3-1.2 Decision Tree**

# In[ ]:


clf = DecisionTreeClassifier()
clf # 특별하게 매개변수를 건드리지 않았으므로 다 디폴트 값이 주어짐


# In[ ]:


scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_fold, n_jobs=1, scoring = scoring)
print(score) # 아까와 동일


# In[ ]:


round(np.mean(score)*100,2)


# ## ** 3-1.3 Random Forest

# In[ ]:


clf = RandomForestClassifier(n_estimators=13) #13개의 decision tree 사용
clf


# In[ ]:


scoring = 'accuracy'
score = cross_val_score(clf, train_data,target, cv=k_fold, n_jobs=1, scoring = scoring)
print(score)


# In[ ]:


# decision tree Score
round(np.mean(score)*100,2)


# In[ ]:




