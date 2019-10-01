#!/usr/bin/env python
# coding: utf-8

# Titnaic intro

# # 1. Titanic data explore

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head(5)


# In[ ]:


test.head(5) #survived 칼럼만 없다. 왜냐? 타겟이기 때문(종속변수 = 예측해야하는 것이기 때문)


# In[ ]:


train.shape


# In[ ]:


test.shape #칼럼 한개가 적다 = survived


# In[ ]:


train.info() #총 891개가 있어야 결측값(NaN) 없는 것, 그러나 Age, Cabin같은경우 NaN이 많다 


# In[ ]:


test.info()


# In[ ]:


train.isnull().sum() #이렇게 NaN(Null) 값을 표시할 수도 있다


# # <h3>1-2. Data Visualization 데이터 시각화 #

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set() #setting seaborn default for plots - 디폴트 값으로 설정


# In[ ]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts() #feature에 따라 생존한 value(사람) 카운트
    dead = train[train['Survived']==0][feature].value_counts() #feature에 따라 죽은 value(사람) 카운트 
    df = pd.DataFrame([survived,dead]) #데이터프레임으로 묶고
    df.index = ['Survived','Dead'] #인덱스 달아주고 
    df.plot(kind='bar',stacked=True, figsize=(10,5)) #차트그리기 


# In[ ]:


bar_chart('Sex') #성별에 따라서 죽었는지 살았는지 


# 남자가 많이 죽었음을 알 수 있다 - 여자들이 더 많이 살아남았다.

# In[ ]:


bar_chart('Pclass') #클래스에 따라서 죽었는지 살았는지 - 위 Data dictionary 참조


# 3등석(값 싼 좌석)일수록 많이죽었다

# In[ ]:


bar_chart('SibSp') #가족수에 따라서 죽었는지 살았는지 - 위 Data dictionary 참조


# 혼자 탄 경우 조금 더 많이 죽었다

# In[ ]:


bar_chart('Embarked') #승선한 선착장에 따라서 죽었는지 살았는지 - 위 Data dictionary 참조


# 애매하긴 하지만 S에서 탔을 경우 더 많이 죽을 가능성이 있다.

# # 2. Feature Engerning 데이터 전처리

# In[ ]:


train.head(5)


# have to deal with null values.
# 
# 1. 일반적으로 머신러닝 알고리즘은 text를 잘 읽지 못한다. 그래서 이를 숫자로 바꿔주는 과정을 진행한다 - one hot encoding 등
# 
# 2. Nan(null값)을 채워줄 방법이 필요하다. 평균값으로 대체하거나 null값을 가지고 있는 row를 drop하는 등 다양한 방법이 있다.

# # <h3> 2-1. Name

# 이름탭에서 이름 자체는 큰 영향이 없다. 하지만 Mr, Mrs, Miss 등 성별, 결혼 유무를 알 수 있는 타이틀은 매우 중요한 정보가 될 것이다. 따라서 해당 타이틀을 추출한 후 이름 칼럼을 삭제한다.

# In[ ]:


train_test_data = [train, test] # train과 test set 합침


# In[ ]:


train_test_data #train(891 rows) + test(418 rows)가 합쳐진것 확인


# In[ ]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
    


# In[ ]:


train['Title'].value_counts()


# In[ ]:


test['Title'].value_counts()


# Title dictionary::<br>
# Mr = 0<br>
# Mrs = 1<br>
# Mrs = 2<br>
# Others = 3<br>
# 
# 참고로 위 숫자는 order 관계가 아니다(우선순위, 순서가 없다)

# In[ ]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    
    #위 Title Dictionary에 맞게 숫자를 mapping 해준다. 숫자로 바꾸는 이유는 아까 말했던 것처럼
    #대부분 머신러닝 알고리즘들은 텍스트를 읽지 못하기 때문 


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# 이제 Name 칼럼을 삭제하자

# In[ ]:


train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# In[ ]:


bar_chart('Title')


# 역시 남자(=0) 이 많이 죽었다

# # <h3> 2-2. Sex

# In[ ]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


bar_chart('Sex')


# # <h3> 2-3. Age

# Nan값을 채워야 한다. 이 때 Title을 바탕으로 0은 0나이의 평균(남자 나이의 평균), 1은 1나이의 평균(기혼 여성이면 기혼 여성의 평균) 미혼여성은 미혼여성의 평균..등 Title별로 group을 지어서 해당 group에 속하는 평균을 Nan값에 넣는다

# In[ ]:


train.head(5)


# In[ ]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

#train의 Age 칼럼의 nan값을 train의 title로 gourp을 지어서 해당 그룹의 age칼럼의 median값으로 대체하겠다.
#0 = Mr, 1 = Mrs, 2 = Miss, 3 = Others


# In[ ]:


test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
#test 값을 ~ 위와동일F


# In[ ]:


train.head(5)


# Age의 NaN값이 제거되었음을 알 수 있다.

# # <h3> 2-4. Age_Binning

# Age의 scale이 넓기 때문에 Scale을 조절하기로 한다. 
# 정교화 방법을 사용할수도 있지만, 이번에는 십대 이십대 삼십대 등 구간을 정하고
# 그 구간별로 그룹화하도록 한다.
# 
# 코딩 처음배울때 점수에따라 학점 부여하는 while문, switch문 등을 생각하면 된다.

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# age dictionary::
# <br>
# child: 0<br>
# young: 1<br>
# adult: 2<br>
# mid-age: 3<br>
# senior: 4<br>
# 
# 16세 이하는 0(child), 17~26세는 1(young) 등등.. 으로 변환한다는 것이다.
# <br>

# In[ ]:


train.head(5)


# In[ ]:


bar_chart('Age')


# 이제 좀 더 깔끔하게 보인다

# # <h3> 2-5. Embarked

# embarked는 탑승한 선착장에 관한 정보이다. 고소득 거주자 지역에서 탑승하였으면 1등석일 확률이 높고 생존할 확률이 높아지겠지만, 그 반대라면 낮아지지 않을 가능성이 있기 때문에 어느정도 유의미한 변수라 할 수 있다.

# In[ ]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

#Embarked 칼럼에서 Pclass가 1인 인스턴스의 갯수를 카운트하여 Pclass1 변수에 담는다
#2, 3도 반복


# In[ ]:


df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# 결과를 볼 때, Q지역은 좀 못사는 곳인것 같은데.. 그 외는 파악하기 힘들다. 하지만 전체 탑승객중 S의 비율이 압도적으로 높기 때문에, Embarked가 Nan이면 그냥 S라고 해도 무방하다는 가정을 할 수 있다.

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


train.head(5)


# In[ ]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[ ]:


train.head(5) #마찬가지로 text는 못읽으니 숫자로 매핑해준다


# # <h3> 2-6. Fare

# In[ ]:


# Nan인 인스턴스가 속한 Pclass의 median값을 해당 결측지를 가진 인스턴스에 넣어준다. 아까 Age랑 동일
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[ ]:


train.head(5)


# Fare도 역시 Scale 해준다

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[ ]:


train.head(5)


# # <h3> 2-7. Cabin

# In[ ]:


train.Cabin.value_counts()


# 앞에 알파벳만 따와보도록 한다

# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts() #Pclass=1에 해당하는 Cabin 값을 카운트
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts() #반복
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# 1등석에는 ABCDE로 시작하는 cabin이 많지만 2등석 3등석은 아예 없다.

# In[ ]:


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


#Pclass의 median으로 Cabin 결측치 대체
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# # <h3> 2-8. Familysize

# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1 #sib = 형제자매, Parch = 부모자식
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1 #즉 형제자매 수 + 부모자식수 + 나 = 우리가족수 


# In[ ]:


train["FamilySize"].max()


# In[ ]:


test["FamilySize"].max()


# In[ ]:


#FamilySize의 범위는 1~11이다. 따라서 위에서 설명한 비닝 방식으로 정규화를 해준다

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[ ]:


train.head(5)


# In[ ]:


#티켓번호, 형제자매수, 부모가족수 칼럼은 드랍하도록 한다
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1) #인덱스 필요없음 


# In[ ]:


train.head(5)


# In[ ]:


#train.to_csv('train_dropnulll.csv', index=False)


# In[ ]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape #survived를 때서 target값으로 준다 


# # 3. Modeling

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import numpy as np


# In[ ]:


train.info()


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import xgboost as xgb
#use 10 fold cross validation
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) #10개의 fold로 나눈다


# <h2> We will use Voting Classifier 

# - RF Classifier
# - GB Classifier
# - SVM Classifier
# - XGB Classifier

# <h5> RF

# In[ ]:


RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [3, 8, 8],
              "min_samples_split": [2, 3, 8],
              "min_samples_leaf": [1, 3, 8],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC, rf_param_grid, cv=k_fold, scoring="accuracy",  verbose = 1)
#print(score)

gsRFC.fit(train_data,target)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_
 


# <h5> GB

# In[ ]:


GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=k_fold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(train_data,target)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# <h5> SVC 

# In[ ]:


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=k_fold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(train_data,target)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# In[ ]:


gsSVMC.best_estimator_


# <h5>XGB Classifier

# In[ ]:


XGBC = XGBClassifier()
xgb_param_grid = {'max_depth':[3,5,7],
                  'min_child_weight':[3,5,6],
                  'gamma': [ 0, 0.001, 0.01, 0.1, 1],
                  'learning_rate':[0.1, 0.05, 0.01]}

gsXGBC = GridSearchCV(XGBC,param_grid = xgb_param_grid, cv=k_fold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsXGBC.fit(train_data,target)

XGBC_best = gsXGBC.best_estimator_

# Best score
gsXGBC.best_score_




# <h5> Voting Classifier

# In[ ]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best), 
('svc', SVMC_best),('gbc',GBC_best), ('xgb', XGBC_best)], voting='hard', n_jobs=4)

votingC = votingC.fit(train_data, target)


# In[ ]:


votingC.predict


# In[ ]:


test_data = test.drop("PassengerId", axis=1).copy()


# In[ ]:


prediction = votingC.predict(test_data) 


# In[ ]:


#케글에 제출할 csv파일 저장
#submission = pd.DataFrame({
#        "PassengerId": test["PassengerId"],
#        "Survived": prediction
#    })

#submission.to_csv('submission.csv', index=False)


# # Reference

# https://www.youtube.com/channel/UCxP77kNgVfiiG6CXZ5WMuAQ

# http://scikit-learn.org/stable/index.html

# In[ ]:




