#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook is a very basic and simple introductory primer to the method of ensembling (combining) base learning models, in particular the variant of ensembling known as Stacking. In a nutshell stacking uses as a first-level (base), the predictions of a few basic classifiers and then uses another model at the second-level to predict the output from the earlier first-level predictions.
# 
# The Titanic dataset is a prime candidate for introducing this concept as many newcomers to Kaggle start out here. Furthermore even though stacking has been responsible for many a team winning Kaggle competitions there seems to be a dearth of kernels on this topic so I hope this notebook can fill somewhat of that void.
# 
# I myself am quite a newcomer to the Kaggle scene as well and the first proper ensembling/stacking script that I managed to chance upon and study was one written in the AllState Severity Claims competition by the great Faron. The material in this notebook borrows heavily from Faron's script although ported to factor in ensembles of classifiers whilst his was ensembles of regressors. Anyway please check out his script here:
# 
# [Stacking Starter][1] : by Faron 
# 
# 
# Now onto the notebook at hand and I hope that it manages to do justice and convey the concept of ensembling in an intuitive and concise manner.  My other standalone Kaggle [script][2] which implements exactly the same ensembling steps (albeit with different parameters) discussed below gives a Public LB score of 0.808 which is good enough to get to the top 9% and runs just under 4 minutes. Therefore I am pretty sure there is a lot of room to improve and add on to that script. Anyways please feel free to leave me any comments with regards to how I can improve
# 
# 
#   [1]: https://www.kaggle.com/mmueller/allstate-claims-severity/stacking-starter/run/390867
#   [2]: https://www.kaggle.com/arthurtok/titanic/simple-stacking-with-xgboost-0-808

# <font color='blue'>
# 이 노트북은 앙상블링 기초 학습 모델, 특히 stacking으로 알려져 있는 앙상블 변형에 대한 매우 기초적이고 단순한 소개 입문서다. 간단히 말하면 stacking은 첫번째 레벨에서 몇 개의 기초적인 classifer로 부터 예측을 하고, 두번째 레벨의 다른 모델에서 앞의 예측들을 사용한다.
# 
# 타이타닉 데이터 셋은 이 개념을 많은 캐글 입문자들에게 소개할 수 있는 기본적인 후보이다. 게다가 많은 팀들이 캐글 competition에서 우승하는데 stacking이 일조했음에도 불구하고 이 topic에 대한 커널이 부족해 보인다. 그래서 나는 이 노트북이 그 공백을 채울 수 있길 희망한다.
# 
# 나 또한 캐글 scean에서 신참이고 첫번째로 우연히 발견하고 공부했던 적절한 앙상블/stacking 스크립트는 AllState Severity Claims 컴패티션에서 great Faron이라는 사람이 쓴 것이었다. 비록 그의 regressor 앙상블은 classifer의 앙상블로 변형됐지만 노트북의 내용은 Faron의 스크립트에서 많이 차용했다. 어쨌든 그의 스크립트를 여기서 확인해라:
# 
# [Stacking Starter][1] : by Faron 
# 
# 
# 이제 손에 있는 노트북과 나는 이 것이 앙상블링의 개념을 쉽고 간결하게 정의하고 전달할 수 있기를 바랍니다. 동일한 앙상블링 스텝을(다른 파라미터를 쓸지라도) 구현한 나의 다른 독립적인 캐글 스크립트[script][2]는 public 리더보드 점수가 0.808이고 상위 9%에 들어가기 충분하며 단 4분안에 실행된다. 그러므로 나는 그 스크립트에 더하고 향상시킬 수 있는 여지가 많다고 확신한다. 어쨌든 내가 어떻게 향상 시킬 수 있을지에 대해 댓글을 편하게 달아줘.
# 
#   [1]: https://www.kaggle.com/mmueller/allstate-claims-severity/stacking-starter/run/390867
#   [2]: https://www.kaggle.com/arthurtok/titanic/simple-stacking-with-xgboost-0-808
# </font>

# In[ ]:


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
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold


# <font color='red'>
#     
# - sklearn.cross_validation : 윈도우, 아나콘다 환경인데 임포트할 수 없다고 떴음. cross_validation 대신 model_selection으로 명칭이 변경되었다고 함.
# 
# - seaborn : 시각화 라이브러리
# 
# - plotly : 시각화 라이브러리이며 conda install로 설치 해결
# 
# - xgboost : gradient boosting을 제공하는 프레임워크.
# 
# - gradient boosting? : 기존 gradient descent는 loss를 파라미터로 미분해서 최소화 시켰는데 gradient boosting은 loss를 현재까지 학습된 모델 함수로 미분한다. 이 미분값을 다음 모델의 타겟으로 넘기고 다음 모델을 피팅한다. 기존 모델은 이 새로운 모델을 흡수하는데, 이 과정을 반복해서 bias를 줄여나간다고 한다. 
# http://4four.us/article/2017/05/gradient-boosting-simply 참조
# </font>

# # Feature Exploration, Engineering and Cleaning 
# 
# Now we will proceed much like how most kernels in general are structured, and that is to first explore the data on hand, identify possible feature engineering opportunities as well as numerically encode any categorical features.

# <font color='blue'>
# 이제 우리는 다른 대부분의 커널들이 구성된 것 처럼 진행할 것이다. 그리고 직접 데이터를 탐색하고, 피쳐 엔지니어링 기회를 식별하고 카테고리컬한 피쳐를 숫자로 인코딩해라.
# </font>

# In[ ]:


# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Store our passenger ID for easy access
PassengerId = test['PassengerId']

train.head(3)


# Well it is no surprise that our task is to somehow extract the information out of the categorical variables 
# 
# **Feature Engineering**
# 
# Here, credit must be extended to Sina's very comprehensive and well-thought out notebook for the feature engineering ideas so please check out his work 
# 
# [Titanic Best Working Classfier][1] : by Sina
# 
# 
#   [1]: https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier

# <font color='blue'>
# 우리의 업무가 어떻게든 카테고리컬 변수들로부터 정보를 추출해야하는건 놀랍지도 않다.
# 
# ** 피쳐 엔지니어링**
# 
# 여기에, 피쳐엔지니어링 아이디어들에 대한 Sina의 포괄적이고 잘 검토된 노트북으로 확장되어야 한다. 그의 업적을 확인해라.
# 
# [Titanic Best Working Classfier][1] : by Sina
# 
# 
# - [1]: https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier
# 
# 
# </font>
# 
# 
# 
# 
# <font color='red'>
# 
# 
# 
# 
# 배운점
# 
# 
# - 각 featur별로 group by 했을 때, 평균 survived 값을 확인하더라. '성별, 나이 등이 생존에 영향을 줄거같은데?' 라고 근거없는 막연한 직관으로 넘어갔던 내 방식을 반성. numeric한 변수는 4~5개 range로 나눔.
# 
# - 형재/배우자 수, 자식/부모 수 라는 2개의 feature를 합쳐 family size feature를 만들어보기도 하고, isAlone feature를 뽑기도 했다. 다양한 feature를 생성해내려 노력할 필요성 느낌.
# 
# - embarked 같은 경우엔 missing value가 존재함. Sina는 가장 많이 발생한 값으로 fillna 함. 나같은 경우 missing value가 있는 몇몇 feature들을 어떻게 처리해야할지 몰라서 missing 여부에 대한 feature를 추가했었는데... 다른 사람들의 방식에 대해 배움. Fare는 median 값으로 채웠고, Age는 평균에서 +- 1표준편차의 range에서 랜덤값 채움. 
# 
# - Name에서 Mr, Dr, Major 등 title을 뽑아낸걸 보고 소름 돋음. Name은 시작하자마자 재끼고 들어갔던 점 반성. 쓸모 없어 보이는 feature에 대해서도 고민을 할 필요 있음을 느낌.
# 
# - 이제 string 형태의 카테고리컬한 변수들을 1, 2, 3 등으로 맵핑해주고, numeric한 변수들은 1~10, 10~20 등 범주를 나누고 1, 2, 등으로 맵핑.
# 
# - feature selection으로 필요한 feature들을 선별해내는 것으로 feature engineering 마무리. 
# 
# - RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier 등 10개의 다양한 classifier들을 대상으로 위에서 만든 데이터셋을 테스트해서 가장 좋은 모델을 선정했음.
# 
# </font>
# 

# In[ ]:


full_data = [train, test]

# Some features of my own that I have added in
# Gives the length of the name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Feature engineering steps taken from Sina
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# Create a New feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;


# In[ ]:


# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)


# All right so now having cleaned the features and extracted relevant information and dropped the categorical columns our features should now all be numeric, a format suitable to feed into our Machine Learning models. However before we proceed let us generate some simple correlation and distribution plots of our transformed dataset to observe ho

# <font color='blue'>
# 좋아, 그래서 지금 features를 clean하게 만들었고, 관련 정보를 추출했고, 카테고리컬한 변수들을 numeric하게 만들었고 ML 모델에
# 입력하기 적당한 포맷으로 만들었다. 그러나 계속 진행하기 전에 간단한 상관관계와 분포 plot을 생성하여 관찰하자. 호우
# </font>
# <font color='red'>
#     
#     
# 이 사람은 SIna가 쓴 feature를 참조했고(완전 똑같진 않음), name length라는 feature를 추가함. cabin 여부도 추가했었지만 끝에 drop 했음.
# </font>

# ## Visualisations 

# In[ ]:


train.head(3)


# **Pearson Correlation Heatmap**
# 
# let us generate some correlation plots of the features to see how related one feature is to the next. To do so, we will utilise the Seaborn plotting package which allows us to plot heatmaps very conveniently as follows

# <font color='blue'>
#     
# **피어슨 상관관계 히트맵**
# 
# 한 feature가 다음 feature와 얼마나 관계있는지 보기 위해 상관관계 plot을 생성하자
# 
# 그러기 위해 우리는 편리하게 히트맵을 그려주는 Seaborn plotting 패키지를 사용한다. 
# </font>

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# **Takeaway from the Plots**
# 
# One thing that that the Pearson Correlation plot can tell us is that there are not too many features strongly correlated with one another. This is good from a point of view of feeding these features into your learning model because this means that there isn't much redundant or superfluous data in our training set and we are happy that each feature carries with it some unique information. Here are two most correlated features are that of Family size and Parch (Parents and Children). I'll still leave both features in for the purposes of this exercise.
# 
# **Pairplots**
# 
# Finally let us generate some pairplots to observe the distribution of data from one feature to the other. Once again we use Seaborn to help us.

# <font color='blue'>
# 
# **Plots에서 벗어나서**
# 
# 피어슨 상관계수 plot이 우리에게 말할 수 있는건, 강한 상관관계를 가지는 feature들이 많지 않다는 것이다. 이건 러닝 모델에 이런 feature들을 feed하는 관점에서 좋다. 왜냐하면 중복되거나 불필요한 데이터가 우리의 트레이닝 셋에 없다는 것이고 우리는 각 feature가 고유의 정보를 가지고 있다는 점에 기쁘다. Family size랑 Parch(부모, 자식 수)가 가장 상관관계가 있던 feature들이고 연습을 위해 나는 여전히 두 feature를 남길 것이다. 
# 
# **Pairplots**
# 
# 마지막으로, 하나의 feature에서 다른 feature로의 분포를 관찰하기 위해 pairplots를 생성하자. 우리는 또다시 Seaborn을 이용한다. 
# </font>

# In[ ]:


g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])


# <font color='red'>
# 
# - Seaborn 이라는 좋은 시각화 라이브러리를 알게 됨. 자주 애용할 듯.
# 
# - pairplot 해석하는 법은 그래프를 봤을 때 점들이 경향성을 띄면 feature간의 상관관계가 있다는 뜻임.
# </font>

# # Ensembling & Stacking models
# 
# Finally after that brief whirlwind detour with regards to feature engineering and formatting, we finally arrive at the meat and gist of the this notebook.
# 
# Creating a Stacking ensemble!

# <font color='blue'>
#     마침내 피쳐 엔지니어링과 포맷팅에 관한 간결한 회오리 바람이 지나간 후, 우리는 이 노트북의 핵심에 도착했다. 스택킹 앙상블을 만들자!
# </font>

# ### Helpers via Python Classes
# 
# Here we invoke the use of Python's classes to help make it more convenient for us. For any newcomers to programming, one normally hears Classes being used in conjunction with Object-Oriented Programming (OOP). In short, a class helps to extend some code/program for creating objects (variables for old-school peeps) as well as to implement functions and methods specific to that class.
# 
# In the section of code below, we essentially write a class *SklearnHelper* that allows one to extend the inbuilt methods (such as train, predict and fit) common to all the Sklearn classifiers. Therefore this cuts out redundancy as  won't need to write the same methods five times if we wanted to invoke five different classifiers.

# <font color='blue'>
#     
# **Python 클래스를 통한 도우미들**
# 
# 여기서 우리는 파이썬의 클래스들을 사용하여 더 편리하게 그것을 만들 수 있다. 프로그래밍에 초심자들은 일반적으로 클래스들이 OOP와 결합되어 있다고 듣는다. 요약하면, 클래스는 객체를 생성하는 것 뿐 아니라 함수와 메소드를 구현하기 위해 코드/프로그램을 확장하는 것을 돕는 것이다. 
# 
# 아래 코드 섹션에서, 우리는 SklearnHelper 라는 Sklearn의 모든 classifier에 대해서 내장 메소드(train, predict, fit같은) 들을 확장해주는 클래스를 필수적으로 작성한다. 그러므로 5개의 다른 classifier을 호출하려고 동일한 메소드를 5번 호출해야하는 중복성을 제거한다. 
#     
# </font>

# In[ ]:


# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
# kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
kf = KFold(n_splits=NFOLDS, random_state=SEED)
kf.get_n_splits(ntrain)
kf = kf.split(ntrain)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend XGboost classifer


# <font color='red'>
#     
# - 모듈 이름 변경되면서 KFold 함수 파라미터가 변경되어서 수정함.
# 
# </font>

# Bear with me for those who already know this but for people who have not created classes or objects in Python before, let me explain what the code given above does. In creating my base classifiers, I will only use the models already present in the Sklearn library and therefore only extend the class for that.
# 
# **def init** : Python standard for invoking the default constructor for the class. This means that when you want to create an object (classifier), you have to give it the parameters of clf (what sklearn classifier you want), seed (random seed) and params (parameters for the classifiers).
# 
# The rest of the code are simply methods of the class which simply call the corresponding methods already existing within the sklearn classifiers. Essentially, we have created a wrapper class to extend the various Sklearn classifiers so that this should help us reduce having to write the same code over and over when we implement multiple learners to our stacker.

# <font color='blue'>
#     
# 이미 이것을 알고 있어도 인내심을 가져줘, 파이썬에서 클래스나 객체를 만들지 않은 사람들 위해서 나는 위에 주어진 코드를 설명할 것이다. 내 기본 classifier를 만들 때 나는 Sklearn 라이브러리에 이미 존재하는 모델들만 사용하므로 오직 그 클래스를 상속해라.
# 
# **def init**: 클래스의 기본 생성자를 호출하는 Python 표준. 즉, 객체(classifier)를 만들려면 clf(원하는 sklearn classifer), seed(임의 seed), params(classifier의 매개변수)를 지정해야 한다. 
# 
# 나머지 코드는 단순히 skelarn classifier 안에 이미 존재하는 메소드를 호출하는 클래스의 메소드이다. 기본적으로 다양한 sklearn 분류기를 확장하는 wrapper 클래스를 만들었으므로 stacker에 여러 learner를 구현할 때 동일한 코드를 반복 작성해야 하는 부담을 줄여준다.
# 
# </font>

# ### Out-of-Fold Predictions
# 
# Now as alluded to above in the introductory section, stacking uses predictions of base classifiers as input for training to a second-level model. However one cannot simply train the base models on the full training data, generate predictions on the full test set and then output these for the second-level training. This runs the risk of your base model predictions already having "seen" the test set and therefore overfitting when feeding these predictions.

# <font color='blue'>
# 
# 이제 introductory 섹션에서 언급한 것 처럼 stacking은 기본 classifier들의 예측을 두번째 수준 모델에 대한 학습용 입력으로 사용한다. 그러나 두번째 수준의 모델이 단순히 full training 데이터 셋의 베이스 모델들을 train 할 수 없다. full test 셋에 대한 예측을 생성하고 두번째 수준 모델을 위해 출력해라. 이미 테스트 셋을 본 상태의 베이스 모델 예측과 이러한 예측을 제공할 때 오버피팅이 발생하는 위험이 있을 수 있다.
# 
# </font>

# In[ ]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

#     for i, (train_index, test_index) in enumerate(kf):
    for i, (train_index, test_index) in kf:
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# <font color='red'>
# 
# - KFold 함수 변경으로 인해 코드 수정 발생. enumerate(kf) -> kf
# 
# - oof_train : K fold 교차검증에서의 test fold들의 예측 값 (결국 train 데이터셋 사람 수만큼의 길이)
# 
# - oof_test : K번의 iteration 마다 실제 test 데이터로 예측하고 K개의 예측 결과에 대해 평균 낸 값 (test 데이터셋 사람 수만큼의 길이)
# </font> 

# # Generating our Base First-Level Models 
# 
# So now let us prepare five learning models as our first level classification. These models can all be conveniently invoked via the Sklearn library and are listed as follows:
# 
#  1. Random Forest classifier
#  2. Extra Trees classifier
#  3. AdaBoost classifer
#  4. Gradient Boosting classifer
#  5. Support Vector Machine

# <font color='blue'>
# 이제 우리의 1단계 분류를 위해 5개의 러닝 모델을 준비하자. 이 모델들은 Sklearn을 통해 편리하게 호출된다. 목록은 아래와 같다.
#     
# 
# 
# 1. Random Forest classifier
# 2. Extra Trees classifier
# 3. AdaBoost classifer
# 4. Gradient Boosting classifer
# 5. Support Vector Machine
# 
# </font>

# **Parameters**
# 
# Just a quick summary of the parameters that we will be listing here for completeness,
# 
# **n_jobs** : Number of cores used for the training process. If set to -1, all cores are used.
# 
# **n_estimators** : Number of classification trees in your learning model ( set to 10 per default)
# 
# **max_depth** : Maximum depth of tree, or how much a node should be expanded. Beware if set to too high  a number would run the risk of overfitting as one would be growing the tree too deep
# 
# **verbose** : Controls whether you want to output any text during the learning process. A value of 0 suppresses all text while a value of 3 outputs the tree learning process at every iteration.
# 
#  Please check out the full description via the official Sklearn website. There you will find that there are a whole host of other useful parameters that you can play around with. 

# <font color='blue'>
#     
# 완성도를 위해 여기 나열된 매개변수에 대해 간단한 요약을 제공한다. 
# 
# **n_jobs** : 학습 프로세스를 위한 코어의 개수. -1이면 모두 사용
# 
# **n_estimators** : 너의 러닝 모델 안 classfication tree의 개수(10개가 디폴트)
# 
# **max_depth** : 트리의 최대 깊이, 또는 얼마나 많은 노드가 확장될 수 있는지. 너무 높게 설정되면 트리가 너무 깊어지면서 오버피팅의 위험 발생
# 
# **verbose** : 러닝 프로세스 동안 텍스트를 출력하고 싶은지 컨트롤. 0은 모든 텍스트를 억제하고 3은 모든 iteration에서 트리 학습 프로세스를 출력한다.
# 
# Sklearn 공식 웹사이트에서 자세한 설명확인해라. 거기서 너는 니가 갖고 놀수 있는 다른 유용한 파라미터들의 전체의 호스트를 찾을 것이다. 
# </font> 

# In[ ]:


# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# Furthermore, since having mentioned about Objects and classes within the OOP framework, let us now create 5 objects that represent our 5 learning models via our Helper Sklearn Class we defined earlier.

# <font color='blue'>
# 더욱이, OOP 프레임워크 내에서 객체와 클래스에 대해 언급한 이후, 이전에 만든 Helper Sklearn 클래스를 통해 5개의 학습 모델을 의미하는 5개의 객체를 만들겠다.
# </font>

# In[ ]:


# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# **Creating NumPy arrays out of our train and test sets**
# 
# Great. Having prepared our first layer base models as such, we can now ready the training and test test data for input into our classifiers by generating NumPy arrays out of their original dataframes as follows:

# <font color='blue'>
# 
# **우리 학습 셋과 테스트 셋의 밖에서 NumPy 배열 만들기**
# 
# 좋아. 우리들의 1단계 베이스 모델들을 준비했다. 우리는 NumPy 배열을 생성함으로써 이제 학습과 테스트할 준비가 됐다. 
# 
# </font>

# In[ ]:


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data


# **Output of the First level Predictions** 
# 
# We now feed the training and test data into our 5 base classifiers and use the Out-of-Fold prediction function we defined earlier to generate our first level predictions. Allow a handful of minutes for the chunk of code below to run.

# <font color='blue'>
# 
# **1단계 예측 결과**
# 
# 5개의 베이스 classifier에 학습과 테스트 데이터를 제공하고 이전에 만든 Out-of-Fold 예측 함수를 사용해서 우리들의 첫번째 레벨 예측을 생성해라. 아래 코드 돌리는데 몇분 걸린다.
# 
# </font>

# In[ ]:


# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")


# **Feature importances generated from the different classifiers**
# 
# Now having learned our the first-level classifiers, we can utilise a very nifty feature of the Sklearn models and that is to output the importances of the various features in the training and test sets with one very simple line of code.
# 
# As per the Sklearn documentation, most of the classifiers are built in with an attribute which returns feature importances by simply typing in **.feature_importances_**. Therefore we will invoke this very useful attribute via our function earliand plot the feature importances as such

# <font color='blue'>
# 
# **다른 classifier에서 생성된 feature importance**
# 
# 이제 학습된 1단계 classifier를 가졌으니 우리는 Sklearn의 매우 멋진 기능을 활용할 수 있다. 이는 매우 간단한 코드 행 하나를 사용해서 훈련 및 테스트 셋에서 다양한 feature의 importance를 출력하는 것이다.
# 
# Sklearn doc에 따르면, 대부분의 classifer들이 feature importance를 뽑는 방법으로 **.feature_importances_** 를 타이핑하면 된다고 한다. 그러므로 우리는 이 매우 유용한 속성을 호출할 것이다. 
# 
# </font>

# In[ ]:


rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)


# So I have not yet figured out how to assign and store the feature importances outright. Therefore I'll print out the values from the code above and then simply copy and paste into Python lists as below (sorry for the lousy hack)

# <font color='blue'>
# 
# 나는 아직 feature importances를 어떻게 할당하고 저장하는지 이해하지 못했다. 그러므로 나는 위 코드의 아웃풋을 출력하고 복사 하여 아래 처럼 Python list로 붙여넣기할 것이다. (비열한 hack에 대해선 미안)
# 
# </font>

# In[ ]:


rf_features = [0.10474135,  0.21837029,  0.04432652,  0.02249159,  0.05432591,  0.02854371
  ,0.07570305,  0.01088129 , 0.24247496,  0.13685733 , 0.06128402]
et_features = [ 0.12165657,  0.37098307  ,0.03129623 , 0.01591611 , 0.05525811 , 0.028157
  ,0.04589793 , 0.02030357 , 0.17289562 , 0.04853517,  0.08910063]
ada_features = [0.028 ,   0.008  ,      0.012   ,     0.05866667,   0.032 ,       0.008
  ,0.04666667 ,  0.     ,      0.05733333,   0.73866667,   0.01066667]
gb_features = [ 0.06796144 , 0.03889349 , 0.07237845 , 0.02628645 , 0.11194395,  0.04778854
  ,0.05965792 , 0.02774745,  0.07462718,  0.4593142 ,  0.01340093]


# Create a dataframe from the lists containing the feature importance data for easy plotting via the Plotly package.

# <font color='blue'>
# 
# Plotly 패키지를 통해 쉽게 plotting 하도록 feature importance를 포함하는 이 리스트로부터 데이터 프레임을 생성해라
# 
# </font>

# In[ ]:


cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    })


# **Interactive feature importances via Plotly scatterplots**
# 
# I'll use the interactive Plotly package at this juncture to visualise the feature importances values of the different classifiers  via a plotly scatter plot by calling "Scatter" as follows:

# <font color='blue'>
# 
# 
# **Plotly scatterplot을 통한 feature importance 상호작용**
# 
# Plotly 패키지를 사용하여 Scatter를 호출하고 plot 분포도를 통해 여러 분류기준의 featureimportance를 시각화한다.
# 
# 
# </font>

# In[ ]:


# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# Now let us calculate the mean of all the feature importances and store it as a new column in the feature importance dataframe.

# <font color='blue'>
# 
# 이제 모든 feature importance의 평균을 내고, feature importance 데이터 프레임의 한 컬럼에 저장해라
# 
# </font>

# In[ ]:


# Create the new column containing the average of values

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe.head(3)


# **Plotly Barplot of Average Feature Importances**
# 
# Having obtained the mean feature importance across all our classifiers, we can plot them into a Plotly bar plot as follows:

# <font color='blue'>
# 
# 
# **평균 feature importance의 Plotly Barplot**
# 
# 
# 평균 feature importance를 우리들의 모든 classifier로부터 얻은 후, 우리는 그들을 Plotly bar plot에 그릴 수 있다.
# 
# </font>

# In[ ]:


y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')


# # Second-Level Predictions from the First-level Output

# **First-level output as new features**
# 
# Having now obtained our first-level predictions, one can think of it as essentially building a new set of features to be used as training data for the next classifier. As per the code below, we are therefore having as our new columns the first-level predictions from our earlier classifiers and we train the next classifier on this.

# <font color='blue'>
# 
# 
# **1단계 아웃풋을 새로운 feature로**
# 
# 
# 1단계 예측을 얻었으니, 다음 classifier를 위한 학습 데이터로 사용될 새로운 feature set을 필수적으로 만들어야겠다는 생각이 든다. 아래 코드처럼 우리는 새 컬럼을 가지게 했다. 
# 
# </font>

# In[ ]:


base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()


# **Correlation Heatmap of the Second Level Training set**

# In[ ]:


data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')


# There have been quite a few articles and Kaggle competition winner stories about the merits of having trained models that are more uncorrelated with one another producing better scores.

# <font color='blue'>
# 
# 학습된 모델들이 서로 관계가 없을 수록 좋은 점수를 내는 장점에 대한 몇몇 기사와 캐글 competition 우승자 이야기가 있다.
# 
# 
# 
# </font>
# 
# 
# 
# <font color='red'>
# 
# 
# 
# - ravel은 flattened array 반환하는 함수
# 
# </font>

# In[ ]:


x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


# Having now concatenated and joined both the first-level train and test predictions as x_train and x_test, we can now fit a second-level learning model.

# <font color='blue'>
# 
# 이제 x_train, x_test와 같이 1단계 훈련과 테스트 예측을 붙였으므로 우리는 이제 2단계 학습 모델을 fit 할 수 있다.
# 
# </font>

# ### Second level learning model via XGBoost
# 
# Here we choose the eXtremely famous library for boosted tree learning model, XGBoost. It was built to optimize large-scale boosted tree algorithms. For further information about the algorithm, check out the [official documentation][1].
# 
#   [1]: https://xgboost.readthedocs.io/en/latest/
# 
# Anyways, we call an XGBClassifier and fit it to the first-level train and target data and use the learned model to predict the test data as follows:

# <font color='blue'>
# 
# 
# **XGBoost를 이용한 2단계 학습모델 **
# 
# 
# 우리는 여기서 부스트된 tree 학습 모델을 위한 매우 유명한 라이브러리 XGBoost를 선택한다. large-scale boosted tree 알고리즘을 최적화 하기 위해 만들어졌다. 더 자세한 내용은 공식 문서 참조해라.
# 
# 어쨌든 우리는 XGBClassifier를 호출하고 1단계 학습,target 데이터를 fit하고 아래와 같이 테스트 데이터를 예측하기 위해 학습된 모델을 사용한다. 
# 
# </font>

# In[ ]:


gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


# Just a quick run down of the XGBoost parameters used in the model:
# 
# **max_depth** : How deep you want to grow your tree. Beware if set to too high a number might run the risk of overfitting.
# 
# **gamma** : minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be.
# 
# **eta** : step size shrinkage used in each boosting step to prevent overfitting

# <font color='blue'>
# 
# 모델에서 사용된 XGBoost 파라미터들을 빠르게 훑는다
# 
# **max_depth** : 트리를 얼마나 깊게 성장시킬지. 너무 높으면 오버피팅 가능성 있음.
# 
# **gamma** : 트리의 리프노드에서 추가 파티션을 만들기 위한 최소 loss 감소. 클수록 알고리즘은 보수적(?) 보존적(?)
# 
# **eta** : 오버 피팅을 방지하기 위해 각 부스팅 단계에서 사용되는 step size 축소.
# 
# </font>

# **Producing the Submission file**
# 
# Finally having trained and fit all our first-level and second-level models, we can now output the predictions into the proper format for submission to the Titanic competition as follows:

# <font color='blue'>
# 
# 
# **제출 파일 생성** 
# 
# 1단계와 2단계 모델 학습시키고 fit했으니, 우리는 이제 적절한 포맷에 맞춰 예측을 제출할 수 있다.
# 
# </font>

# In[ ]:


# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)


# **Steps for Further Improvement**
# 
# As a closing remark it must be noted that the steps taken above just show a very simple way of producing an ensemble stacker. You hear of ensembles created at the highest level of Kaggle competitions which involves monstrous combinations of stacked classifiers as well as levels of stacking which go to more than 2 levels. 
# 
# Some additional steps that may be taken to improve one's score could be:
# 
#  1. Implementing a good cross-validation strategy in training the models to find optimal parameter values
#  2. Introduce a greater variety of base models for learning. The more uncorrelated the results, the better the final score.

# <font color='blue'>
# 
# 
# **향상 시키기 위한 step** 
# 
# 마지막으로, 위 과정은 매우 단순한 앙상블 stacker  생성 과정이다. 너는 2단계 이상 stacking되고 괴물스러운 결합을 가지는 앙상블이 최고 수준 kaggle competition에서 만들어졌다는것을 들었다.
# 
# 스코어를 향상시키기 위한 추가적인 단계로는:
# 
# 1. 모델 학습시키면서 최적 파라미터 값을 찾기위한 좋은 cross-validation 전략을 구현해라.
# 2. 학습을 위한 다양한 기본 모델을 소개해라. 결과가 더 관련성이 없을 수록 최종 점수가 높아진다. 
# 
# </font>

# ### Conclusion
# 
# I have this notebook has been helpful somewhat in introducing a working script for stacking learning models. Again credit must be extended to Faron and Sina. 
# 
# For other excellent material on stacking or ensembling in general, refer to the de-facto Must read article on the website MLWave: [Kaggle Ensembling Guide][1]. 
# 
# Till next time, Peace Out
# 
#   [1]: http://mlwave.com/kaggle-ensembling-guide/

# <font color='blue'>
# 
# 
# **결론** 
# 
# 나는 이 노트북이 stacking 학습 모델을 위한 스크립트를 소개하는데 도움이 되면 좋겠다. 다시 신용은 Faron과 Sina에게 확장되야 한다.
# 
# stacking과 앙상블링을 위한 다른 훌륭한 자료는, 사실상 무조건 읽어야 하는 웹사이트 기사 MLWave Kaggle Ensembling Guide.를 참고해라.
# 
# 다음시간까지 피스아웃
# 
# </font>

# <font color='red'>
# 
# - 세미나에서 'k-fold cross validation이 중요하다', 'stacking, ensemble로 점수를 한계까지 끌어올린다', 'XGBoost라는 라이브러리를 많이 사용한다'는 등의 얘기들을 들었는데 이번 커널에서 전부 코드로 볼 수 있었음.
# 
# - 다른 사람들이 feature engineering 어떻게 하는지 알 수 있었음.
# 
# 
# </font>
