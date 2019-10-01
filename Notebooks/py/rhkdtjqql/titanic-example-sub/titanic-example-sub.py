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


#데이터 전처리를 위한 패키지 호춣
# data analysis and wrangling
#데이터 분석, 핸들링 작업
#데이터 랭글링(Data Wrangling) 혹은 데이터 먼징(Data Munging)은 원자료(raw data)를 또다른 형태로 수작업으로 전환하거나 매핑하는 과정이다. 
#이를 통해서 반자동화 도구의 도움으로 데이터를 좀더 편리하게 소비한다. 데이터 랭글링에는 먼징(munging), 데이터 시각화, 데이터 집합, 통계모형 학습 뿐만 아니라 많은 다른 잠재적 용도도 포함된다.
#일반적으로 데이터 먼징은 일반적인 단계를 따르는데 데이터 원천(Data Source)으로부터 원래 최초 형태로 자료를 추출하는 것으로 시작한다. 
#알고리듬(예로, 정렬)을 사용해서 원자료를 "먼징(munging"하거나 사전 정의된 자료구조로 데이터를 파싱(parsing)한다. 
#그리고 나서 마지막으로 저장이나 미래 사용을 위해서 작업완료한 콘텐츠를 데이터 싱크(data sink)에 놓아둔다. 
#인터넷의 급격한 확산으로 이러한 기술이 가용한 데이터 양이 증가하고 있는 기관에서는 점점 중요해지고 있다.
#데이터 랭글러(Data Wrangler)는 랭글링을 수행하는 사람이다.
#출처 위키피티아


import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:





# In[ ]:


#데이터 호출은 Kaggle의 데이터 호출 기능을 사용하도록 한다
#호출을 사용하여 부른 데이터 셋은 아래와 같이 사용하도록 한다.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# In[ ]:


print(train_df.columns)


# In[ ]:


#판다스를 사용하여 데이터 탐색을 한다.
#데이터에 대한 설명은 다음의 링크를 참조하도록 한다.
#https://www.kaggle.com/c/titanic/data
#데이터셋의 컬럼명 확인
print(train_df.columns.values)
print(train_df.columns)
#head를 통한 변수들의 속성 파악
train_df.head()
train_df.values
#데이터 타입 확인
train_df.dtypes

'''PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object'''
#Which features are categorical?

#These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based? 
#Among other things this helps us select the appropriate plots for visualization.

#Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
#카테고리칼  데이터
#Which features are numerical?

#Which features are numerical? These values change from sample to sample. 
#Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.

#Continous: Age, Fare. Discrete: SibSp, Parch.



#Which features may contain errors or typos?
#에러나 오타를 가진 데이터

#This is harder to review for a large dataset, however reviewing a few samples from a smaller dataset may just tell us outright, which features may require correcting.
#데이터 전체를 보기 보다 샘플 데이터를 통해 수정이 필요한 특성을 볼 수 있음

#Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.
#이름 속성이 에러나 오타를 포함하고 있다. 따옴표,제목,괄혹,
train_df.tail()


# In[ ]:



#Which features contain blank, null or empty values?

#These will require correcting.

#Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
#Cabin > Age are incomplete in case of test dataset.
#What are the data types for various features?

#Helping us during converting goal.

#Seven features are integer or floats. Six in case of test dataset.
#Five features are strings (object).

#null값 확인
#Cabin > Age > Embarked가 트레이닝셋에서 null값 포함
#총 891개의 행에서 Cabin > Age > Embarked는 각각 204,714,889개의 데이터만 가지고 있음
train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


#What is the distribution of numerical feature values across the samples?
#샘플에서의 숫자형 데이터의 분포 확인
#This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.
#초기 탐색에서 트레이닝 데이터 셋이 실제 데이터에 대해 얼마나 대표하는지에 대해 알 수 있도록 해줌
#Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
#샘플은 전체는 2224개 중 40%인 891개의 승객 데이터를 가지고 있다.
#Survived is a categorical feature with 0 or 1 values.
#생존 여부는 1,0으로 표현
#Around 38% samples survived representative of the actual survival rate at 32%.
#샘플의 생존률은 38%이고 실제 생존률 32%이다 
#Most passengers (> 75%) did not travel with parents or children.
#70% 승객은 부모나 자녀와 동행하지 않음
#Nearly 30% of the passengers had siblings and/or spouse aboard.
#약 30%의 승객은 형제나 배우자와 동행
#Fares varied significantly with few passengers (<1%) paying as high as $512.
#요금은 매우 다양했으면 약 1% 미만의 승객 요금은 512 달러 정도로 높아다
#Few elderly passengers (<1%) within age range 65-80.
#약 1%의 승객은 65~80세의 나이를 가졌다.

#train_df.describe()
train_df.describe(percentiles=[.61, .62])
#train_df.describe(percentiles=[.75, .8])
#분위수를 고를 수 있음
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
#####잘모르겠음


# In[ ]:


#What is the distribution of categorical features?

#Names are unique across the dataset (count=unique=891)
#Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
#Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
#Embarked takes three possible values. S port used by most passengers (top=S)
#Ticket feature has high ratio (22%) of duplicate values (unique=681).
train_df.describe(include=['O'])
#카테고리 데이터만 뽑아서 보여줌
train_df.describe(include='all')
#모든 데이터에 대해서 보여줌
#None to both (default). The result will include only numeric-typed columns or, if none are, only categorical columns.
train_df.describe()
#기본은 오직 숫자형만 보여주고 숫자형 없으면 문장형도 보여줌

#A list of dtypes or strings to be included/excluded. To select all numeric types use numpy numpy.number. 
#To select categorical objects use type object. See also the select_dtypes documentation. eg. df.describe(include=[‘O’])
#If include is the string ‘all’, the output column-set will match the input one.


# In[ ]:


#Assumtions based on data analysis
#We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.
#기존에 수행된 데이터 분석에 기반항 가정을 하고 이 적절한 조치를 하기전 가정을 검증할것이다. 

#Correlating.
#상관관계
#We want to know how well does each feature correlate with Survival.
#우린는 각 변수와 생존과의 상관관계에 대해 알고 싶음
#We want to do this early in our project and match these quick correlations with modelled correlations later in the project.
#우리는 프로젝트 초기에 이것에 대해 조사하고 나중에 모델로 된 상관관계와 매치시킬것이다.

#Completing.
#완료하기(?)
#We may want to complete Age feature as it is definitely correlated to survival.
#나이가 생존과 상관관계가 있음
#We may want to complete the Embarked feature as it may also correlate with survival or another important feature.
#embarked가 생존이나 다른 중요한 변수와 상관관계가 있다고 봄


#Correcting.

#Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
#티켓 변수는 높은 비율(22%)의 중복을 가지고 있음으로 우리의 분석에서 제외 될 것이며 티켓과 생존과의 상관관계도 없는 것으로 보임( 근거??)
#Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
#cabin번수도 트레이닝 데이터셋과 테스트 데이터 셋 모두에서 너무 많은 null과 제대로 작성되지 않은 비율이 높아 제외됨
#PassengerId may be dropped from training dataset as it does not contribute to survival.
#승객 아이디변수는 생존에 크게 작용하지 않기 때문에 제외한다. 
#Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.
#이름은 상대적으로 비표준화된 변수로서 생존에 직접적으로 상관이 없음으로 제외한다.

#Creating.

#We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
#Parch, SibSp 변수를 통해 총 가족수를 나타내는 가족이라는 새로운 변수를 생성하도록 한다.
#We may want to engineer the Name feature to extract Title as a new feature.
#이름 변수에서 title(칭호,작위,직함)을 추출하여 새로운 변수로 생성하도록 한다.
#We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
#나이를 통해  순서형 범주형 변수인 Age bands 새로이 들도록 한다.
#We may also want to create a Fare range feature if it helps our analysis.
#분석에 용이하도록  Fare range라는 변수를 새로 생성한다.

#Classifying.

#We may also add to our assumptions based on the problem description noted earlier.
#초기를 데이터 탐색을 기반으로 가정을 추가함
#Women (Sex=female) were more likely to have survived.
#여성의 생존률이 매우 높다
#Children (Age<?) were more likely to have survived.
#어린이의 생존율이 매우 높다
#The upper-class passengers (Pclass=1) were more likely to have survived.
#상위 등급의 승객이 생존율이 높다.


# In[ ]:


#Analyze by pivoting features
#피봇팅을 통한 분석

#To confirm some of our observations and assumptions, 
#we can quickly analyze our feature correlations by pivoting features against each other. 
#We can only do so at this stage for features which do not have any empty values. 
#위 작업은 널값이나 빈값이 없는 데이터에 대해서만 피보팅 가능
#It also makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.
#범주형 번수나 순서형,이산형 변수에 대해서만 위 작업이 의미가 잆다

#Pclass We observe significant correlation (>0.5) among Pclass=1 and Survived (classifying #3). 
#We decide to include this feature in our model.
#좌석등급이 1인 것과 생존과는 0.5이상의 상관관계가 있는 것을 관찰했음으로 좌석등금은 모델에 포함
#Sex We confirm the observation during problem definition that Sex=female had very high survival rate at 74% (classifying #1).
#성별도 마찬가지로 여성이 매우 생존율이 높다.
#SibSp and Parch These features have zero correlation for certain values.
#SibSp and Parch는 특정 값에 대해 상관관계가 없다.
#It may be best to derive a feature or a set of features from these individual features (creating #1).
#SibSp and Parch를 통해 통해 파생변수를 만들거나 변수 집합을 만드는 것이 최선이다.


# In[ ]:


#좌석등급과 생존과의 관계 
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


#Analyze by visualizing data
#Now we can continue confirming some of our assumptions using visualizations for analyzing the data.
#데이터 시각화 
#분석을 위한 시각화를 통해 가정을 이어감


#Correlating numerical features
#Let us start by understanding correlations between numerical features and our solution goal (Survived).
#숫자형 변수와 생존과의 상관관계

#A histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. 
#The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands.
#This helps us answer questions relating to specific bands (Did infants have better survival rate?)
#구간화된 나이와 같은 숫자형 변수 분석에는 히스토그램이 패턴을 분석하는 효과적임
#Note that x-axis in historgram visualizations represents the count of samples or passengers.
#히스토그램 시각화에서 x축은 샘플이나 승객의 수를 뜻함

#Observations.
#관측

#Infants (Age <=4) had high survival rate.
#Oldest passengers (Age = 80) survived.
#Large number of 15-25 year olds did not survive.
#Most passengers are in 15-35 age range.


#Decisions.


#This simple analysis confirms our assumptions as decisions for subsequent workflow stages.
#이작업은 후속 작업을 가정 선택하는데 도움을 준다. 
#We should consider Age (our assumption classifying #2) in our model training.
#나이를 변수로 고려
#Complete the Age feature for null values (completing #1).
#나이에 널값을 채워라
#We should band age groups (creating #3).
# 나이를 구간화해라


# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
#구간을 20씩 잡고 나이-생존율 히스토그램 생성


# In[ ]:


#Correlating numerical and ordinal features
#We can combine multiple features for identifying correlations using a single plot. 
#This can be done with numerical and categorical features which have numeric values.
#숫자형 변수와 순서형 변수의 상관관계 분석
#한개의 그래프에서 상관관계 분석을 위해서 여러개의 변수를 동시에 나타낼 수 있다.
#이것은 숫자형 변수와 숫자값을 포함하는 범주형간에 작업을 할 수 있다.

#Observations.
#관측
#Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
#대부분의 승객은 3등급 승객이지만 대부분 생존하지 못함. 2번가정에 대한 근거
#Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
#2,3등급에 있는 유아는 대부분 생존 가정2에 대한 강화
#Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
#1등급 승객 대부분 생존 가정3 강화

#Pclass varies in terms of Age distribution of passengers.
#좌석등급은 다양한 나이 분포를 가짐

#Decisions.

#Consider Pclass for model training.
#모델 학습에 좌석등급 고려


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
# 좌석등급/생존율에 따른 나이분포
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:


#Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.
#성별과 좌석등급을 기반의 나이값 추축을 위한 빈  Array 생성
guess_ages = np.zeros((2,3))
guess_ages
#Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.
for dataset in combine: #combine에 있는 데이터셋 마다
    for i in range(0, 2):#0,1 성별이 0,1
        for j in range(0, 3):#0,1,2 등급이 1,2,3-> 파이썬은 0부터 시작임으로 0,1,2 로 표현
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()       
            #성별과 좌석 등급에 따라 루프를 돌면서 NA값을 빼고 guess_df에 나이값 넣기
    
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()
            #guess_df의 중위값을 age_guess에 넣음

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int(age_guess/0.5 + 0.5)*0.5
            #age_guess 0 ~ .25 버림/ .25 ~ .75 -> 0.5  / .75~1.0 ->올림
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]
    #루프를 돌면 나의 공백에 나이 중위값을 넣어줌
    dataset['Age'] = dataset['Age'].astype(int)
#저장된 나이는 인트로 형변환
train_df.head()


# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# In[ ]:


###################최신 커넬에서 다시 참고
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[ ]:



train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

#최신 커널에서 수정사항 반영 필요



# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5) #5단위로 자르기
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()


# In[ ]:


#Create new feature combining existing features
#We can create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:


#Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.


# In[ ]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[ ]:


Completing a categorical feature
Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance.


# In[ ]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


Converting categorical feature to numeric
We can now convert the EmbarkedFill feature by creating a new numeric Port feature.


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# In[ ]:


Quick completing and converting a numeric feature
We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently for this feature. We do this in a single line of code.

Note that we are not creating an intermediate new feature or doing any further analysis for correlation to guess missing feature as we are replacing only a single value. The completion goal achieves desired requirement for model algorithm to operate on non-null values.

We may also want round off the fare to two decimals as it represents currency.


# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[ ]:


We can not create FareBand


# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# In[ ]:


test_df.head(10)


# In[ ]:


Model, predict and solve
Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:

Logistic Regression
KNN or k-Nearest Neighbors
Support Vector Machines
Naive Bayes classifier
Decision Tree
Random Forrest
Perceptron
Artificial neural network
RVM or Relevance Vector Machine


# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. Reference Wikipedia.

Note the confidence score generated by the model based on our training datase


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.

Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
Inversely as Pclass increases, probability of Survived=1 decreases the most.
This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
So is Title as second highest positive correlation.


# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. Reference Wikipedia.

Note that the model generates a confidence score which is higher than Logistics Regression model.


# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. Reference Wikipedia.

KNN confidence score is better than Logistics Regression but worse than SVM.


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem. Reference Wikipedia.

The model generated confidence score is the lowest among the models evaluated so far.


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time. Reference Wikipedia.


# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Reference Wikipedia.

The model confidence score is the highest among models evaluated so far.


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


The next model Random Forests is one of the most popular. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Reference Wikipedia.

The model confidence score is the highest among models evaluated so far. We decide to use this model's output (Y_pred) for creating our competition submission of results.


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


Model evaluation
We can now rank our evaluation of all the models to choose the best one for our problem. While both Decision Tree and Random Forest score the same, we choose to use Random Forest as they correct for decision trees' habit of overfitting to their training set.


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:


# submission.to_csv('../output/submission.csv', index=False)


# 

# 

# 
