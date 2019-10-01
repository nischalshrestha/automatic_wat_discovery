#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# <h3> Predict survival on the Titanic </h3>

# <h3>Problem analysis</h3>
# 
# <p> In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.</p><br />
# <p> RMS 타이타닉 침몰 사고 시 구명정이 탑승 승객보다 적었기 때문에 일부 사람들만 생존할 수 있었다.<br />
# 승객 중 어떤 그룹의 사람들은 다른 그룹들보다 생존할 확률이 높았다. ex)여성, 아이들, 상류층<br />
# 타이타닉호 탑승 승객 데이터를 학습하여 탑승자 정보가 들어왔을 때 생존 혹은 사망할 확률을 예측한다.</p>

# <h4>- Load data and analysis</h4>

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import seaborn as sns
sns.set() # setting seaborn default for plots

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# <h3>Data Dictionary</h3>
# <br/>
# 
# - survived : 0 = No(Death), 1 = Yes(Alive)
# - pclass : Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd (1 > 2 > 3)
# - sibsp : siblings / spouses (형제자매, 배우자가 함께 탑승한 수) (혼자면 0)
# - parch : parent / children (부모님, 자식이 함께 탑승한 수) (혼자면 0)
# - cabin : 객실 번호
# - embarked : 탑승 선착장 C = Cherbourg, Q = Queenstown, S = Southampton
# <br /><br />
# <p>train data는 12개의 열을 갖고 test data는 예측하려는 Survived 열은 제거되어 11개의 열을 갖는다.</p>

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train.corr()


# In[ ]:


plt.figure(figsize=(8, 8))
sns.heatmap(train.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)
plt.show()


# <h3>Lost Information</h3>
# <br/>
# 
# <p>
#     dataframe.info()를 통해 Age, Cabin이 비어있는 것을 알 수 있음 (train에서 Embarked도 빠져있음)<br />
#     dataframe.isnull().sum()을 통해서 null인 값이 몇행인지 확인 가능
# </p>

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# <h4>- Data visualization</h4>

# In[ ]:


# 산 사람과 죽은 사람의 각 피쳐 특징 확인
def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts() # Survived 값이 1인 행들의 feature열 수 count
    dead = train[train['Survived'] == 0][feature].value_counts() # Survived 값이 0인 행들의 feature열 수 count
    df = pd.DataFrame([survived, dead]) # 산사람과 죽은사람으로 나누어서 DataFrame으로 저장
    df.index = ['Survived', 'Dead'] # index명 지정
    df.plot(kind='bar', stacked=True, figsize=(10, 5)) # bar chart 그리기
    plt.show()


# In[ ]:


bar_chart('Sex')


# 여자가 남자보다 생존할 가능성이 높다.

# In[ ]:


bar_chart('Pclass')


# 1등급 석의 사람은 생존 확률이 높고 3등급 석의 사람은 생존 확률이 낮다.

# In[ ]:


bar_chart('SibSp')


# 형제자매 없이 혼자 탔을 경우 생존 확률이 낮다.

# In[ ]:


bar_chart('Parch')


# 부모자식 없이 혼자 탔을 경우 생존 확률이 낮다.

# In[ ]:


bar_chart('Embarked')


# Queenstown에서 탑승한 승객이 생존 확률이 낮다

# <h2>Feature Engineering</h2>
# <p>feature들을 vector로 만드는 과정 <br/>
# data를 숫자로 만들어주는 것 <br/>
# NaN = Not a Number</p>

# In[ ]:


train.head(10)


# <h3>1. Name (English honorifics)</h3>
# <p>자식 여부, 계급 등을 알 수 있는 지표일 것이라 가정</p>

# In[ ]:


train_test_data = [train, test] # train data와 test data 결합

# train_test_data의 Name필드에서 Title을 뽑음
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-za-z]+)\.', expand=False)


# In[ ]:


# train data의 타이틀 종류 및 인원 확인
train['Title'].value_counts()


# In[ ]:


# test data의 타이틀 종류 및 인원 확인
test['Title'].value_counts()


# In[ ]:


# 타이틀별로 Mr는 0, Miss는 1, Mrs는 2, 그 외 나머지는 3으로 매핑

title_mapping = {"Mr":0, "Miss":1, "Mrs":2,
                 "Master":3, "Dr":3, "Rev":3, "Major":3, "Col":3, "Mlle":3, "Lady":3,
                 "Mme":3, "Sir":3, "Jonkheer":3, "Ms":3, "Capt":3, "Don":3, "Dona":3,
                "Countess":3}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


train.head(10)


# In[ ]:


bar_chart('Title')


# Mr 남성은 생존 확률이 적고, Miss, Mrs 여성은 생존 확률이 높음. (더이상 Name 필드는 필요가 없으므로 피쳐 삭제)

# In[ ]:


# 데이터셋 중 필요없는 피쳐 삭제
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[ ]:


train.head()


# <h3>2. Sex</h3>
# <p> vector로 변경 </p>

# In[ ]:


sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


bar_chart('Sex')


# <h3>3. Age</h3>
# <p>NaN인 데이터가 포함되어 있기에 주의해야 한다.<br/>
# (Missing Data)<br/><br/>
# <h4>방법 1. NaN 이외 전체 탑승객들의 나이의 평균 혹은 중간값을 대입</h4></p>
# 
# - NaN인 데이터의 성별(타이틀)을 나눠 성별별 탑승객들의 나이의 중간값을 대입

# In[ ]:


# Age 필드의 NaN값을 Title 그룹별의 나이 중간값으로 채움
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


# train data의 나이에 따른 생사 확인
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()


# In[ ]:


# 나이대별 생사 확인
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(0, 20)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(20, 30)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(30, 40)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(40, 60)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(60)


# In[ ]:


# 나이대에 따라 그룹 나눔
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[dataset['Age'] > 62, 'Age'] = 4


# <p>Child : 0<br/>
# Young : 1<br/>
# Adult : 2<br/>
# mid-age : 3<br/>
# senior : 4
# </p>

# In[ ]:


train.head()


# In[ ]:


bar_chart('Age')


# <h3>4.Embarked</h3>

# In[ ]:


Pclass1 = train[train['Pclass'] == 1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass'] == 2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass'] == 3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10, 5))


# Southampton에서의 탑승객이 모든 등급 좌석에서 50% 이상을 차지함.<br/>
# 따라서 NaN 데이터를 S로 채운다.

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


train.head(10)


# In[ ]:


embarked_mapping = {"S":0, "C":1, "Q":2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# <h3>5. Fare</h3>

# In[ ]:


# 등급별 중간값을 NaN값에 넣어줌
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()

plt.show()


# In[ ]:


for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 2,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 4,
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 6


# In[ ]:


train.head()


# <h3>6. Cabin</h3>

# In[ ]:


train.Cabin.value_counts()


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind = 'bar', stacked=True, figsize=(10, 5))


# In[ ]:


# scaling
# 머신 러닝 모델은 값의 차이가 클 수록 더 큰 의미를 부여하기 때문에 값을 스케일링 해줌
cabin_mapping = {"A":0, "B":0.7, "C":1.4, "D":2.1, "E":2.8, "F":3.5, "G":4.2, "T":4.9}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# In[ ]:


train.head()


# <h3>7. FamilySize</h3>

# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'FamilySize', shade=True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


family_mapping = {1:0, 2:0.5, 3:1.0, 4:1.5, 5:2.0, 6:2.5, 7:3.0, 8:3.5, 9:4.0, 10:4.5, 11:5.0}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[ ]:


# import numpy as np
# corr = train.corr()
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# f, ax = plt.subplots(figsize=(11, 9))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


train.head()


# In[ ]:


# 필요없는 항목 drop
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[ ]:


# 모델 입력데이터 구성을 위한 train_data 셋 구성
#train_data = 입력 , target = 출력
train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape


# In[ ]:


train_data.head(10)


# In[ ]:


test = test.drop(['PassengerId'], axis=1)


# In[ ]:


test.head(10)


# In[ ]:


plt.figure(figsize=(8, 8))
sns.heatmap(train.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)
plt.show()


# In[ ]:


import tensorflow as tf
train_x = train_data
df = pd.DataFrame(target) # 산사람과 죽은사람으로 나누어서 DataFrame으로 저장
df.columns = ['Survived'] # index명 지정
train_y = df
test_x = test
test_y = pd.read_csv('../input/gender_submission.csv')
df = pd.DataFrame(test_y['Survived'])
df.columns = ['Survived']
test_y = df


# In[ ]:


X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.get_variable("W", shape=[8, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]), name='bias')
H = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y*tf.log(H) + (1-Y) * tf.log(1-H))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

predicted = tf.cast(H > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


# In[ ]:


import time
startTime = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:train_x, Y:train_y})
        if step % 1000 == 0:
            print(step, cost_val)
                  
    print('-----------------------------')
    print('train_data = ', len(train_x), 'test_data = ', len(test_x))

    h, c, y, a = sess.run([H, predicted, Y, accuracy], feed_dict={X:test_x, Y:test_y})
    print('\n Predicted: ', c, '\nCorrect (Y): ', y, '\nAccuracy: ', a)
    
#     h, c = sess.run([H, predicted], feed_dict={X:test_x, Y:test_y})
#     print('test_x = ', test_x, ', predicted = ', c)
endTime = time.time()
print(endTime - startTime, " sec")


# ---------------------------------------------------------------------------------------------

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#Logistic Regression 모델

logreg = LogisticRegression()
logreg.fit(train_x, train_y)
Y_pred = logreg.predict(test_x)
acc_log = round(logreg.score(train_x, train_y) * 100 , 2)
acc_log


# In[ ]:


# Support Vector Machines 모델

svc = SVC()
svc.fit(train_x, train_y)
Y_pred = svc.predict(test_x)
acc_svc = round(svc.score(train_x, train_y) * 100, 2)
acc_svc


# In[ ]:


#K Neighbors Classifier 모델

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_x, train_y)
Y_pred = knn.predict(test_x)
acc_knn = round(knn.score(train_x, train_y) * 100, 2)
acc_knn


# In[ ]:


# Gaussian Naive Bayes 모델

gaussian = GaussianNB()
gaussian.fit(train_x, train_y)
Y_pred = gaussian.predict(test_x)
acc_gaussian = round(gaussian.score(train_x, train_y) * 100, 2)
acc_gaussian


# In[ ]:


# Perceptron 모델

perceptron = Perceptron()
perceptron.fit(train_x, train_y)
Y_pred = perceptron.predict(test_x)
acc_perceptron = round(perceptron.score(train_x, train_y) * 100, 2)
acc_perceptron


# In[ ]:


# Linear SVC 모델
linear_svc = LinearSVC()
linear_svc.fit(train_x, train_y)
Y_pred = linear_svc.predict(test_x)
acc_linear_svc = round(linear_svc.score(train_x, train_y) * 100, 2)
acc_linear_svc


# In[ ]:


# Stochastic Gradient Descent 모델

sgd = SGDClassifier()
sgd.fit(train_x, train_y)
Y_pred = sgd.predict(test_x)
acc_sgd = round(sgd.score(train_x, train_y) * 100, 2)
acc_sgd


# In[ ]:


# Decision Tree 모델

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_x, train_y)
Y_pred = decision_tree.predict(test_x)
acc_decision_tree = round(decision_tree.score(train_x, train_y) * 100, 2)
acc_decision_tree


# In[ ]:


# Random Forest 모델

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_x, train_y)
Y_pred = random_forest.predict(test_x)
acc_random_forest = round(random_forest.score(train_x, train_y) * 100, 2)
acc_random_forest


# In[ ]:


models = pd.DataFrame({
    'Model' : ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree'],
    'Score' : [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree]
})
models.sort_values(by='Score', ascending=False)


# In[ ]:


# 예측 결과 csv로 저장

test = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
    "PassengerId":test["PassengerId"],
    "Survived":decision_tree.predict(test_x)
})


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head()


# In[ ]:




