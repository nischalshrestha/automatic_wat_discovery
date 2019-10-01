#!/usr/bin/env python
# coding: utf-8

# この[Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)を参考にしています

# In[ ]:


import numpy as np
import pandas as pd

# 可視化
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# 機械学習ライブラリ
from sklearn.linear_model import LogisticRegression #ロジスティック回帰
from sklearn.svm import SVC, LinearSVC # サポートベクターマシン
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from sklearn.neighbors import KNeighborsClassifier # K近傍法
from sklearn.naive_bayes import GaussianNB # 単純ベイズ分類器
from sklearn.linear_model import Perceptron # パーセプトロン
from sklearn.linear_model import SGDClassifier # 確率的勾配降下法
from sklearn.tree import DecisionTreeClassifier # 決定木


# In[ ]:


# 入力
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ### 仮定に基づくデータ解析
# #### 相関の発見
# - どの特徴量が生存と相関関係にあるか
# - 早期にデータ間の相関を発見し，モデル化した相関と一致させたい
# 
# #### 欠損データの補完
# - 年齢(Age)は生存と明らかに相関するので，補完する
# - 乗船港(Embarked)は生存あるいは他の重要な特徴量と相関するので，補完する
# 
# #### データの修正
# - 乗船券(Ticket)は分析においてデータの重複が多く，また，生存と相関しないので除去する
# - 客室(Cabin)は訓練・評価データともに欠損が多いため，除去する
# - 乗客ID(PassengerId)は生存に影響しないため，除去する
# - 氏名(Name)の特徴量は一般的でなく，直接生存に影響しないため，除去する．
# 
# #### 生成
# - 配偶者(Parch)や兄弟(SibSp)を合計し，新しい特徴量の"家族(Family)"として生成する
# - 名前(Name)を取り除き，新しい特徴量として設計する
# - 年代(Age bands)のための新しい特徴量を作成する．連続した数値的な特徴を順序分類の特徴に変換する．
# - 運賃(Fare)を分析のために活用したい
# 
# #### 分類
# - 女性(Sex=femare)は生存した可能性が高い
# - 子ども(Age<$n$)は生存した可能性が高い
# - 富裕層(Pclass=1)は生存した可能性が高い

# In[ ]:


### 訓練データの生存(Survived)との相関をみるため，各属性の平均値を算出する
# 所得
pclass_survived_corr = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# 性別
sex_survived_corr = train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# 兄弟数
sibsp_survived_corr = train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# 兄弟数
parch_survived_corr = train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
display(pclass_survived_corr, sex_survived_corr, sibsp_survived_corr, parch_survived_corr)


# In[ ]:


# データセットから乗船券と客室特徴の削除
train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]


# In[ ]:


# 名前特徴から敬称の抽出
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


# 敬称ごとの平均生存率
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


# 敬称を構造データへ変換
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[ ]:


# データセットから名前と乗客IDの削除
train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]


# In[ ]:


# 性別特徴の構造化
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)

train.head()


# In[ ]:


### それぞれの性別と階級の組み合わせごとの年齢の中央値を欠損補完に利用
guess_ages = np.zeros((2, 3))
for dataset in combine:
    # 性別と階級の組み合わせごとの年齢の中央値を算出
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess/0.5+0.5)*0.5
            
    # 年齢の欠損を中央値で埋める
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i, j]
    # 整数にキャスト
    dataset['Age'] = dataset['Age'].astype(int)


# In[ ]:


### 年齢特徴を5分割した年齢層と生存の相関をみる
train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


### 年齢を年齢層ごとにラベリング
for dataset in combine:    
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
train = train.drop(['AgeBand'], axis=1)
combine = [train, test]


# In[ ]:


### 兄弟と配偶者から家族サイズ特徴の生成
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # 家族サイズ = 1に関しては，特にIsAloneとして特徴の生成
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train, test]
# train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:


### 年齢と階級を掛けた新たな特徴の生成
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train.loc[:, ['Age*Class', 'Age', 'Pclass']].head()


# In[ ]:


### 乗船港の欠損には頻出する値を使用
freq_port = train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
# train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


### 運賃の欠損には中央値を使用し，構造化する
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

combine = [train, test]
    
train.head(10)


# In[ ]:


X_train = train.drop('Survived', axis=1)
Y_train = train['Survived']
X_test = test.drop('PassengerId', axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


### ロジスティック回帰
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
# 精度を小数点第2位まで算出
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# 特徴と生存の相関係数を算出
coeff = pd.DataFrame(train.columns.delete(0)) # ID以外のテーブルを取得
coeff.columns = ['Feature']
coeff['Correlation'] = pd.Series(logreg.coef_[0])
coeff.sort_values(by='Correlation', ascending=False)


# In[ ]:


### サポートベクターマシン
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


### K近傍法
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


### ガウシアン分布ナイーブベイズ
gaus = GaussianNB()
gaus.fit(X_train, Y_train)
Y_pred = gaus.predict(X_test)
acc_gaus = round(gaus.score(X_train, Y_train) * 100, 2)
acc_gaus


# In[ ]:


### パーセプトロン
perp = Perceptron()
perp.fit(X_train, Y_train)
Y_pred = perp.predict(X_test)
acc_perp = round(perp.score(X_train, Y_train) * 100, 2)
acc_perp


# In[ ]:


### 線形SVMクラス分類
lsvc = LinearSVC()
lsvc.fit(X_train, Y_train)
Y_pred = lsvc.predict(X_test)
acc_lsvc = round(lsvc.score(X_train, Y_train) * 100, 2)
acc_lsvc


# In[ ]:


### 確率的勾配降下法
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


### 決定木
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)
acc_dt = round(dt.score(X_train, Y_train) * 100, 2)
acc_dt


# In[ ]:


### ランダムフォレスト
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
acc_rf = round(rf.score(X_train, Y_train) * 100, 2)
acc_rf


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_rf, acc_gaus, acc_perp, 
              acc_sgd, acc_lsvc, acc_dt]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": Y_pred
})
submission.to_csv('submission.csv', index=False)


# In[ ]:




