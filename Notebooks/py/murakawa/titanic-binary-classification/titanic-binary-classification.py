#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# ## データフレームの作成

# In[ ]:


# pandasでデータの読み込みを実施
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# データの概要を確認（トレーニング）
train.head(10)


# In[ ]:


# データの概要を確認（テスト）
test.head(10)


# 上記のカラムの意味は以下：<br>1. PasesngerId：乗客を一意に表すID<br>2.Survived：生きて脱出したか、亡くなったか<br>
# 3.Pclass：チケットのランク（1 = 1st, 2 = 2nd, 3 = 3rd）<br>
# 4.Name：氏名<br>
# 5.Sex：性別<br>
# 6.Age：年齢<br>
# 7.SibSp：兄弟や配偶者の数<br>
# 7.Parch：子供の数<br>
# 8.Fare：運賃<br>
# 9.Cabin：キャビン番号<br>
# 10.Embarked：港湾 (C = Cherbourg, Q = Queenstown, S = Southampton)

# ## 欠損値の処理
# ### トレーニングデータの確認

# In[ ]:


# 欠損値の確認
have_nan_cols = []
for column in train.columns:
    print(column, 'have nan values', train[column].isna().sum())
    if train[column].isna().sum() != 0:
        have_nan_cols.append(column)


# In[ ]:


# 年齢を一意に表示。身元不明や取得忘れの可能性が高い。
train.Age.unique()


# In[ ]:


# 年齢全体の平均値で補う
train.Age.fillna(train.Age.mean(), inplace=True)


# In[ ]:


# キャビン番号を一意に表示。不明であった可能性が高い。
train.Cabin.unique()


# In[ ]:


# 最頻値で補う
train.Cabin.fillna(train.Cabin.mode()[0], inplace=True)


# In[ ]:


# 港を一意に表示。不明であった可能性が高い。
train.Embarked.unique()


# In[ ]:


# 最頻値で補う
train.Embarked.fillna(train.Embarked.mode()[0], inplace=True)


# In[ ]:


train.head()


# ### テストデータの確認

# In[ ]:


have_nan_cols = []
for column in test.columns:
    print(column, 'have nan values', test[column].isna().sum())
    if test[column].isna().sum() != 0:
        have_nan_cols.append(column)


# In[ ]:


test.Age.fillna(train.Age.mean(), inplace=True)
test.Cabin.fillna(train.Cabin.mode()[0], inplace=True)
test.Fare.fillna(train.Fare.mean(), inplace=True)


# In[ ]:


test.head()


# ## 特徴選択

# In[ ]:


# 確実に不要と思われる特徴を削除。ここでは名前、チケット名、乗客ID
train.drop(['Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)
test.drop(['Name', 'Ticket'], inplace=True, axis=1)


# In[ ]:


train.head()


# In[ ]:


# 獲特徴の相関関係を取得。多重共線性は発生していないと思われる。
k = 13
corrmat = train.corr()
cols = corrmat.nlargest(k, 'Survived').index
# df_train[cols].head()
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(16, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# ## Pclass

# In[ ]:


# 部屋のクラスによって生存率は異なるかを調査。Pclassが小さい方が生存率が高いことがわかる。
train.pivot_table(index='Pclass', columns='Survived', aggfunc='count').Age


# ## Sex

# In[ ]:


# 性別によって生存率は異なるかを調査。女性の方が生存率が高いことがわかる。
train.pivot_table(index='Sex', columns='Survived', aggfunc='count').Age


# ## Age

# In[ ]:


train.Age.unique()


# ### 年齢層がばらけすぎているため、年齢でグルーピングを行う。

# In[ ]:


train["Age_Group"] = train.Age.apply(lambda x : 0 if x >= 0  and x < 10 else (1 if x >= 10 and x < 20 else (2 if x >= 20 and x < 30 else (3 if x >= 30 and x < 40 else (4 if x >= 40 and x < 50 else (5 if x >= 50 and x < 60 else 6))))))
test["Age_Group"] = test.Age.apply(lambda x : 0 if x >= 0  and x < 10 else (1 if x >= 10 and x < 20 else (2 if x >= 20 and x < 30 else (3 if x >= 30 and x < 40 else (4 if x >= 40 and x < 50 else (5 if x >= 50 and x < 60 else 6))))))


# In[ ]:


train.Age_Group.unique()


# In[ ]:


# 年齢によって生存率は異なるかを調査。子供の方が生存率が高いことがわかる。
train.pivot_table(index='Age_Group', columns='Survived', aggfunc='count').Age


# ## Sibsp & Parch

# In[ ]:


# 配偶者を持つ人が優先的に助けられていることがわかる。
train.pivot_table(index='SibSp', columns='Survived', aggfunc='count').Age


# In[ ]:


# 子供を持つ人が優先的に助けられていることがわかる。
train.pivot_table(index='Parch', columns='Survived', aggfunc='count').Age


# ## Fare

# In[ ]:


train.describe().Fare


# In[ ]:


print('チケット価格が中央値より高い人たちの生存者数\n', train[train.Fare > 14].Survived.value_counts())
print('チケット価格が中央値より低い人たちの生存者数\n', train[train.Fare < 14].Survived.value_counts())


# In[ ]:


print('チケット価格が75%より高い人たちの生存者数\n', train[train.Fare > 31].Survived.value_counts())
print('チケット価格が75%より低い人たちの生存者数\n', train[train.Fare < 31].Survived.value_counts())


# ### チケットの価格が高い人たちの生存率が高いことがわかる。

# ## Embarked

# In[ ]:


# Cから乗船した人の方が生存率が高い
train.pivot_table(index='Embarked', columns='Survived', aggfunc='count').Age


# ## Sibsp & Parchの特徴追加

# In[ ]:


# 兄弟や配偶者、子供がいるかいないかの特徴を作成する
train["isSibsp"] = train.SibSp.apply(lambda x : 0 if x == 0 else 1)
train["isParch"] = train.Parch.apply(lambda x : 0 if x == 0 else 1)
test["isSibsp"] = test.SibSp.apply(lambda x : 0 if x == 0 else 1)
test["isParch"] = test.Parch.apply(lambda x : 0 if x == 0 else 1)


# ## 特徴データの整形

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# 不要になった特徴データを削除
drop_cols = ['Age', 'SibSp', 'Parch', 'Cabin']
train.drop(drop_cols, axis=1, inplace=True)
test.drop(drop_cols, axis=1, inplace=True)


# In[ ]:


# カテゴリデータを数値のラベルに変換
label = LabelEncoder()
cat_cols = ['Pclass', 'Sex', 'Embarked', 'Age_Group', 'isSibsp', 'isParch']
for column in cat_cols:
    train[column] = label.fit_transform(train[column])
    test[column] = label.fit_transform(test[column])


# In[ ]:


# Sex, Embarkedが文字列→数値に変換されている
train.head()


# 

# In[ ]:


# Fareの正規化(値を0 ~ 1に変換する)
ms = MinMaxScaler()
train.Fare = ms.fit_transform(train.Fare.values.reshape(-1, 1))
test.Fare = ms.fit_transform(test.Fare.values.reshape(-1, 1))


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## モデルに投入するデータを作成

# In[ ]:


# Xには特徴データのみを投入し、yには目的変数のみを投入
X = train.drop('Survived', axis=1)
y = train.Survived


# In[ ]:


# トレーニングデータと評価データを分割する
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.3, random_state=0,
)


# ## lightGBMでモデルを作成

# In[ ]:


# lightGBMが読み込める形にデータを整形
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


# In[ ]:


# lightGBMのパラメータを設定
params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.15,
        'num_leaves': 32,
        'min_data_in_leaf': 1,
        'num_iteration': 100,
        'verbose': 1
}


# In[ ]:


# 誤差関数の計算開始
gbm = lgb.train(params,
            lgb_train,
            num_boost_round=50,
            valid_sets=lgb_eval,
            early_stopping_rounds=10)


# In[ ]:


PassengerId = test.PassengerId.values
TEST = test.drop('PassengerId', axis=1).values
prediction = gbm.predict(TEST)


# ## XGBoostでモデルを作成

# In[ ]:


# xgboostが読み込める形にデータを整形
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_test, label=y_test)


# In[ ]:


# xgboostのパラメータを設定
param = {'max_depth': 2, 
         'eta': 0.15,
         'gamma' : 0,
         'min_child_weight' : 1,
         'silent': 1, 
         'objective': 'reg:linear',
         'nthread' : 6,
         'eval_metric' : 'auc'
        }


# In[ ]:


# トレーニング用と評価用の誤差関数を定義
evallist = [(dval, 'eval'), (dtrain, 'train')]


# In[ ]:


# 誤差関数の計算開始
num_round = 500
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)


# In[ ]:


PassengerId = test.PassengerId.values
TEST = test.drop('PassengerId', axis=1).values
prediction = gbm.predict(TEST)


# In[ ]:


submissions = pd.DataFrame({'PassengerId' : PassengerId,
                            'Survived' : prediction})
submissions.Survived = submissions.Survived.apply(lambda x : 1 if x > 0.49 else 0)
submissions.to_csv('./submissions.csv', index=False)


# In[ ]:




