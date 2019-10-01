#!/usr/bin/env python
# coding: utf-8

# # kaggle - Titanic: Machine Learning from Disaster

# In[ ]:


import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


warnings.simplefilter("ignore")


# # 変数
# 
# ## 独立変数（説明変数）
# 
# - PassengerId: 乗客ID
# - Pclass: 客室の等級（1st, 2nd , 3rd）
# - Name: 名前
# - Sex: 性別
# - Age: 年齢
# - SibSp: 共に乗船していた兄弟（siblings）や 配偶者（spouses）の数
# - Parch: 共に乗船していた親（parents）や子供（children）の数
# - Ticket: チケットのタイプ
# - Fare: チケットの料金
# - Cabin: 客室番号
# - Embarked: 乗船港（**Q**ueenstown, **C**herbourg, **S**outhampton）
# 
# ## 従属変数（目的変数）
# - Survived：生存者かどうか（1: 助かった、0：助からなかった）

# ## Pandasで下ごしらえ

# In[ ]:


# データの読み込み（トレーニングデータとテストデータにすでに分かれていることに注目）
df_train = pd.read_csv('../input/train.csv') # トレーニングデータ
df_test = pd.read_csv('../input/test.csv') # テストデータ


# In[ ]:


# SexId を追加
df_train['SexId'] = df_train['Sex'].map({'male': 1, 'female': 0})
df_test['SexId'] = df_test['Sex'].map({'male': 1, 'female': 0})


# In[ ]:


# FamilySize = SibSp + Parch
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']


# In[ ]:


# Ageの欠損値保管
df_train['AgeNull'] = df_train['Age'].isnull()
age_median = df_train['Age'].median()
df_train['Age'].fillna(age_median, inplace=True)
df_test['Age'].fillna(age_median, inplace=True)


# In[ ]:


# Embarked
common_embarked = df_train['Embarked'].value_counts().index[0]
df_train['Embarked'].fillna(common_embarked, inplace=True)
df_test['Embarked'].fillna(common_embarked, inplace=True)
df_train['EmbarkedNum'] = df_train.Embarked.map({'S': 0, 'C': 1, 'Q': 2})
df_test['EmbarkedNum'] = df_test.Embarked.map({'S': 0, 'C': 1, 'Q': 2})


# In[ ]:


# inputs = ['FamilySize', 'SexId', 'Age', 'EmbarkedNum']
inputs = ['FamilySize', 'SexId', 'Age']


# In[ ]:


X_train = df_train[inputs].values.astype('float32')
X_test = df_test[inputs].values.astype('float32')


# In[ ]:


y_train = df_train['Survived'].values


# In[ ]:


if df_train.columns.values.__contains__('PassengerId'):
    df_train.index = df_train.pop('PassengerId') 


# In[ ]:


if df_test.columns.values.__contains__('PassengerId'):
    df_test.index = df_test.pop('PassengerId')


# # 機械学習

# ### モデル選択

# In[ ]:


# 分類モデルの読み込み

# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
# K最近傍法
from sklearn.neighbors import KNeighborsClassifier
# サーポートベクターマシン
from sklearn.svm import SVC
# 決定木
from sklearn.tree import DecisionTreeClassifier
# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
# 勾配ブースティング
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


# 交差検証用モジュールの読み込み

from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score


# In[ ]:


#　複数の分類器を用意
classifiers = [
    ('lr', LogisticRegression()), 
    ('knn', KNeighborsClassifier()),
    ('linear svc', SVC(kernel="linear")),
    ('rbf svc', SVC(gamma=2)),
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier(random_state=42)),
    ('gbc', GradientBoostingClassifier())
]


# In[ ]:


# それぞれのモデルに対して、交差検証（CV）をかける
import time
results = {}
exec_times = {}

for name, model in classifiers:
    tic = time.time()
    result = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    exec_time = time.time() - tic
    exec_times[name] = exec_time
    results[name] = result
    
    print("{0:.3f} ({1:.3f}): time {2:.2f}s, {3}".format(result.mean(), result.std(), exec_time, name))


# In[ ]:


# 結果をデータフレームに入れる
df_results = pd.DataFrame(results)


# In[ ]:


# ボックスプロットによる結果の描画
df_results[df_results.median().sort_values(ascending=True).index].boxplot(vert=False);


# ## GradientBoostingClassifierで学習

# In[ ]:


# === 線形モデル ===
# モジュールの読み込み
# from sklearn import linear_model
#  モデル構築
# model = linear_model.LogisticRegression()

# === サポートベクターマシン ===
# モジュールの読み込み
#from sklearn import svm
#  モデル構築
#model = svm.SVC()

# === K最近傍法 ===
# モジュールの読み込み
#from sklearn.neighbors import KNeighborsClassifier
#  モデル構築
#model = KNeighborsClassifier()

# === ランダムフォレスト ===
# モジュールの読み込み
#from sklearn import ensemble
#  モデル構築
#model = ensemble.RandomForestClassifier(n_estimators=5, max_depth=10)

# === 勾配ブースティング ===
# モジュールの読み込み
from sklearn import ensemble
#  モデル構築
model = ensemble.GradientBoostingClassifier()


# In[ ]:


# 学習
model.fit(X_train, y_train)


# In[ ]:


# トレーニングセットに対する予測
y_train_pred = model.predict(X_train)


# In[ ]:


# テストセットに対する予測
y_test_pred = model.predict(X_test)


# In[ ]:


# 評価基準モジュール（metrics）の読み込み
from sklearn import metrics


# In[ ]:


# トレーニングデータに対する予測精度を計算
print(metrics.accuracy_score(y_train, y_train_pred))


# In[ ]:


df_test['Survived'] = y_test_pred


# In[ ]:


warnings.simplefilter("ignore")

df_fi = pd.DataFrame(model.feature_importances_, index=df_train[inputs].columns)
df_fi.sort(columns=0, inplace=True)
df_fi.plot(kind='barh', legend=False)


# ## パラメーターチューニング

# In[ ]:


# ランダムサーチ用にRandomizedSearchCVモジュールを読み込む
from sklearn.grid_search import RandomizedSearchCV
# 分布を指定するためにscipy.statsを読み込む
import scipy.stats as stats


# ### GradientBoostingClassifier Ver

# In[ ]:


# "loss": 'deviance', 
# "learning_rate": 0.1, 
# "n_estimators": 100, 
# "subsample": 1.0, 
# "min_samples_split": 2, 
# "min_samples_leaf": 1, 
# "min_weight_fraction_leaf": 0.0, 
# "max_depth": 3, 
# "init": None, 
# "random_state": None, 
# "max_features": None, 
# "verbose": 0, 
# "max_leaf_nodes": None, 
# "warm_start": False, 
# "presort": 'auto'

# パラメータ空間上に分布を指定する（今回はランダムフォレストを仮定）
param_dist = {
            "n_estimators": np.arange(75, 125),
            "min_samples_split": stats.randint(2, 11), 
            "min_samples_leaf": stats.randint(1, 5), 
            "max_features": stats.randint(1, 3)
}


# In[ ]:


# ランダムサーチCVオブジェクトを作る
random_search_gbc = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), 
                                   param_distributions=param_dist, cv=10, 
                                   n_iter=10, random_state=42)


# In[ ]:


# ランダムサーチを実行
tic = time.time() # 時間計測開始
random_search_gbc.fit(X_train, y_train)
toc = time.time() # 時間計測終了


# In[ ]:


# 結果を表示
from pprint import pprint
print("Best score: {0}\nExecution time: {1:.2f} sec".format(random_search_gbc.best_score_, toc - tic))
print("Best param:")
pprint(random_search_gbc.best_params_)


# In[ ]:


# ベストなパラメータを別名で保存
gbc_best_params = random_search_gbc.best_params_
print(gbc_best_params)


# In[ ]:


# モデルの構築（ランダムサーチで見つけたベストなパラメータを使用）
best_gbc_model = GradientBoostingClassifier(random_state=42, **gbc_best_params)


# In[ ]:


# モデルの学習
best_gbc_model.fit(X_train, y_train)


# In[ ]:


# トレーニングデータに対する予測精度
print("mean accuracy (train): {0:.4f}".format(best_gbc_model.score(X_train, y_train)))


# ### LogisticRegressionVer

# In[ ]:


# "penalty": 'l2', 
# "dual": False, 
# "tol": 0.0001, 
# "C": 1.0, 
# "fit_intercept": True, 
# "intercept_scaling": 1, 
# "class_weight": None, 
# "random_state": None, 
# "solver": 'liblinear', 
# "max_iter": 100, 
# "multi_class": 'ovr', 
# "verbose": 0, 
# "warm_start": False, 
# "n_jobs": 1

# パラメータ空間上に分布を指定する（今回はランダムフォレストを仮定）
param_dist = {
            "class_weight": ['balanced', None], 
            "max_iter": np.arange(75, 125)
}


# In[ ]:


# ランダムサーチCVオブジェクトを作る
random_search_lr = RandomizedSearchCV(LogisticRegression(random_state=42), 
                                   param_distributions=param_dist, cv=10, 
                                   n_iter=10, random_state=42, n_jobs=-1)


# In[ ]:


# ランダムサーチを実行
tic = time.time() # 時間計測開始
random_search_lr.fit(X_train, y_train)
toc = time.time() # 時間計測終了


# In[ ]:


# 結果を表示
from pprint import pprint
print("Best score: {0}\nExecution time: {1:.2f} sec".format(random_search_lr.best_score_, toc - tic))
print("Best param:")
pprint(random_search_lr.best_params_)


# In[ ]:


# ベストなパラメータを別名で保存
lr_best_params = random_search_lr.best_params_
print(lr_best_params)


# In[ ]:


# モデルの構築（ランダムサーチで見つけたベストなパラメータを使用）
best_lr_model = LogisticRegression(random_state=42, **lr_best_params)


# In[ ]:


# モデルの学習
best_lr_model.fit(X_train, y_train)


# In[ ]:


# トレーニングデータに対する予測精度
print("mean accuracy (train): {0:.4f}".format(best_lr_model.score(X_train, y_train)))


# ### RandomForestClassifier Ver

# In[ ]:


# "n_estimators": 10, 
# "criterion": 'gini', 
# "max_depth": None, 
# "min_samples_split": 2, 
# "min_samples_leaf": 1, 
# "min_weight_fraction_leaf": 0.0, 
# "max_features": 'auto', 
# "max_leaf_nodes": None, 
# "bootstrap": True, 
# "oob_score": False, 
# "n_jobs": 1, 
# "random_state": None, 
# "verbose": 0, 
# "warm_start": False, 
# "class_weight": None

# パラメータ空間上に分布を指定する（今回はランダムフォレストを仮定）
param_dist = {
            "n_estimators": np.arange(75, 125),
            "min_samples_split": stats.randint(2, 11), 
            "min_samples_leaf": stats.randint(1, 5), 
            "max_features": stats.randint(1, 3)
}


# In[ ]:


# ランダムサーチCVオブジェクトを作る
random_search_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), 
                                   param_distributions=param_dist, cv=10, 
                                   n_iter=10, random_state=42)


# In[ ]:


# ランダムサーチを実行
tic = time.time() # 時間計測開始
random_search_rf.fit(X_train, y_train)
toc = time.time() # 時間計測終了


# In[ ]:


# 結果を表示
from pprint import pprint
print("Best score: {0}\nExecution time: {1:.2f} sec".format(random_search_rf.best_score_, toc - tic))
print("Best param:")
pprint(random_search_rf.best_params_)


# In[ ]:


# ベストなパラメータを別名で保存
rf_best_params = random_search_rf.best_params_
print(rf_best_params)


# In[ ]:


# モデルの構築（ランダムサーチで見つけたベストなパラメータを使用）
best_rf_model = RandomForestClassifier(random_state=42, **rf_best_params)


# In[ ]:


# モデルの学習
best_rf_model.fit(X_train, y_train)


# In[ ]:


# トレーニングデータに対する予測精度
print("mean accuracy (train): {0:.4f}".format(best_rf_model.score(X_train, y_train)))


# ## モデルアンサンブルによる予測

# In[ ]:


# VotingClassifierの読み込み
from sklearn.ensemble import VotingClassifier


# In[ ]:


# 複数のモデルを用意。各モデルのハイパーパラメータはチューニング済みと仮定
classifiers = [
    ('gbc', GradientBoostingClassifier(random_state=42, **gbc_best_params)),
    ('lr', LogisticRegression(random_state=42, **lr_best_params)),
    ('rf', RandomForestClassifier(random_state=42, **rf_best_params))
]


# In[ ]:


# VotingClassifierの作成
models = VotingClassifier(classifiers, weights=[1, 1, 1])


# In[ ]:


# トレーニング
models.fit(X_train, y_train)


# In[ ]:


# トレーニングデータに対する予測精度
print("mean accuracy (train): {0:.4f}".format(models.score(X_train, y_train)))

