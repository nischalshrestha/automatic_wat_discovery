#!/usr/bin/env python
# coding: utf-8

# **Reference Kernel**
# [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
# 
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import sklearn
import re
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

#Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier,
                             AdaBoostClassifier,
                             GradientBoostingClassifier,
                             ExtraTreesClassifier)

from sklearn.svm import SVC
from sklearn.cross_validation import KFold


# # Feature Exploration, Engineering and Cleaning
# 
# 分析を行うためにデータを整理し、使用できる状態にしていく。
# 特徴エンジニアリングを実行するので、既存の特徴から関連する特徴を作成する。

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

PassengerId = test['PassengerId']
train.head(3)


# In[ ]:


full_data = [train, test]
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

#Create a new feature reserve cabin
train['Has_Cabin'] = train['Cabin'].apply(lambda x:0 if type(x) == float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x:0 if type(x) == float else 1)

#Create a new feature FamilySize
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#Create a new feature IsAlone y=1/n=0
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#Remove all NULLs in Embarked columns
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

#Remove all NULLs in Fare columns
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

#Create a new feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    #無効の値に対して、平均と標準偏差を使用したランダム値を割り振る
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

#乗客の名前から敬称(Mr.Msなど)を取得する関数
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return

#Create a new feature Title
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

#変わった敬称を管理しやすくするためにカテゴライズ、一般的なものに変更
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    #Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    #Mapping titles
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    #Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
    #Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91 ) &
                (dataset['Fare'] <= 14.454 ) , 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454 ) &
                (dataset['Fare'] <= 31 ) , 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    #Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4;


# In[ ]:


#Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test = test.drop(drop_elements, axis = 1)


# # Visualisations
# 
# ここから視覚的にわかりやすいように表やグラフを表示する

# In[ ]:


train.head(3)


# ## Pearson Correlation Heatmap ( ピアソン相関ヒートマップ )
# 
# ### 相関関係について
# 相関関係( correlation coefficient )とは、二つの変数が連動して変化する度合いを示す統計学的指標のこと。
# ここでいう相関とは、ピアソンの積率相関係数を指している。
# 
# ### ヒートマップ
# 個々の値のデータ行列を色として表現した可視化グラフの一種。
# 色によってデータ量を視覚化することで直感的に把握することができるようになる。
# 
# 
# 
# 関連する一つの特徴が次の特徴とどのように関連しているか確認するために相関プロットを作成する。
# そのために、Seabornというグラフ描画ができるパッケージを使用する。
# 
# 
# 
# 参考になりそうなリンク
# 
# https://seaborn.pydata.org/
# 
# https://myenigma.hatenablog.com/entry/2015/10/09/223629
# 
# 
# 

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Feature', y=1.05, size=15)

#各データの相関指数を出す関数　　train.corr()

sns.heatmap(train.astype(float).corr(), linewidth=0.1, vmax=1.0,
           square=True, cmap=colormap, linecolor='white',annot=True)


# 各データ同士の相関はそこまで強くないことがヒートマップからわかる。
# 
# これは学習モデルにこれらの特徴を追加するという観点からは優れているらしい。
# 拡張には余分な情報はなく、かつ固有の情報を持っていることを意味しているらしい。

# ## Pairplots( ペアプロット図（散布図行列）)
# 
# ペアプロットと検索しても、どのような図を示すのかは何もでてこなかった。
# 
# プロット図と散布図についてはでてきたので、メモ。
# 
# プロット図・・・データ集合の描画手法の一種で、二種類以上の変数の関係をグラフで表す目的で使われる。
# 方法のことで図形自体はいろいろな種類があった。
# 箱ひげ図、等値線、散布図などをプロット図という。
# 
# 散布図・・・分布図ともいう。縦軸、横実に二項目の量や大きさを対応させ、データを点で描画したもの。相関関係を把握することができて、データ群が右上りなら正の相関、右下がりなら負の相関を表す。

# In[ ]:


g = sns.pairplot(train[ [u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked', u'FamilySize', u'Title'] ],
                hue='Survived', palette='seismic', size=1.2, diag_kind='kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))

g.set(xticklabels=[])


# # Ensembling & Stacking models
# 
# scikit-learn(Sklearn)という機械学習のライブラリを使用していくので、使いやすくするためにヘルパークラスを作成する。

# In[ ]:


# 使いやすいようにパラメーターの変数を用意する
ntrain = train.shape[0]
ntest = test.shape[0]
#乱数の固定値
SEED = 0
#交差検証の回数
NFOLDS = 5
#交差検証(クロスバリデーション)でパラメータを検証して、過学習が起こらないパラメータを決定する
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

# Sklearnのヘルパー
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
        return self.clf.fit(x,y).feature_importances_
    
    def afeature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)


# ## Out-of-Fold Predictinons
# 
# オーバーフィッティング（過学習）を防ぐため
# モデルとトレーニングセットを引数に、トレーニングセットでモデルをトレーニングし、追加されたモデルの予測で新しいトレーニングセットを返すヘルパ関数。

# In[ ]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# ## 第一レベルモデルの生成
# 1. Random Forest classifier
# 2.Extra Trees classifier
# 3.AdaBoost classifier
# 4.Gradinet Boosting classifier
# 5.Support Vector Machine
# 
# ### パラメータ
# n_jobs：トレーニグプロセスに使用されるコアの数。-1に設定すると、すべてのコアが使用される。
# 
# n_estimators：トレーニングプロセスで使用される分類木の数。10がデフォルト
# 
# max_depth：分類木のノード数、どのくらい拡張するかを決定する。数が大きすぎると、過学習になる危険がある。
# 
# verbose：学習プロセエス中にテキストを出力するかどうかを制御します。0の値はすべてのテキストを抑制し、3の値は各繰り返しでツリー学習プロセスを出力します。
# 
# 

# In[ ]:


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


# In[ ]:


#create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# In[ ]:


#生存しているかの行列を一次元配列にする
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values #trainデータの配列
x_test = test.values #testデータの配列


# ### 第一レベル予測
# 5つの分類器を元にOut-of-fold prediction関数を実行し、第一レベル予測を出す。

# In[ ]:


#Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  #Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test) #Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) #AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test) #Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test) #Support Vector Classifier

print('Training is complete')


# Sklearnの機能で特徴の重要度を算出することができる。

# In[ ]:


rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)


# In[ ]:


#ここでリストにして変数に代入しているが、値が変わっている理由がわからない
rf_features = [0.10474135,  0.21837029,  0.04432652,  0.02249159,  0.05432591,  0.02854371
  ,0.07570305,  0.01088129 , 0.24247496,  0.13685733 , 0.06128402]
et_features = [ 0.12165657,  0.37098307  ,0.03129623 , 0.01591611 , 0.05525811 , 0.028157
  ,0.04589793 , 0.02030357 , 0.17289562 , 0.04853517,  0.08910063]
ada_features = [0.028 ,   0.008  ,      0.012   ,     0.05866667,   0.032 ,       0.008
  ,0.04666667 ,  0.     ,      0.05733333,   0.73866667,   0.01066667]
gb_features = [ 0.06796144 , 0.03889349 , 0.07237845 , 0.02628645 , 0.11194395,  0.04778854
  ,0.05965792 , 0.02774745,  0.07462718,  0.4593142 ,  0.01340093]


# In[ ]:


cols = train.columns.values

feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
     'Extra Trees feature importances': et_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature
    })


# 特徴の重要度の関わりを散布図にして表示

# In[ ]:


#Scatter plot

#Random Forest
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
        colorscale = 'Portland',
        showscale = True
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
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig,filename = 'scatter2010')


#Extra Trees
trace = go.Scatter(
    y = feature_dataframe['Extra Trees feature importances'].values,
    x = feature_dataframe['features'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Extra Trees feature importances'].values,
        colorscale = 'Portland',
        showscale = True
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
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = 'scatter2010')


#AdaBoost
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale = 'Portland',
        showscale = True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize = True,
    title = 'AdaBoost Feature Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter2010')


#Gradient Boost
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale = 'Portland',
        showscale = True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize = True,
    title = 'Gradient Boosting Feature Importances',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='scatter2010')


# 次に　特徴の重要度を計算し、データフレームに保存します。

# In[ ]:


feature_dataframe['mean'] = feature_dataframe.mean(axis = 1)
feature_dataframe.head(11)


# 5つのモデルから出た特徴の重要度を平均し、棒グラフにして表示する

# In[ ]:


y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values

data = [go.Bar(
    x = x,
    y = y,
    width = 0.5,
    marker = dict(
        color = feature_dataframe['mean'].values,
        colorscale = 'Portland',
        showscale = True,
        reversescale = False
    ),
    opacity = 0.6
)]

layout = go.Layout(
    autosize = True,
    title = 'Barplots of Mean Fearure Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = 'bar-direct-labels')


# ### 第二レベル予測
# 第一レベルの予測結果を新しい特徴として使用する。  
# 各モデルの項目を作成し、結果を格納する

# In[ ]:


base_predictions_train = pd.DataFrame({
    'RandomForest': rf_oof_train.ravel(),
    'ExtraTrees': ada_oof_train.ravel(),
    'AdaBoost': ada_oof_train.ravel(),
    'GradientBoost': gb_oof_train.ravel()
})

base_predictions_train.head()


# ### 第二レベルのトレーニングセットの相関ヒートマップ

# In[ ]:


data = [
    go.Heatmap(
        z = base_predictions_train.astype(float).corr().values,
        x = base_predictions_train.columns.values,
        y = base_predictions_train.columns.values,
        colorscale = 'Viridis',
        showscale = True,
        reversescale = True
    )
]

py.iplot(data, filename = 'labelled-heatmap')


# お互いに無関係で、いいスコアを出すモデルを持つことにはメリットがあると、コンテストの勝者やかなりの数の記事で言われているらしい。

# In[ ]:


x_train = np.concatenate(
    ( 
        et_oof_train,
        rf_oof_train,
        ada_oof_train,
        gb_oof_train,
        svc_oof_train
    ),
    axis = 1
)

x_test = np.concatenate(
    (
        et_oof_test,
        rf_oof_test,
        ada_oof_test,
        gb_oof_test,
        svc_oof_test
    ),
    axis = 1
)


# 第一レベルの予測結果を訓練データとテストデータに結合して、第二レベル予測に使用する。

# ### 第二レベル予測にはXGBoostモデルを使用する
# XGBoostは大規模なブーストツリーアルゴリズムを最適かするために構築された。
#     
# https://xgboost.readthedocs.io/en/latest/
#   

# In[ ]:


gbm = xgb.XGBClassifier(
    n_estimators = 2000,
    max_depth = 4,
    min_child_weight = 2,
    gamma = 0.9,
    subsample = 0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight = 1
).fit(x_train, y_train)

predictions = gbm.predict(x_test)


# max_depth:
#   
# gamma：
#   
# eta：

#  提出するファイルを作成する
# 

# In[ ]:


StackingSubmission = pd.DataFrame({
    'PassengerID': PassengerId,
    'Survived': predictions
})

StackingSubmission.to_csv('StackingSubmission.csv', index = False)


# In[ ]:




