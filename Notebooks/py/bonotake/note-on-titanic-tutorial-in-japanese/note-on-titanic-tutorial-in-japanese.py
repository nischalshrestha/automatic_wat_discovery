#!/usr/bin/env python
# coding: utf-8

# # Note on Titanic tutorial: May 6, 2017 (in Japanese)
# このGW、[KaggleのTitanicチュートリアル](https://www.kaggle.com/c/titanic)に取り組んだので、メモ。ちなみに、kaggleはもちろん、機械学習もデータ分析もPythonも初心者です。
# 
# 英語で書こうかと思ったけどとりあえず日本語で。
# 
# とりあえず最高スコア（_0.79426_）を出したときの再現。ただし、今やってみても値が一致しない（5人分ほど違う）ので、たまたま運が良かったのかもしれない。（train_test_split で分割した後のサンプルでモデルを作ったからかもしれない）
# 
# 以下、さらっと書いてますが実際は試行錯誤の結果です。
# 
# ## 参考にした他の人の仕事
# - https://www.kaggle.com/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline
# - https://www.kaggle.com/startupsci/titanic-data-science-solutions
# - [Python機械学習プログラミング 達人データサイエンティストによる理論と実践](http://book.impress.co.jp/books/1115101122)
# 
# ## ライブラリ一式を読み込む
# とりあえず、必要なのをがさごそと。

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier


# ## ファイルを読み込む

# In[ ]:


df = pd.read_csv('../input/train.csv')
df_final = pd.read_csv('../input/test.csv')


# "final"なんて名前にしてるのは、train.csvから更にトレーニングデータとテストデータに分割していたので、名前が被らないように。深い意味はない。

# ## データを眺める
# とりあえず……

# In[ ]:


df.head()


# 欠測値がありそう。カウントしてみる。
# 
# 

# In[ ]:


df.count()


# In[ ]:


df_final.count()


# Age, Fare, Cabin, Embarked が欠測してるっぽい。（Embarked は df のみ、Fare は df_final のみ）
# 
# じっと眺めて、Age, Fare, Embarkedは補完する方針で。Cabinは欠測しすぎてて使えなさそうなので、捨てることを考える。

# ## 欠測値の補完
# ### Age の補完
# 参照サイトを参考に、以下のコードで。

# In[ ]:


def guess_ages(_df1):
    result = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = _df1[(_df1['Sex'] == ['male','female'][i]) &                                   (_df1['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.mean()
            result[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
    return result

def impute_ages(_df1, guess):
    for i in range(0, 2):
        for j in range(0, 3):
            _df1.loc[ (_df1.Age.isnull()) & (_df1.Sex == ['male','female'][i]) 
                     & (_df1.Pclass == j+1), 'Age'] = guess[i,j]

    _df1['Age'] = _df1['Age'].astype(int)

    return _df1


# In[ ]:


guess = guess_ages(df)
df = impute_ages(df, guess)
guess = guess_ages(df_final)
df_final = impute_ages(df_final, guess)


# ここではguessの値をdf、df_final 別々に計算しているが、統一したほうが良いのかもしれない。
# ただ、統一してもスコアの向上は見られなかった。

# ### Embarked の補完
# とりあえず、集計してみる。

# In[ ]:


df[['Embarked','PassengerId']].groupby('Embarked').count()


# 最頻値Sが圧倒的に多いので、Sで補完することにする。

# In[ ]:


df['Embarked'] = df['Embarked'].fillna('S')


# ### Fareの補完
# ひとまず、欠測してるサンプルを見てみる。

# In[ ]:


df_final.loc[df_final['Fare'].isnull()]


# これを見る限り、Pclassが3、EmbarkedがSで、要は安いチケットを買っている可能性。
# 
# ということで、そういうチケットを買ってそうな人達の運賃から推測する。

# In[ ]:


df_final['Fare'].fillna(df_final[(df_final['Pclass'] == 3) &                                  (df_final['Embarked'] == 'S')]['Fare'].dropna().median(), inplace=True)


# ## Feature engineering
# ###  'Title' の追加
# 'Name'がこのままだと使いものにならなさそうだけど、肩書は参考になるかも。
# 
# ということで、肩書を抽出、新たな特徴として追加する。

# In[ ]:


def get_title(_df1):
    names = _df1.Name.values
    title = []
    for name in names:
        parts = name.split()
        for p in parts:
            if '.' in p:
                title.append(p)
                break

    return title

def add_title(_df1):
    title = get_title(_df1)
    _df1['Title'] = title
    
    _df1['Title'] = _df1['Title'].replace(['Lady.', 'Countess.','Capt.', 'Col.',         'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')
    _df1['Title'] = _df1['Title'].replace('Mlle.', 'Miss.')
    _df1['Title'] = _df1['Title'].replace('Ms.', 'Miss.')
    _df1['Title'] = _df1['Title'].replace('Mme.', 'Mrs.')
    
    return _df1


# In[ ]:


df = add_title(df)
df.head()


# TitleとSurvivedの傾向を見てみる。

# In[ ]:


df[['Survived', 'Title']].groupby('Title', as_index=False)                             .mean().sort_values(by='Survived', ascending=False)


# 'Sex'との比較。

# In[ ]:


df[['Survived', 'Sex', 'Title']].corr()


# ↑なんでかcorr で Title が出てこない……拗ねる。
# まいいや。次行く。同じことをdf_final にもしておく。

# In[ ]:


df_final = add_title(df_final)


# ### 家族サイズ FamilySize 追加
# 

# In[ ]:


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df_final['FamilySize'] = df_final['SibSp'] + df_final['Parch'] + 1


# 正味、このFamilySizeの追加（によってSibSpとParchが不要になったこと）がスコア上昇にかなり効いた。次元減らすの大事。
# 
# ちなみに：

# In[ ]:


df[['Survived', 'FamilySize']].groupby('FamilySize', as_index=False).mean().sort_values(by='Survived', ascending=False)


# 2～4人だと生き残る確率が高くて、1人（独り身）や5人以上の大家族だと逆に死んでるっぽい。
# 
# そういう意味では、ここでバンド化した方がいいんだろうけど、今回はしてなかった。

# ### 特徴量の選択
# ここで、要らないものを捨てちゃう。

# In[ ]:


df2 = df.drop(['Ticket', 'Name', 'Cabin', 'SibSp', 'Parch', 'PassengerId', 'Survived'], axis=1)
df2_final = df_final.drop(['Ticket', 'Name', 'Cabin', 'SibSp', 'Parch', 'PassengerId'], axis=1)


# In[ ]:


df2.head()


# ## カテゴリ値の数値化
# 残る特徴量のうち、カテゴリ値は Sex、Embarked、Title。
# 
# このうち、Sex は2値で、そのまま数値化して順序付けしても問題なさそうなので、そうする。
# 
# 他の特徴量については、one-hotエンコーディングを使う。
# 
# ### Sexの数値化

# In[ ]:


df['Sex'] = df['Sex'].replace('male', 1)
df['Sex'] = df['Sex'].replace('female', 0)

df_final['Sex'] = df_final['Sex'].replace('male', 1)
df_final['Sex'] = df_final['Sex'].replace('female', 0)


# 同じことは sklearn.preprocessing.LabelEncoder を使えばもっとスマートにできるようだ。

# ### 残りのカテゴリ値のone-hotパラメータ化
# 次のコマンドで一気に。

# In[ ]:


df2 = pd.get_dummies(df2)
df2_final = pd.get_dummies(df2_final)

df2.head()


# In[ ]:


df2_final.head()


# 本当は更に、各値の正規化 or 標準化 をした方がいいはずなんだけど、実際やっても特に影響なかった（スコアが横ばい、もしくは低下した）。
# 
# 大体、以上が前処理。
# 
# ## モデルの選択
# 後はモデルの選択。
# 
# ### サンプルの分割
# train_test_split を使う。

# In[ ]:


X = df2.values
X_final = df2_final.values
y = df.Survived.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ### Logistic Regression

# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, y_train)

lr.score(X_train, y_train)


# In[ ]:


lr.score(X_test, y_test)


# トレーニングデータとテストデータの差がない（↑むしろ逆？）になってて悪くない感じ。
# 
# ### SVM

# In[ ]:


svm = SVC()
svm.fit(X_train, y_train)

svm.score(X_train, y_train)


# In[ ]:


svm.score(X_test, y_test)


# やや過学習気味。
# 
# ### Random Forest

# In[ ]:


forest = RandomForestClassifier()
forest.fit(X_train, y_train)

forest.score(X_train, y_train)


# In[ ]:


forest.score(X_test, y_test)


# めっちゃ過学習してる。以上から、ロジスティック回帰採用。
# 
# ちなみに、random forest の結果から特徴量の重要度を観てみると：

# In[ ]:


importances=forest.feature_importances_
std = np.std([forest.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
sorted_important_features=[]
predictors = df2.columns
for i in indices:
    sorted_important_features.append(predictors[i])

get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt

plt.figure()
plt.title("Feature Importances By Random Forest Model")
plt.bar(range(np.size(predictors)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')

plt.xlim([-1, np.size(predictors)])


# FareとAgeがめっちゃ効いている。
# 
# この辺りから、特徴量を選択する方針もアリだと思えて次元削減をいろいろ試してみたけど、どれも効かなかった。

# ## 実際に予測する

# In[ ]:


def to_csv(fname, result):
    #csvファイルの作成
    results = []
    index = df_final['PassengerId'].values
    for idx, r in zip(index, result):
        results.append([idx, r])
    df_result = pd.DataFrame(results)
    df_result.columns = ['PassengerId', 'Survived']
    df_result.to_csv(fname, index=False)


# In[ ]:


lr.fit(X, y)
result = lr.predict(X_final)

to_csv('result.csv', result)


# 以上でおしまい。
