#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/startupsci/titanic-data-science-solutions

# Variable	Definition	Key<br>
# survival 	Survival 	0 = No, 1 = Yes<br>
# pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd<br>
# sex 	Sex 	<br>
# Age 	Age in years<br> 	
# sibsp 	# of siblings / spouses aboard the Titanic 	<br>
# parch 	# of parents / children aboard the Titanic 	<br>
# ticket 	Ticket number 	<br>
# fare 	Passenger fare 	<br>
# cabin 	Cabin number 	<br>
# embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton<br>
# 
# Variable Notes<br>
# pclass: A proxy for socio-economic status (SES)<br>
# 1st = Upper<br>
# 2nd = Middle<br>
# 3rd = Lower<br>
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5<br>
# 
# sibsp: The dataset defines family relations in this way...<br>
# Sibling = brother, sister, stepbrother, stepsister<br>
# Spouse = husband, wife (mistresses and fiances were ignored)<br>
# 
# parch: The dataset defines family relations in this way...<br>
# Parent = mother, father<br>
# Child = daughter, son, stepdaughter, stepson<br>
# Some children travelled only with a nanny, therefore parch=0 for them.<br>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set()


# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from imblearn.over_sampling import SMOTE


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.columns = ['PassengerId', "Survi'ved", 'P class', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',  # わざと "'" と ' ' を付ける
                    'Ticket', 'Fare', 'Cabin', 'Embarked']
train_df.head()


# In[ ]:


train_df.columns = [c.lower().replace("'", "").replace(' ', '') for c in train_df.columns]
train_df.columns


# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()


# In[ ]:


test_df.columns = [c.lower().replace("'", "").replace(' ', '') for c in test_df.columns]
test_df.columns


# train_df, test_df 両方での操作のためにリストを作成する

# In[ ]:


combine = [train_df, test_df]


# In[ ]:


fig, axes = plt.subplots(figsize=(3,3))
print(f"{train_df.groupby(['survived']).size()}")
print(pd.DataFrame(train_df.groupby(['survived']).size()).apply(lambda x: x / sum(x))) # 
train_df.groupby(['survived']).size().plot.pie(autopct='%.2f%%')


# In[ ]:


train_df.describe()


# In[ ]:


num_cols = [col for col in train_df.columns if train_df[col].dtypes not in ['object']]
num_cols


# In[ ]:


train_df.describe(exclude='number')


# Age と Cabin に欠損値あり

# In[ ]:


null_cols = [col for col in train_df.columns if train_df[col].isnull().any()]
null_cols


# In[ ]:


ctg_cols = [c for c in train_df.columns if train_df[c].dtypes in ['object']]
ctg_cols


# 一致の確認

# In[ ]:


cols_combine = num_cols + ctg_cols
print([col for col in cols_combine if col not in train_df.columns])
print([col for col in train_df.columns if col not in cols_combine])
set(cols_combine) == set(train_df.columns)


# ### groupby([col]).size() でカテゴリ別の頻度、さらに可視化（pandas plot.pie() で円グラフ）

# In[ ]:


fig, axes = plt.subplots(1, 6, figsize=(20, 3))
for i, col in enumerate(['survived', 'pclass', 'sex', 'sibsp', 'parch', 'embarked']):
    ax = axes.ravel()[i]
    df_i = pd.DataFrame(train_df[col])
#     print(f"{i}, {df_i.groupby([col]).size()}")  # カテゴリ別の頻度と構成比
#     print(f"\n{pd.DataFrame(df_i.groupby([col]).size()).apply(lambda x: x / sum(x))}\n")
    df_i.groupby([col]).size().plot.pie(ax=ax, legend=False,
                                        colors=['lightpink', 'lightgrey', 'lavender'])
    ax.set_title(col)


# Analyze by pivoting features 特徴量の相関を解析する

# ['parch'] を例にとって .groupby(['']).mean() のおさらい。survived が 01 なのですべての要素の和を要素数で割った平均 mean() はそのカテゴリ内の1（この場合は生き残った人）の構成比になる

# In[ ]:


(0 * 445 + 1 * 233) / (445 + 233) # parch 0


# In[ ]:


print(f"{train_df[['parch', 'survived']].groupby(['parch', 'survived']).size()}\n")
print(f"上記の平均を mean() で取得\n{train_df[['parch', 'survived']].groupby(['parch']).mean()}")


# 上記を for で回して可視化

# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(14,3))
for i, col in enumerate(['pclass', 'sex', 'sibsp', 'parch', 'embarked']):
    ax = axes.ravel()[i]
    gby_mean = train_df[[col, 'survived']].groupby([col]).mean()
    print(f"{gby_mean}\n")
    gby_mean.plot.bar(ax=ax, legend=False)
    ax.set_xlabel(''); ax.set_title(col.upper())
    ax.set_xticklabels(gby_mean.index, rotation=0)

fig.tight_layout()


# 上記を pd.pivot_table で表現してみる

# まず parch でテスト

# In[ ]:


# pvt_parch = pd.pivot_table(train_df,
#                index=['parch'],
#                columns=['survived'],
#                values=['passengerid'],
#                aggfunc={'passengerid': [len]},)
# print(f"{pvt_parch}") # NaN があるので
# pvt_parch = pvt_parch.fillna(0) # .fillna(0) で置換
# print(f"{pvt_parch}")


# In[ ]:


# pvt_parch_2 = pd.pivot_table(train_df,
#                index=['parch'],
#                columns=['survived'],
#                values=['name'],                      # ここと
#                aggfunc={'name': [len]}).fillna(0)    # ここを
# print(f"{pvt_parch_2}")                              # 'name'に変えても同じ結果になる


# 列方向の比率に変換

# In[ ]:


# # def get_ratio(x):
# #     return x / sum(x)
# # pvt_ratio = pvt_parch.apply(get_ratio, axis=1)
# # print(f"{(pvt_ratio)}")
# pvt_ratio = pvt_parch.apply(lambda x: x / sum(x), axis=1) # 上記関数を lambda にした
# print(f"{pvt_ratio}")
# pvt_ratio.iloc[:, 1]


# pd.pivot_table を for で回して可視化（カラム内のカテゴリ別の suevived の比率）

# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(14,3))
for i, col in enumerate(['pclass', 'sex', 'sibsp', 'parch', 'embarked']):
    ax = axes.ravel()[i]
    pivot_col = pd.pivot_table(train_df,
                               index=[col],
                               columns=['survived'],
                               values=['name'],
                               aggfunc={'name': [len]}).fillna(0)
    pivot_ratio = pivot_col.apply(lambda x: x / sum(x), axis=1)
    survived_ratio = pivot_ratio.iloc[:, 1]
    survived_ratio.plot.bar(ax=ax, color='silver')
    ax.set_xlabel(''); ax.set_title(col.upper())
    ax.set_xticklabels(survived_ratio.index, rotation=0)
    print(f"{(survived_ratio)}")
fig.tight_layout()


# 量的特徴量ごとのヒストグラム。重ねて表示

# In[ ]:


print(f"{[c for c in num_cols if c != 'survived']}, {len([c for c in num_cols if c != 'survived'])}")
print(f"{train_df['survived'].unique()}")
fig, axes = plt.subplots(2, 3, figsize=(12,4))
for i, col in enumerate([c for c in num_cols if c != 'survived']):
    ax = axes.ravel()[i]
    df_i = train_df[[col, 'survived']]
    ax.set_ylabel('')
    ax.set_title(col.upper())
    colors = ['red', 'blue']
    for i_s in set(train_df['survived']):
        df_is = df_i[df_i['survived'] == i_s][col]
        df_is.plot.hist(ax=ax, alpha=.3, legend=False, color=colors[i_s], bins=20)
fig.tight_layout()


# 並べて表示

# In[ ]:


for i, col in enumerate([c for c in num_cols if c != 'survived']):  # これで for が 6回まわる
    df_i = train_df[[col, 'survived']]
#     print(df_i)
    fig, axes = plt.subplots(1, 2, figsize=(10,1))
#     hist, bins = np.histogram(df_i) # ax.hist は欠損値(age)でエラー
    colors = ['red', 'blue']
    for i_s in set(train_df['survived']):
        ax = axes.ravel()[i_s]
        df_is = df_i[df_i['survived'] == i_s][col]
        df_is.plot.hist(ax=ax, alpha=.3, legend=False, sharex=True, color=colors[i_s], bins=20)
#         print(f"{df_is.max()}")
        ax.set_ylabel('')
        ax.set_title("{} - survived:{}".format(col.upper(), i_s))
        ax.set_xlabel("max: {}".format(df_is.max()))
# fig.tight_layout()


# In[ ]:


pvt_age = pd.pivot_table(train_df
               , index=['age']
               , columns=['survived']
#                , values=['passengerid']
               , aggfunc={'passengerid': {len}}).fillna(0)


# In[ ]:


fig, axes = plt.subplots(figsize=(20,2))
pvt_age.plot.bar(ax=axes, legend=False, color=['pink', 'blue'])
# pvt_age.columns.shape


# 復習を兼ねて5歳刻みで pd.cut でビニング処理をする

# In[ ]:


[int(c) for c in np.linspace(0, 80, 17)]


# In[ ]:


bin_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
age_cut = pd.cut(train_df['age'], bin_list)


# In[ ]:


train_df['agebin'] = age_cut
train_df.head()


# In[ ]:


pvt_bin = pd.pivot_table(train_df
                        , index=['agebin']
                        , columns=['survived']
                        , values=['passengerid']
                        , aggfunc={'passengerid': [len]})
pvt_bin


# In[ ]:


fig, axes = plt.subplots(figsize=(12,2))
pvt_bin.fillna(0).plot.bar(ax=axes, width=.8)


# 上記ピボットテーブルよりも以下のヒストグラムの bins 調節のほうがきれい

# In[ ]:


fig, axes = plt.subplots(figsize=(12,2))
df_age_srv = train_df[['age', 'survived']]
xticks_range = np.linspace(0, 80, 17)
for i in set(df_age_srv['survived']): # {0, 1}
    df_i = df_age_srv[df_age_srv['survived'] == i]['age']
    df_i.plot.hist(ax=axes, alpha=.3, bins=40, label="survived: {}".format(i))
    axes.set_xticks(xticks_range); axes.set_ylabel('')
    axes.set_title("histogram of age in survived")
    axes.grid(); axes.legend()
# xticks_range


# pclass 別の age のヒストグラムを作ってみる

# In[ ]:


df_ts = train_df[['pclass', 'age', 'survived']] 
print(f"set(df_ts['pclass']): {set(df_ts['pclass'])}")
# print(f"{df_ts.head()}\n")
fig, axes = plt.subplots(1, 3, figsize=(14,2))
for i, cls in enumerate(set(df_ts['pclass'])):
    df_class = df_ts[df_ts['pclass'] == cls]
    colors = ['red', 'blue']
    for i_suv in set(df_class['survived']):
        ax = axes.ravel()[i]
        df_class_suv = df_class[df_class['survived'] == i_suv]
        df_class_suv['age'].plot.hist(ax=ax, bins=20, color=colors[i_suv], alpha=.3,
                                      label="survived: {}".format(i_suv), legend=True)
        ax.set_ylabel(''); ax.set_title("pclass: {}".format(cls))


# 横並びにしてy軸の一致がやっと出来たー ＼(^o^)／ 

# In[ ]:


df_i = train_df[['pclass', 'age', 'survived']]
print(f"{set(df_i['pclass'])}")
for i, cls in enumerate(set(df_i['pclass'])):  # {1, 2, 3} イテレート3回
    df_cls = df_i[df_i['pclass'] == cls]
#     hist, bins = np.histogram(df_cls['age'].dropna(), bins=20)
#     print(f"{i}: {hist}\n{bins}\n{hist.max()}")
#     print(f"{np.arange(max(hist)+1)}")
    fig, axes = plt.subplots(1, 2, figsize=(10,1.5))
    hist_max = 0
    bins = 20
    for isv in set(df_i['survived']): # y_ticks 取得のためのイテレート
#         ax = axes.ravel()[isv]
        df_sv = df_cls[df_cls['survived'] == isv]
        hist, bins = np.histogram(df_sv['age'].dropna(), bins=bins) # np.histogram で hist を取得
#         print(f"{i}:{isv}, {hist}, {max(hist)}")
        if max(hist) > hist_max:
            hist_max = max(hist)    # hist の max を取得して
        y_ticks = np.arange(0, hist_max+1+4, 5) # それをもとに y_ticks 設定。いったんループ終了
#         print(f"y_ticks: {y_ticks}")
    
    colors = ['pink', 'lightblue']
    for i_fin in set(df_i['survived']): # 上記で取得したy_ticksを使って再度イテレートして横一列にグラフ描画
#         print(f"i_fin: {i}, {i_fin}, {y_ticks}")
        ax = axes.ravel()[i_fin]
        df_sv = df_cls[df_cls['survived'] == i_fin]
        df_sv['age'].plot.hist(ax=ax, bins=bins, yticks=y_ticks, color=colors[i_fin]) # yticks=y_ticks で左右のy軸が一致
        ax.set_ylabel(''); ax.set_title(f"pclass: {cls} - survived={i_fin}")


# In[ ]:


grid = sns.FacetGrid(train_df, row='pclass', col='survived', size=2, aspect=1.6)
grid.map(plt.hist, 'age', bins=20)
grid.add_legend()


# In[ ]:


print(f"{set(train_df['embarked'])}")
# grid = sns.FacetGrid(train_df, row='embarked', height=2.2, aspect=1.6)
grid = sns.FacetGrid(train_df, row='embarked', aspect=1.6)
grid.map(sns.pointplot, 'pclass', 'survived', 'sex', palette='deep')
# grid.map(sns.pointplot(x='pclass', y='survived', data='sex', palette='deep') )


# Decisions.
# 
#     Add Sex feature to model training.
#     Complete and add Embarked feature to model training.
#     
# 結論
# 
#     モデルトレーニングにSex特徴量を追加する。
#     モデルトレーニングにEmbarked特徴量を補完して追加する。
# 

# Correlating categorical and numerical features
# 
# We may also want to correlate categorical features (with non-numeric values) and numeric features. We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).

# In[ ]:


# grid = sns.FacetGrid(train_df, row='embarked', col='survived', height=2, aspect=1.6)
grid = sns.FacetGrid(train_df, row='embarked', col='survived', aspect=1.6)
grid.map(sns.barplot, 'sex', 'fare', alpha=.5, ci=None)
# grid.map(sns.barplot, 'fare', 'sex', alpha=.5, ci=None)


# 観察
# 
#     より高い運賃を支払う乗客はより良い生存率を示した。 仮説【Fareの範囲を特徴量として作成】。
#     乗船港は生存率と相関する。
# 
# 結論
# 
#     Fare特徴量のバンディング(一定の区間で区切ってカテゴリカルにする)を検討する。

# 
# Wrangle data
# 
# We have collected several assumptions and decisions regarding our datasets and solution requirements. So far we did not have to change a single feature or value to arrive at these. Let us now execute our decisions and assumptions for correcting, creating, and completing goals.
# Correcting by dropping features
# 
# This is a good starting goal to execute. By dropping features we are dealing with fewer data points. Speeds up our notebook and eases the analysis.
# 
# Based on our assumptions and decisions we want to drop the Cabin (correcting #2) and Ticket (correcting #1) features.
# 
# Note that where applicable we perform operations on both training and testing datasets together to stay consistent.
# 

# In[ ]:


train_df.head()


# In[ ]:


# エラー防止のためここで改めて読み込み
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

for df in combine:
    df.columns = [c.replace('_', '').replace(' ', '').lower() for c in df.columns]
combine = [train_df, test_df]

print(f"Before: {train_df.shape}, {test_df.shape}, {combine[0].shape}, {combine[1].shape}")

train_df = train_df.drop(['ticket', 'cabin'], axis=1)
test_df = test_df.drop(['ticket', 'cabin'], axis=1)

combine = [train_df, test_df]

print(f"After: {train_df.shape}, {test_df.shape}, {combine[0].shape}, {combine[1].shape}")

combine[1].head()


# Creating new feature extracting from existing
# 
# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# In the following code we extract Title feature using regular expressions. The RegEx pattern (\w+\.) matches the first word which ends with a dot character within Name feature. The expand=False flag returns a DataFrame.

# 既存のものから新しい特徴量を作成する
# 
# NameとPassengerIdを削除する前に、NameからTitle(肩書)を抽出し、TitleとSurvivedの相関関係を調べたいと思います。
# 
# 次のコードでは、正規表現を使用してTitleを抽出します。 正規表現パターン (\w+\.)は、Name特徴量内のドット文字で終わる最初の単語と一致します。 expand = FalseフラグはDataFrameを返します。

# 正規表現の練習

# In[ ]:


print(f"{type(train_df[['name']])}\n{type(train_df['name'])}\n{type(train_df.name)}\n")
print(f"train_df['name'].head(): ### まずは名前を表示\n{train_df['name'].head()}\n")
print("これだと先頭の一文字のみを取得")
print("train_df['name'].str.extract('([A-Za-z])', expand=False).head()")
print(f"{train_df['name'].str.extract('([A-Za-z])', expand=False).head()}\n")
print("+（プラス）追加で英字の1回以上の繰り返し。すなわち , の直前まで")
print("train_df['name'].str.extract('([A-Za-z]+)', expand=False).head()")
print(f"{train_df['name'].str.extract('([A-Za-z]+)', expand=False).head()}\n")
print("メタ文字の.（ドット）は任意の一文字を表すので\でエスケープしてname内の.を表現。英字プラスドットを抽出")
print("train_df['name'].str.extract('([A-Za-z]+)\.', expand=False).head()")
print("{}".format(train_df['name'].str.extract('([A-Za-z]+)\.', expand=False).head()))


# In[ ]:


for dataset in combine:
    dataset['title'] = dataset['name'].str.extract('([A-Za-z]+)\.', expand=False)

train_df.head()


# In[ ]:


pd.concat([train_df['title'], test_df['title']], axis=0).value_counts()


# In[ ]:


pd.crosstab(train_df['sex'], train_df['title'])


# In[ ]:


pd.pivot_table(train_df
              , index=['sex']
              , columns=['title']
              , aggfunc={'name': [len]}
              , fill_value=0)


# In[ ]:


# train_df[['sex', 'title']].groupby(['sex', 'title']).size()


# In[ ]:


_, axes = plt.subplots(figsize=(8,2))
pd.crosstab(train_df['title'], train_df['sex']).plot.bar(ax=axes)


# 上記を pd.pivot_table で描画（練習 14-Aug-2018, Tue）

# In[ ]:


pv_t = pd.pivot_table(train_df
              , index=['sex']
              , columns=['title']
              , values=['age']
              , aggfunc={'age': [len]}
              , fill_value=0).T
fig, axes = plt.subplots(1, 2, figsize=(12,2))
ex_col = [('age', 'len', 'Mr'), ('age', 'len', 'Miss'), ('age', 'len', 'Master'), ('age', 'len', 'Mrs')] # 右のグラフで除くカラム
new_col = [indx for indx in pv_t.index if indx not in ex_col]
cols, titles = [pv_t.index, new_col], ['all titles', 'titles with a few elements']
for i, ax in enumerate(axes.ravel()): # カラムごとの件数にバラツキがあるので右側は上位を除いたグラフを表示
    pv_t.loc[cols[i], :].plot.bar(ax=ax)
    ax.set_xticklabels(pv_t.loc[cols[i], :].index.levels[2])
    ax.set_xlabel('')
    ax.set_title(titles[i])


# In[ ]:


grid = sns.FacetGrid(train_df, size=1.5, aspect=1.5, row='title', col='survived')
grid.map(plt.hist, 'age')


# seaborn だと yticks が変更できないので再描画

# In[ ]:


print(f"{train_df.groupby(['title']).size().shape}")
fig, axes = plt.subplots(3, 6, figsize=(14,5))
colors = ['red', 'blue']
for i, idx in enumerate(train_df.groupby(['title']).size().index):
    ax = axes.ravel()[i]
    df_t = train_df[train_df['title'] == idx][['age', 'survived']].dropna()
#     print(f"{idx}:\n{df_t}")
#     print(f"\n{idx}:")
    for i_s in set(train_df['survived']):
        df_hist = df_t[df_t['survived'] == i_s]
#         print(f"{i_s}:\n{df_hist}, {df_hist.shape}")
        if df_hist.shape[0] == 0:  # Empty DataFrame の時は
            continue              # continue文でループ内の処理をスキップする
        df_hist['age'].plot.hist(ax=ax, bins=20, label="{}".format(i_s), color=colors[i_s], alpha=.3)
        ax.set_ylabel(''); ax.set_title(idx); ax.legend()
fig.tight_layout()
# print(f"{train_df[train_df['title'] == 'Master']}")


# 年齢が低くて survived が拮抗している master を見てみる

# In[ ]:


train_df[train_df['title'] == 'Master']


# えー何これどういうこと？兄弟が少ない子を優先的に助けたってこと？

# In[ ]:


pd.pivot_table(train_df[train_df['title'] == 'Master']
              , index=['survived']
              , columns=['sibsp']
              , values=['passengerid']
              , aggfunc={'passengerid': [len]}
              , fill_value=0
              , )


# We can replace many titles with a more common name or classify them as Rare.

# In[ ]:


print(f"{set(train_df['title']) - set(test_df['title'])}")
print(f"{set(test_df['title']) - set(train_df['title'])}")
n_train = [n for n in set(train_df['title'])]
n_test = [n for n in set(test_df['title'])]
# print(f"{n_train}\n{n_test}")
# print(f"{n_train in n_test}\n{n_test in n_train}")
print(set(n_train + n_test))


# In[ ]:


print(f"{train_df.shape}, {test_df.shape}")
title_all = pd.concat([train_df['title'], test_df['title']])
print(f"{title_all.shape}")
print(f"{pd.DataFrame(title_all).groupby(['title']).size()}\n")
print(pd.DataFrame(title_all).groupby(['title']).size().index)


# In[ ]:


for dataset in combine:
    dataset['title'] = dataset['title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer'                                                  , 'Lady', 'Major', 'Rev', 'Sir'], 'Rare')
    dataset['title'] = dataset['title'].replace('Mlle', 'Miss')
    dataset['title'] = dataset['title'].replace('Mme', 'Mrs')
    dataset['title'] = dataset['title'].replace('Ms', 'Miss')
    
train_df[['title', 'survived']].groupby(['title']).mean()


# In[ ]:


print(f"{train_df.groupby(['title']).size().shape}")
fig, axes = plt.subplots(figsize=(6,1.5))
train_df.groupby(['title']).size().plot.bar(color='silver', rot=False); axes.set_xlabel('')
fig, axes = plt.subplots(2, 3, figsize=(10,4))
colors = ['red', 'blue']
for i, id in enumerate(train_df.groupby(['title']).size().index):
    ax = axes.ravel()[i]
    df_i = train_df[train_df['title'] == id][['age', 'survived']].dropna()
#     print(f"{i}: {id}")
    for isv in set(train_df['survived']):
#         print(f"{isv}::")
        df_hist = df_i[df_i['survived'] == isv]
#         print(f"{df_hist}")
        df_hist['age'].plot.hist(ax=ax, bins=20, label="{}".format(isv), color=colors[isv], alpha=.3)
        ax.set_ylabel(''); ax.set_title("{}".format(id))
        ax.legend()
fig.tight_layout()
{k: v for k, v in zip(train_df.groupby(['title']).size().index, train_df.groupby(['title']).size().values)}


# We can convert the categorical titles to ordinal.

# カテゴリカルなタイトルを序数に変換する

# In[ ]:


train_df.groupby(['title']).size().sort_values(ascending=False).index


# In[ ]:


train_df.groupby(['title']).size().index.isin(['Master', 'Miss', 'Mr', 'Mrs', 'Rare']).any()


# In[ ]:


title_mapping = {c: i + 1 for i, c in enumerate(train_df.groupby(['title'])                                                .size().sort_values(ascending=False).index)}
print(f"{title_mapping}")

if train_df.groupby(['title']).size().index.isin(['Master', 'Miss', 'Mr', 'Mrs', 'Rare']).any():   # エラー防止のため
    for dataset in combine:
        dataset['title'] = dataset['title'].map(title_mapping)
        dataset['title'] = dataset['title'].fillna(0)
    
train_df.head()


# In[ ]:


train_df['title'].value_counts()


# Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.

# In[ ]:


for i, dataset in enumerate(combine):
    print(f"{[col for col in dataset.columns.values]}, {dataset.shape}")


# In[ ]:


train_df.columns.isin(['passengerid', 'name']).any()


# In[ ]:


if train_df.columns.isin(['passengerid', 'name']).any():
    train_df = train_df.drop(['passengerid', 'name'], axis=1)
    test_df = test_df.drop(['name'], axis=1)
combine = [train_df, test_df]
for dataset in combine:
    print(f"{[n for n in dataset.columns]}, {dataset.shape}")


# Converting a categorical feature
# 
# Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
# 
# Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

# In[ ]:


set(train_df['sex'])


# In[ ]:


if set(train_df['sex']) == {'female', 'male'}:
    for dataset in combine:
        dataset['sex'] = dataset['sex'].map({'male': 0, 'female': 1})
train_df.head()


# Completing a numerical continuous feature
# 
# Now we should start estimating and completing features with missing or null values. We will first do this for the Age feature.
# 
# We can consider three methods to complete a numerical continuous feature.
# 
#     A simple way is to generate random numbers between mean and standard deviation.
# 
#     More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using median values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
# 
#     Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.
# 
# Method 1 and 3 will introduce random noise into our models. The results from multiple executions might vary. We will prefer method 2.
# 

# 連続的数値の特徴量を補完する
# 
# 今度は、欠損値またはnull値を持つ特徴量を、その欠損値を推定して補完する必要があります。まず、age特徴量でこれを行います。
# 
# 数値連続的特徴量を補完するのに、ここでは3つの方法が考えられます。
# 
#     簡単な方法は、平均と標準偏差の間の乱数を生成することです。
# 
#     欠損値を推測するより正確な方法は、他の相関する特徴量を使用することです。今回のケースでは、年齢、性別、およびPclassの間の相関を記録する。 PclassとGenderの特徴量の組み合わせのセット全体でageの中央値を使用してAgeの値を推測します。Pclass = 1、Gender = 0、Pclass = 1、Gender = 1などの中間の年齢などなど。
# 
#     方法1と2を組み合わせる。中央値に基づいて年齢値を推測する代わりに、PclassとGenderの組み合わせのセットに基づいて、平均と標準偏差の間になる乱数を使用する。
# 
# 方法1と3はランダムノイズをモデルに導入することになり、複数回の実行結果が異なる可能性があります。よって方法2を優先します。

# In[ ]:


np.corrcoef(train_df['pclass'], train_df['age'].fillna(0))


# 相関する特徴量として pclass と sex を選んだのはそれが age と強く相関しているというよりも、特徴量内のカテゴリ数が少ない（計算が容易）からという理由の方が大きいのかもしれない（以下を参照）。ただカテゴリごとのばらつきが少ないというのはある。

# In[ ]:


col_short = []
fig, axes = plt.subplots(2, 4, figsize=(12, 3))
i_hist = 0
for i, col in enumerate(train_df.columns):
    if train_df[col].nunique() < 10:
        col_short.append(col)
        print(f"{i}:{col} \n{train_df[col].nunique()}: {train_df[col].unique()}\ni_hist: {i_hist}\n")
        ax = axes.ravel()[i_hist]
        train_df.groupby(col).size().plot.bar(ax=ax, color='silver')
        ax.set_xlabel(''); ax.set_title(col)
        i_hist += 1
fig.tight_layout()
print(f"col_short: {col_short}, {len(col_short)}")
print(f"上記以外のカラム: {train_df.columns[~train_df.columns.isin(col_short)]}\n")
print(f"train_df.head(): \n{train_df.head()}")


# 別に相関が強いってわけではない。pclass には負の相関があるが。

# In[ ]:


list_corr_coef = []; name_corr_coef = [c for c in col_short if c != 'embarked']
for col in name_corr_coef:
    corr_coef = np.corrcoef(train_df[col], train_df['age'].fillna(0))
    print(f"{corr_coef[0, 1]}")
    list_corr_coef.append(corr_coef[0, 1])
fig, axes = plt.subplots(figsize=(10,2))
xticks = np.arange(len(name_corr_coef))
axes.bar(xticks, list_corr_coef)
axes.set_xticks(xticks)
axes.set_xticklabels(name_corr_coef); axes.grid()


# In[ ]:


# grid = sns.FacetGrid(train_df, row='pclass', col='sex', height=2, aspect=1.6)
grid = sns.FacetGrid(train_df, row='pclass', col='sex', aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)


# 自力で age の欠損値処理をする。テキストと同様に 'pclass' と 'sex' の組み合わせから 'age' を算出して補完する。

# まずはそのためにデータフレームのコピーを作成

# In[ ]:


train_df_copy = train_df.copy()
test_df_copy = test_df.copy()
print(f"{[c for c in train_df_copy.columns]}, {train_df_copy.columns.shape}")
print(f"{[c for c in test_df_copy.columns]}, {test_df_copy.columns.shape}")
print(f"{[c for c in train_df_copy.columns if c not in [c for c in test_df_copy.columns]]}: column(s) only in train_df_copy")


# In[ ]:


train_df_copy.head(3)


# まずは train_df_copy でプリントしてみる

# In[ ]:


print(f"{set(train_df_copy['pclass'])}\n{set(train_df_copy['sex'])}\n")
for i in set(train_df_copy['pclass']):
    df_i = train_df_copy[train_df_copy['pclass'] == i]
    for j in set(train_df_copy['sex']):
        df_j = df_i[df_i['sex'] == j]
#         print(f"{i}, {j}\n{df_j.head(3)}")
        age_target = df_j['age'].median()
        print(f"{i}, {j}: {age_target}")  # i, j それぞれで df を作って抽出及び age_target の算出はできたけど、
        print(f"{df_j[df_j['age'].isnull()].head()}\n")
#         print(f"{train_df_copy.loc[(train_df_copy['pclass'] == i) & (train_df_copy['sex'] == j) & \
#               (train_df_copy['age'].isnull()), 'age']}")  # 欠損値補完は train_df_copy の操作なので loc 行設定で3種の bool 設定
#         print((train_df_copy['pclass'] == i) & (train_df_copy['sex'] == j) & (train_df_copy['age'].isnull())) # bool 行抽出


# In[ ]:


train_df_copy.head(3)


# In[ ]:


nullrow_age = train_df_copy['age'].isnull()
print(f"{train_df_copy[nullrow_age].head(10)}\n")


# train_df_copy, test_df_copy 上で欠損値補完

# In[ ]:


combine_copy =[train_df_copy, test_df_copy]

for dataset in combine_copy:
    print(f"{dataset.columns[0]}")
    for i in set(dataset['pclass']):
        df_i = dataset[dataset['pclass'] == i]
        for j in set(dataset['sex']):
            df_j =df_i[df_i['sex'] == j]
            age_target = df_j['age'].median() ## .dropna() が抜けてる
            print(f"{i}, {j}:\n{df_j[df_j['age'].isnull()].head(3)}\n{age_target} <--")
            
            print(f"{dataset.loc[(dataset['pclass'] == i) & (dataset['sex'] == j) & (dataset['age'].isnull()), 'age'].head(3)}\n")
            dataset.loc[(dataset['pclass'] == i) & (dataset['sex'] == j) & 
                        (dataset['age'].isnull()), 'age'] = age_target
    
    dataset['age'] = dataset['age'].astype(int)


# In[ ]:


print(f"{train_df_copy[nullrow_age].head(10)}")


# train_df 'pclass', 'sex', 'age' を可視化

# In[ ]:


print(f"{set(train_df['pclass'])}")
print(f"{set(train_df['sex'])}")

bins = 20
colors = ['r', 'b']
fig, axes = plt.subplots(1, 3, figsize=(14,2))
for i in set(train_df['pclass']):
#     print(f"\n{i}:")
    df_i = train_df[train_df['pclass'] == i]
    
    hist_max = 0
    for j in set(train_df['sex']):  # hist_max 取得のためだけのイテレート
        df_j = df_i[df_i['sex'] == j]
        hist, h_bins = np.histogram(df_j['age'].dropna(), bins=bins)
        print(f"{i}, {j}\n{hist}, {hist.max()}\n{h_bins}")
        if hist.max() > hist_max:
            hist_max = hist.max()
        print(f"hist_max: {hist_max}") # hist_max を取得して一旦ループ終了

#     print(f"{df_i.head(3)}")
    for j in set(train_df['sex']):
        ax = axes.ravel()[i-1]
#         print(f"{j}:")
        df_j = df_i[df_i['sex'] == j]
#         print(f"{df_j.head(2)}")
        df_j['age'].dropna().hist(ax=ax, color=colors[j], alpha=.3, bins=bins, label="sex: {}".format(j))
        ax.vlines(x=df_j['age'].median(), ymin=0, ymax=hist_max, color=colors[j])
        ax.vlines(x=df_j['age'].mean(), ymin=0, ymax=hist_max, color=colors[j], alpha=.7)
        ax.set_title("pclass: {}".format(i))
        ax.set_xlabel('age')
        ax.legend()


# Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.

# Pclass x Genderの組み合わせに基づいて推測されたAge値を格納する為の、空の配列を準備することから始めましょう。

# In[ ]:


guess_ages = np.zeros([2, 3])
guess_ages


# Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

# 今度は、Sex（0または1）とPclass（1,2,3）を繰り返し、6つの組み合わせのAgeの推測値を計算します。

# In[ ]:


print(f"train_df['age'].isnull().sum(): {train_df['age'].isnull().sum()}")
row_null_age = train_df['age'].isnull()
train_df.loc[row_null_age, :].head()


# In[ ]:


for id, dataset in enumerate(combine):
    print(f"dateset: {id}\n")
    for i in range(0, 2):
        print(f"{i}:")
        for j in range(0, 3):
            print(f" {j}:")
            guess_df_all = dataset[(dataset['sex'] == i) & (dataset['pclass'] == j+1)]
#             print(f"{guess_df_all.head()}")
            guess_df = guess_df_all['age'].dropna()
#             print(f"{guess_df.head()}, {type(guess_df)}")
            age_guess = guess_df.median()
#             print(f"{age_guess}")
            
            # Convert random age float to nearest .5 age
            print(f"{int( age_guess/0.5 + 0.5 ) * 0.5}\n")
            guess_ages[i, j] = int( age_guess/0.5 + 0.5 ) * 0.5

    print(f"{guess_ages}\n")
            
    for i in range(0, 2):
        for j in range(0, 3):
#             print(f"{i}: {j}\n{dataset.loc[(dataset['age'].isnull()) & (dataset['sex'] == i) & (dataset['pclass'] == j+1),\
#             'age'].head()}\n")
            dataset.loc[(dataset['age'].isnull()) & (dataset['sex'] == i) & (dataset['pclass'] == j+1), 'age'] = guess_ages[i, j]
    
    dataset['age'] = dataset['age'].astype(int)

train_df.loc[row_null_age, :].head()


# In[ ]:


train_df.age.isnull().sum()


# In[ ]:


train_df.loc[row_null_age, :].head(10)


# In[ ]:


(train_df['age'] != train_df_copy['age']).all()


# In[ ]:


print(f"{(train_df['age'] != train_df_copy['age']).sum()}")
train_df[train_df['age'] != train_df_copy['age']]


# Create new feature combining existing features
# 
# We can create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets.
# 

# 既存の特徴量を組み合わせて新しい特徴量を作成する
# 
# ParchとSibSpを組み合わせてFamilySizeとして新しい特徴量を作成できます。 これにより、データセットからParchとSibSpを削除できます。

# In[ ]:


print(f"{train_df.groupby(['sibsp']).size()}")
print(f"{train_df.groupby(['parch']).size()}")


# 行(passengerid)毎に sibsip と parch の値（兄弟の数と親子の数）を足したものを新しい特徴として採用する

# In[ ]:


train_df['familysize'] = train_df['sibsp'] + train_df['parch'] + 1 # ゼロにならないように 1 を足す
train_df.head()


# In[ ]:


for dataset in combine:
    dataset['familysize'] = dataset['sibsp'] + dataset['parch'] + 1
train_df[['sibsp', 'parch', 'familysize']].head()


# In[ ]:


train_df.groupby(['familysize']).mean()['survived'].sort_values(ascending=False)


# We can create another feature called IsAlone.
# <br>IsAloneという別の特徴量を作成することができます。

# In[ ]:


combine = [train_df, test_df]
for dataset in combine:
    dataset['isalone'] = 0
    dataset.loc[dataset['familysize'] == 1, 'isalone'] = 1 # 二つ上で兄弟の数も親子の数もゼロに1を足したので1が兄弟親子がゼロ
    print(f"{dataset.head()}")
train_df[['sibsp', 'parch', 'familysize', 'isalone']].head()


# In[ ]:


train_df[['isalone', 'survived']].groupby(['isalone']).mean()


# Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.<br>
# IsAloneが良さげなので、Parch、SibSp、およびFamilySize特徴量を削除します。

# In[ ]:


print(f"Before: {train_df.columns.values}")
if train_df.columns.isin(['sibsp', 'parch', 'familysize']).any():
    train_df = train_df.drop(['sibsp', 'parch', 'familysize'], axis=1)
if test_df.columns.isin(['sibsp', 'parch', 'familysize']).any():
    test_df = test_df.drop(['sibsp', 'parch', 'familysize'], axis=1)
    
print(f"{train_df.columns.values}")
print(f"{test_df.columns.values}")
train_df.head()


# 
# Completing a categorical feature
# 
# Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance.<br>
# 
# カテゴリカル特徴量を補完する
# 
# Embarked特徴量は、乗船港に基づいてS、Q、Cの値を取ります。 トレーニングデータセットには2つの欠損値がありますが、今回は最も一般的な出現で埋めることにします。

# In[ ]:


print(f"{train_df.embarked.isnull().sum()}")
train_df.groupby(['embarked']).size().sort_values(ascending=False)


# In[ ]:


freq_port = train_df['embarked'].dropna().mode()[0]
freq_port


# In[ ]:


combine = [train_df, test_df]
for dataset in combine:
    dataset['embarked'] = dataset['embarked'].fillna(freq_port)
print(f"{train_df.embarked.isnull().sum()}")
train_df.groupby(['embarked']).size().sort_values(ascending=False)


# In[ ]:


train_df[['embarked', 'survived']].groupby(['embarked']).mean().sort_values('survived', ascending=False)


# Converting categorical feature to numeric<br>
# We can now convert the EmbarkedFill feature by creating a new numeric Port feature.<br>
# カテゴリカル特徴量を数値に変換する<br>
# 欠損値を補完したので、Embarked特徴量を数値に変換出来るようになりました。

# In[ ]:


print("{}".format({v: i for i, v in enumerate(['S', 'C', 'Q'])}))
combine = [train_df, test_df]
if dataset.groupby(['embarked']).size().index.isin(['C', 'Q', 'S']).any():
    for dataset in combine:
        dataset['embarked'] = dataset['embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
train_df.groupby(['embarked']).size()
print(f"After: {train_df.groupby(['embarked']).size().index.values}")
train_df.head()


# Quick completing and converting a numeric feature
# 
# We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently for this feature. We do this in a single line of code.
# 
#  Note that we are not creating an intermediate new feature or doing any further analysis for correlation to guess missing feature as we are replacing only a single value. The completion goal achieves desired requirement for model algorithm to operate on non-null values.
# 
# We may also want round off the fare to two decimals as it represents currency.<br>
# 
# クイック補完と数値特徴量の変換
# 
# この特徴量の最頻値（最も頻繁に登場する値）を使用して、テストデータセットの値が不足している場合に、Fare特徴量を補完することができます。 これは1行のコードで行えます。
# 
#     補完する目的は、欠損値を埋めて、モデルアルゴリズムがある程度望ましい状態で動作することです。必要以上に欠損値の推測に時間を掛ける必要はありません。
# 
# 通貨を表しているので、運賃の小数点第二位以下を四捨五入する場合もあります。

# In[ ]:


null_fare_row = test_df['fare'].isnull()
print(f"Before: \n{test_df.loc[null_fare_row, :]}")
# print(f"{}")
test_df['fare'].fillna(test_df['fare'].dropna().median(), inplace=True)
print(f"After: \n{test_df.loc[null_fare_row, :]}")


# Let us create Age bands and determine correlations with Survived.

# Ageの区間(Band)を作成し、Survivedとの相関を見てみましょう。

# pd.cut と pd.qcut の比較

# In[ ]:


h_bins = 50
fig, axes = plt.subplots(figsize=(12,2))
train_df['age'].plot.hist(bins=h_bins, alpha=.3)
hist, n_bins = np.histogram(train_df['age'], bins=h_bins)
print(f"{hist.min()}, {hist.max()}")
print("")

bins=5
cut, cbins = pd.cut(train_df['age'], bins=bins, retbins=True)
print("cut: {}".format(pd.DataFrame(cut).groupby(['age']).size()))
print(f"cbins: {cbins}\n")
axes.vlines(x=cbins, ymin=hist.min(), ymax=hist.max(), color='y', alpha=.8, label='cut_bins')

cut, cbins = pd.cut(train_df['age'], bins=bins, retbins=True, labels=False)
print("cut( , labels=False): {}".format(pd.DataFrame(cut).groupby(['age']).size()))
print(f"cbins: {cbins}\n")

cut, cbins = pd.qcut(train_df['age'], q=bins, retbins=True)
print("qcut: {}".format(pd.DataFrame(cut).groupby(['age']).size()))
print(f"cbins: {cbins}")

axes.vlines(x=cbins, ymin=hist.min(), ymax=hist.max(), color='r', alpha=.8, label='qcut_bins')

axes.legend()


# 以下は実行しない。train_df と test_df で pd.cut( , retbins=True, labels=False) で取得した数字（indicators of the bins）を充てる。

# 先生は pd.cut を使っている

# In[ ]:


# train_df['ageband'] = pd.cut(train_df['age'], 5)
# train_df


# In[ ]:


# train_df[['ageband', 'survived']].groupby(['ageband']).mean()


# In[ ]:


# train_df.groupby(['ageband']).size()


# 最もスコアの高い age と fare のビニングの組み合わせを探す

# In[ ]:


train_df.head(3)


# In[ ]:


pipe_rf = Pipeline([('scl', StandardScaler()), ('est', RandomForestClassifier(n_estimators=100, random_state=0))])
pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=0))])


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

age_max, fare_max = 12, 10
age_bins, fare_bins = np.arange(2, age_max + 1), np.arange(2, fare_max + 1) # ビニングのリスト作成
print(f"{age_bins}\n{fare_bins}\n")

train_df_bins = train_df.copy() # train_df_bins という作業用コピー上でビニングを試す

age_list, fare_list, rf_acc_list, rf_f1_list, gb_acc_list, gb_f1_list = [], [], [], [], [], []
score_name = ['rf_acc', 'rf_f1', 'gb_acc', 'gb_f1']
best_score = 0
for age_bin in age_bins:
    bins = age_bin
    for fare_bin in fare_bins:
        q = fare_bin
        cut, cbins = pd.cut(train_df['age'], bins=bins, retbins=True, labels=False)
        train_df_bins['age'] = cut
        qcut, qbins = pd.qcut(train_df['fare'], q=q, retbins=True, labels=False)
        train_df_bins['fare'] = qcut

        train_df_bins['age_class'] = train_df_bins['age'] * train_df_bins['pclass']

        X_bin = train_df_bins.iloc[:, 1:]
        y_bin = train_df_bins.iloc[:, [0]]
        
        indices = np.arange(X_bin.shape[0])
        X_bin_train, X_bin_valid, y_bin_train, y_bin_valid, indices_train, indices_valid = train_test_split(
            X_bin, y_bin, indices, random_state=0)
        
        pipe_rf.fit(X_bin_train, y_bin_train.values.ravel())
        pipe_gb.fit(X_bin_train, y_bin_train.values.ravel())
        
        rf_acc = accuracy_score(y_bin_valid.values.ravel(), pipe_rf.predict(X_bin_valid))
        rf_f1 = f1_score(y_bin_valid.values.ravel(), pipe_rf.predict(X_bin_valid))
        
        gb_acc = accuracy_score(y_bin_valid.values.ravel(), pipe_gb.predict(X_bin_valid))
        gb_f1 = f1_score(y_bin_valid.values.ravel(), pipe_gb.predict(X_bin_valid))
        
        age_list.append(age_bin); fare_list.append(fare_bin); rf_acc_list.append(rf_acc); rf_f1_list.append(rf_f1)
        gb_acc_list.append(gb_acc); gb_f1_list.append(gb_f1)
        
        if max([rf_acc, rf_f1, gb_acc, gb_f1]) > best_score:
            best_score = max([rf_acc, rf_f1, gb_acc, gb_f1])
            best_indx = [rf_acc, rf_f1, gb_acc, gb_f1].index(max([rf_acc, rf_f1, gb_acc, gb_f1]))
            best_age_bin, best_fare_bin = age_bin, fare_bin
        
#         print(f"{bins}: {q}: \n{train_df.columns.values}\n{train_df_bins.columns.values}")
#         print(f"{train_df_bins.groupby(['age']).size()}")
#         print(f"{train_df_bins.groupby(['fare']).size()}")
#         print(f"{train_df_bins.groupby(['age_class']).size()}")

#         print(f"X_bin.head(5):\n{X_bin.head(5)}")
#         print(f"y_bin.head(5):\n{y_bin.head(5)}")
#         print(f"{X_bin.shape}, {y_bin.shape}\n{X_bin_train.shape}, {X_bin_valid.shape}, {y_bin_train.shape}, {y_bin_valid.shape}")
        
#         print(f"{rf_acc}\n{rf_f1}\n{gb_acc}\n{gb_f1}")
        print(f"{bins}: {q}: {[rf_acc, rf_f1, gb_acc, gb_f1]}")
#         print(f"{max([rf_acc, rf_f1, gb_acc, gb_f1])}, {[rf_acc, rf_f1, gb_acc, gb_f1].index(max([rf_acc, rf_f1, gb_acc, gb_f1]))}")
        
print(f"{best_age_bin}, {best_fare_bin}, {best_score}, {best_indx}, {score_name[best_indx]}")
# X_bin_train
# y_bin_train
# X_bin_valid
# y_bin_valid


# In[ ]:


len(age_list)
len(fare_list)
v_index = [(a, f) for a, f in zip(age_list, fare_list)]
print(f"{v_index[rf_acc_list.index(max(rf_acc_list))]}: {max(rf_acc_list)}\n{v_index[rf_f1_list.index(max(rf_f1_list))]}: {max(rf_f1_list)}\n{v_index[gb_acc_list.index(max(gb_acc_list))]}: {max(gb_acc_list)}\n{v_index[gb_f1_list.index(max(gb_f1_list))]}: {max(gb_f1_list)}")


# In[ ]:


fig, axes = plt.subplots(4, 1, figsize=(12,4))
for i, data in enumerate([rf_acc_list, rf_f1_list, gb_acc_list, gb_f1_list]):
    data = np.array(data)
    dindx = list(data).index(data.max())
    dmax = data.max() - data.min()
#     print(dindx)
#     print(list(data).index(dmax))
    ax=axes.ravel()[i]
    pd.DataFrame(data-data.min()).plot.bar(ax=ax, alpha=.7)
    ax.set_xticklabels('')
    ax.set_title(score_name[i])
    ax.plot(dindx, dmax, 'o', color='r', alpha=.5)
    print(f"{score_name[i]:<6}: {data.max():.4f}, {v_index[dindx]}")
axes.ravel()[3].set_xticklabels(v_index); fig.tight_layout()


# 以下を train_df に test_df に適用する。

# In[ ]:


print(f"best_age_bin: {best_age_bin}\nbest_fare_bin: {best_fare_bin}")


# Let us replace Age with ordinals based on these bands.

# これらの区間に基づいて年齢を序数に置き換えましょう。

# In[ ]:


combine = [train_df, test_df]
print(f"{train_df.groupby(['age']).size().shape[0]}")
if train_df.groupby(['age']).size().shape[0] == 71: # 変更前の年齢グループ数
    for i, dataset in enumerate(combine):
        cut, cbins = pd.cut(dataset['age'], bins=best_age_bin, retbins=True, labels=False)
        dataset['age'] = cut
        print(f"{i}: {dataset.groupby(['age']).size().shape[0]}")
        print("")


# In[ ]:


for i, dataset in enumerate(combine):
    print(f"{i}:\n{dataset.groupby(['age']).size()}")


# In[ ]:





# In[ ]:





# In[ ]:


for i, dataset in enumerate(combine):
    print(f"{i}:\n{dataset.head()}\n")


# AgeBand特徴量を削除します。

# In[ ]:


# if 'ageband' in train_df.columns:
#     train_df = train_df.drop(['ageband'], axis=1)
# train_df.head()


# In[ ]:


gby = train_df[['age', 'pclass', 'survived']].groupby(['age', 'pclass']).mean()
gby


# In[ ]:


pivot_mean = pd.pivot_table(train_df
              , index=['age']
              , columns=['pclass']
#               , values=['survived']
              , aggfunc={'survived': [np.mean, len]}
              , )
pivot_mean


# We can also create an artificial feature combining Pclass and Age.<br>
# PclassとAgeを組み合わせた人工的な特徴を作成することもできます。

# In[ ]:


combine = [train_df, test_df]
for dataset in combine:
    dataset['age*class'] = dataset['age'] * dataset['pclass']
    print(dataset[['age*class', 'age', 'pclass']].head())
train_df.loc[:, ['pclass', 'age', 'age*class']].head(10)
train_df.groupby(['pclass', 'age', 'age*class']).size()


# We can create FareBand.

# FareBand特徴量を作成することもできます。

# train_df と test_df で pd.cut( , retbins=True, labels=False) で取得した数字（indicators of the bins）を充てる。

# In[ ]:


print(f"{train_df['fare'].describe()}")
print(f"median: {train_df['fare'].median()}")
bins = 20
hist, bins = np.histogram(train_df['fare'], bins=bins)
# print(f"{hist}, {hist.min()}, {hist.max()}, {hist.mean()}\n{bins}, {bins.min()}, {bins.max()}")
fig, axes = plt.subplots(figsize=(8,2))
train_df.fare.plot.hist(bins=bins, alpha=.3)
# axes.plot(train_df['fare'].median(), 400, 'o')
# axes.hlines(y=hist.max(), xmin=bins.min(), xmax=bins.max(), colors='r', alpha=.5)
axes.vlines(x=train_df['fare'].median(), ymin=hist.min(), ymax=hist.max(), colors='b', label='median')
axes.legend()


# pd.cut と pd.qcut の比較

# In[ ]:


fig, axes = plt.subplots(figsize=(8,2))
bins = 20
train_df['fare'].plot.hist(bins=bins, color='grey', alpha=.3, label='')
hist, hbins = np.histogram(train_df['fare'], bins=bins)

bins = best_fare_bin
cut, cbins = pd.cut(train_df['fare'], bins=bins, retbins=True, labels=False) # 最大値と最小値の間を等間隔で分割する。
print(f"cut.head(3):\n{cut.head(3)}\ncbins: {cbins}\n") # 引数retbins=Trueで、ビン分割されたデータと境界値のリストを同時に取得できる。
print(f"{pd.DataFrame(cut).groupby(['fare']).size()}\n")
axes.vlines(x=cbins, ymin=hist.min(), ymax=hist.max()/3, colors='r', alpha=.5, label='cut_bins')

cut, cbins = pd.qcut(train_df['fare'], q=bins, retbins=True, labels=False) # 各ビンに含まれる個数（要素数）が等しくなるようにビニング
print(f"qcut.head(10): # 確認用\n{cut.head(10)}\ncbins: {cbins}\n")
print(f"{pd.DataFrame(cut).groupby(['fare']).size()}\n")
axes.vlines(x=cbins, ymin=hist.min(), ymax=hist.max()/3*2, colors='b', alpha=.5, label='qcut_bins')
axes.legend()


# In[ ]:


# train_df[['fareband', 'survived']].groupby(['fareband']).mean().sort_values('survived', ascending=False)


# Convert the Fare feature to ordinal values based on the FareBand.

# （FareBandに基づいてFare特徴量を序数に変換します。）

# In[ ]:


train_df.head()


# ここで実行

# In[ ]:


best_fare_bin


# In[ ]:


combine = [train_df, test_df] 
q = best_fare_bin
train_df['fare'].isin([n for n in np.arange(q)]).all()
if not train_df['fare'].isin([n for n in np.arange(q)]).all():
    for i, dataset in enumerate(combine):
        qcut, qbins = pd.qcut(dataset['fare'], q=q, retbins=True, labels=False)
        print(f"{i}:\n{pd.DataFrame(qcut).groupby(['fare']).size()}\n")
        dataset['fare'] = qcut
        dataset['fare'] = dataset['fare'].astype(int)
    
# if 'fareband' in train_df.columns:
#     train_df = train_df.drop(['fareband'], axis=1)


# In[ ]:


for i, dataset in enumerate(combine):
    print(f"{i}:\n{dataset.groupby(['fare']).size()}\n")


# In[ ]:


train_df.head(10)


# And the test dataset.

# テストセットは以下の通り。

# In[ ]:


test_df.head()


# ヒートマップを作ってみる

# In[ ]:


train_df.corr()


# In[ ]:


sns.heatmap(train_df.corr(), cmap='PuBu', annot=True, fmt='.2f')


# In[ ]:


corr_id = [id for id in train_df.corr()['survived'].index if id != 'survived'] # 'survived' 以外のインデクスを取得
corr_id


# In[ ]:


print(f"{train_df.corr().loc['survived', corr_id]}")
fig, axes = plt.subplots(figsize=(7,3))
train_df.corr().loc['survived', corr_id].plot.bar(ax=axes)
# axes.axhline(y=0, color='silver', linestyle='-', linewidth=1)
axes.set_title('corrcoef to survived')
axes.set_xticklabels([id.upper() for id in corr_id])
# axes.grid()
fig.tight_layout()


# 
# Model, predict and solve
# 
# Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:
# 
#     Logistic Regression
#     KNN or k-Nearest Neighbors
#     Support Vector Machines
#     Naive Bayes classifier
#     Decision Tree
#     Random Forrest
#     Perceptron
#     Artificial neural network
#     RVM or Relevance Vector Machine
# 
# 

# In[ ]:


X_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, [0]]
X_test = test_df.drop('passengerid', axis=1).copy()
X_train.shape, y_train.shape, X_test.shape


# In[ ]:


X_train.head(3)
# y_train


# In[ ]:


X_test.head(3)


# 
# 
# Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. Reference Wikipedia.
# 
# Note the confidence score generated by the model based on our training dataset.
# 

# In[ ]:


from sklearn.metrics import precision_score, recall_score
logreg = LogisticRegression()
logreg.fit(X_train, y_train.as_matrix().ravel())
y_pred = logreg.predict(X_test)
print(f"{logreg.score(X_train, y_train):.4f}")
print(f"{accuracy_score(y_train.as_matrix().ravel(), logreg.predict(X_train)):.4f}") # 同じ
print(f"{f1_score(y_train.as_matrix().ravel(), logreg.predict(X_train)):.4f}")
print(f"{confusion_matrix(y_train.as_matrix().ravel(), logreg.predict(X_train))}")
print(f"{precision_score(y_train.as_matrix().ravel(), logreg.predict(X_train)):.4f}")
print(f"{recall_score(y_train.as_matrix().ravel(), logreg.predict(X_train)):.4f}")


# We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.
# 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

# In[ ]:


print(f"{logreg.coef_}, {logreg.coef_.shape}\n{X_train.columns}, {X_train.columns.shape}")
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['feature']
coeff_df['correlation'] = logreg.coef_.ravel()
coeff_df.sort_values('correlation', ascending=False)


# 上記よりもこっちの方が素敵

# In[ ]:


coeff_df = pd.DataFrame([X_train.columns, logreg.coef_.ravel()], index=['feature', 'correlation']).T

# # 以下でも可
# coeff_df = pd.DataFrame({
#       'feature': X_train.columns
#     , 'correlation': logreg.coef_.ravel()}
#     , columns=['feature', 'correlation'] )

fig, axes = plt.subplots(figsize=(8,2))
coeff_df.plot.bar(ax=axes)
axes.set_xticklabels(coeff_df['feature'], rotation=0)

coeff_df.T


# Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
# 
# Inversely as Pclass increases, probability of Survived=1 decreases the most.
# 
# This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
# 
# So is Title as second highest positive correlation.

# Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. Reference Wikipedia https://en.wikipedia.org/wiki/Support_vector_machine .
# 
# Note that the model generates a confidence score which is higher than Logistics Regression model.

# In[ ]:


[c for c in X_train.columns.values], [c for c in X_test.columns.values]


# 面倒なので最初の LogisticRegression からイテレートする

# In[ ]:


logreg = LogisticRegression()
svc = SVC()
knn = KNeighborsClassifier()
gaussian = GaussianNB()
perceptron = Perceptron(max_iter=5)
linear_svc = LinearSVC()
sgd = SGDClassifier()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
gb = GradientBoostingClassifier()

estimators = [logreg, svc, knn, gaussian, perceptron, linear_svc, sgd, decision_tree, random_forest, gb]
est_names = ['LogisticRegression', 'SVC', 'KNeighborsClassifier', 'GaussianNB', 'Perceptron', 'LinearSVC',
             'SGDClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']


# In[ ]:


acc_scores, f1_scores = [], []
for i, est in enumerate(estimators):
    est.fit(X_train, y_train.values.ravel())
    score_acc = accuracy_score(y_train.as_matrix().ravel(), est.predict(X_train))
    score_f1 = f1_score(y_train.as_matrix().ravel(), est.predict(X_train))
    acc_scores.append(score_acc)
    f1_scores.append(score_f1)
        
    print(f"{est_names[i]}:")
#     print(f"score: {est.score(X_train, y_train):>10.4f}")
    print(f"acc_score: {score_acc:.4f}")
    print(f"f1_score:  {score_f1:.4f}")
    print(f"{confusion_matrix(y_train.as_matrix().ravel(), est.predict(X_train))}")
    print(f"pre_score: {precision_score(y_train.as_matrix().ravel(), est.predict(X_train)):.4f}")
    print(f"rec_score: {recall_score(y_train.as_matrix().ravel(), est.predict(X_train)):.4f}\n")    


# In[ ]:


df_score = pd.DataFrame([est_names, acc_scores, f1_scores], index=['est_name', 'accuracy', 'f1_score'], ).T # 転置
df_score # とりあえずリストからデータフレームを作ってみる


# In[ ]:


df_best_acc = df_score[df_score['accuracy'] == df_score['accuracy'].max()] # df_score['accuracy'].max() と一致のデータフレーム作成
best_acc_name = df_best_acc.iloc[0, 0] # 
print(f"best_acc_name ({best_acc_name}): {df_best_acc.iloc[0, 1]}")
best_acc_index = df_best_acc.index.values[0]; print(f"best_acc_index: {best_acc_index}") # インデクスを取得して
best_acc_estimator = estimators[best_acc_index]; print(f"best_acc_estimator: \n{best_acc_estimator}") # estimators[インデクス]
df_best_acc


# 上記の estimator 取得が格好悪い。以下なら一発。

# In[ ]:


print(f"{acc_scores.index(max(acc_scores))}, {est_names[acc_scores.index(max(acc_scores))]}: {max(acc_scores)}")
best_acc_est = estimators[acc_scores.index(max(acc_scores))]  # acc_scores の max の index を取得して estimators にぶつける

best_acc_est


# 上記で学習できる。こっちの方が簡潔。

# In[ ]:


best_acc_est.fit(X_train, y_train.as_matrix().ravel())


# グラフは逆にデータフレームの方が便利。特に .bar()

# In[ ]:


fig, axes = plt.subplots(3, 1, figsize=(10,3)) # 三列作って
h_col = [n for n in df_score.columns if n != 'est_name'] # 'est_name' を除いた数値列の ['accuracy', 'f1_score'] 
print(f"{h_col}")
for i, col in enumerate(h_col): # 1列と2列をそれぞれ描画
    ax = axes.ravel()[i]
#     print(df_score[col])
    df_score[col].plot.bar(ax=ax, color='b')
    ax.set_title(col)
    ax.set_xticklabels('')
df_score.plot.bar(ax=axes.ravel()[2]) # 3列目は一緒に描画
axes.ravel()[2].set_xticklabels([n[:11] for n in df_score['est_name']], rotation=0)
fig.tight_layout()


# In[ ]:


y_pred = best_acc_est.predict(X_test)
y_pred.shape


# In[ ]:


test_df['passengerid'].shape


# In[ ]:


submission = pd.DataFrame({
                    'PassengerId': test_df['passengerid'],
                    'Survived': y_pred,
                })
submission.to_csv('submission.csv', index=False)
submission


# ### テキスト通りはここまで。ここからは Pipeline, GridSearch, StandardScaler 等を使ってより良いスコアを目指す。

# In[ ]:


print(f"X_train.shape: {X_train.shape}\ny_train.shape: {y_train.shape}\nX_test.shape:  {X_test.shape}\n")
print(f"{test_df[['passengerid']].head(3)}\n")
print("{}\n{}".format([c for c in X_train.columns], [c for c in X_test.columns]))
X_train.head()


# In[ ]:


# X_train.isnull().sum()


# In[ ]:


# X_test.isnull().sum()


# In[ ]:


# X_train.dtypes


# In[ ]:


# X_test.dtypes


# 欠損値補完済み、カテゴリカル変数の数値への変換済み。X_train と y_train で holdout を行う。 Pipeline を作成し、GridSearch を行う。RFE はおそらく不要。

# In[ ]:


print(f"X_train.shape: {X_train.shape}\ny_train.shape: {y_train.shape}\nX_test.shape:  {X_test.shape}")
indices = np.arange(X_train.shape[0])
print(f"{indices.shape}")
X_fin_train, X_fin_valid, y_fin_train, y_fin_valid, indices_train, indices_valid = train_test_split(
    X_train, y_train, indices, random_state=0)
print(f"shapes of: X_fin_train, X_fin_valid, y_fin_train, y_fin_valid, indices_train, indices_valid; \n{X_fin_train.shape}, {X_fin_valid.shape}, {y_fin_train.shape}, {y_fin_valid.shape}, {indices_train.shape}, {indices_valid.shape}")


# In[ ]:


X_train


# In[ ]:


pipe_logreg = Pipeline([('scl', StandardScaler()), ('est', LogisticRegression(random_state=0))])
pipe_svc = Pipeline([('scl', StandardScaler()), ('est', SVC(random_state=0))])
pipe_knn = Pipeline([('scl', StandardScaler()), ('est', KNeighborsClassifier())])
pipe_gaussian = Pipeline([('scl', StandardScaler()), ('est', GaussianNB())])
pipe_perceptron = Pipeline([('scl', StandardScaler()), ('est', Perceptron(max_iter=5, random_state=0))])
pipe_linear_svc = Pipeline([('scl', StandardScaler()), ('est', LinearSVC(random_state=0))])
pipe_sgd = Pipeline([('scl', StandardScaler()), ('est', SGDClassifier(max_iter=1000, tol=1e-3, random_state=0))])
pipe_decision_tree = Pipeline([('scl', StandardScaler()), ('est', DecisionTreeClassifier(random_state=0))])
pipe_rf = Pipeline([('scl', StandardScaler()), ('est', RandomForestClassifier(random_state=0))])
pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=0))])
pipe_xgb = Pipeline([('scl', StandardScaler()), ('est', XGBClassifier(random_state=0))])

pipe_xgb.named_steps


# In[ ]:


pipes =[pipe_logreg, pipe_svc, pipe_knn, pipe_gaussian, pipe_perceptron, pipe_linear_svc, pipe_sgd, 
        pipe_decision_tree, pipe_rf, pipe_gb, pipe_xgb]


# In[ ]:


pipe_xgb.named_steps['est']


# In[ ]:


param_grid_logreg = {'est__C': [0.05, 0.1, 1.0, 10.0, 100.0],
                    'est__penalty': ['l1', 'l2']}
param_grid_svc = {
#                 'est__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'est__C': [1, 10, 100, 1000],
                'est__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

param_grid_knn = {'est__n_neighbors': range(1,101),
                  'est__weights': ['uniform', 'distance']}

param_grid_gaussian = {}
param_grid_perceptron = {'est__penalty': [None, 'l2', 'l1', 'elasticnet'],
                'est__alpha': [0.00001, 0.0001, 0.001, 0.01],}

param_grid_linear_svc = {
#                 'est__penalty': ['l1', 'l2'],
#                 'est__loss': ['hinge', 'squared_hinge'],
                'est__dual': [True, False],
                'est__tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                'est__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]}

param_grid_sgd = {'est__loss': ['hinge', 'log', 'modified_huber'],
                'est__penalty': ['none', 'l2', 'l1', 'elasticnet'],
}

param_grid_decision_tree = {'est__criterion': ['gini', 'entropy'],
                    'est__max_depth': range(1, 11),
                    'est__min_samples_split': range(2, 21),
                    'est__min_samples_leaf': range(1, 21),}

param_grid_rf = {
                'est__n_estimators': [10, 50, 100, 150, 200],
                'est__criterion': ['gini', 'entropy'],
                'est__max_features': np.arange(0.05, 1.01, 0.05),
}

param_grid_gb = {
                'est__n_estimators': [50, 100, 150],
                'est__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                'est__max_depth': range(1, 11),
#                 '': [],
}

param_grid_xgb = {
#       'est__n_estimators': [100]
#     , 'est__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.] 
     'est__learning_rate': [1e-1, 0.5, 1.] 
#     , 'est__max_depth': range(1, 11) 
    , 'est__max_depth': range(1, 11, 2) 
#     , 'est__min_child_weight': range(1, 21) 
    , 'est__min_child_weight': range(1, 11) 
    , 'est__subsample': np.arange(0.05, 1.01, 0.05) 
#     , 'est__nthread': [1] 
}


# In[ ]:


param_grids = [param_grid_logreg, param_grid_svc, param_grid_knn, param_grid_gaussian, param_grid_perceptron, param_grid_linear_svc, 
              param_grid_sgd, param_grid_decision_tree, param_grid_rf, param_grid_gb, param_grid_xgb]


# In[ ]:


import time
start_time = time.time()

best_score, best_acc_score, best_f1_score, best_params, best_estimator = [], [], [], [], []
for i, pipe in enumerate(pipes):
    i_time = time.time()
    param_grid = param_grids[i]
    gs = GridSearchCV(pipe, param_grid, cv=3)
    gs.fit(X_fin_train, y_fin_train.as_matrix().ravel())
    best_score.append(gs.best_score_)
    acc_score = accuracy_score(y_fin_valid.as_matrix().ravel(), gs.predict(X_fin_valid))
    best_acc_score.append(acc_score)
    f_score = f1_score(y_fin_valid, gs.predict(X_fin_valid))
    best_f1_score.append(f_score)
    best_params.append(gs.best_params_)
    best_estimator.append(gs.best_estimator_)
    print(f"{i}:\n{pipe.named_steps['est']}")
    print(f"gs.best_score_: {gs.best_score_:.4f}\naccuracy_score: {acc_score:.4f}\nf1_score_valid: {f_score:.4f}")
    print(f"{gs.best_params_}")
    print(f"{gs.best_estimator_.named_steps['est']}")
    
    print(f"{time.time() - i_time:.2f} sec.\n{time.time() - start_time:.2f} sec.\n")


# In[ ]:


est_names = ['LogisticRegression', 'SVC', 'KNeighbors', 'GaussianNB', 'Perceptron', 'LinearSVC', 
             'SGDClassifier', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'XGBClassifier']


# In[ ]:


scores_df = pd.DataFrame([best_score, best_acc_score, best_f1_score]
            , index=['best_score', 'best_acc_score', 'best_f1_score']
            , columns=[c[:10] for c in est_names]).T
scores_df.to_csv('scores_df.csv')
scores_df.sort_values(by=['best_score'], ascending=False)


# In[ ]:


scores_df


# In[ ]:


print(f"max(best_score): {max(best_score)}")
print(f"best_score.index(max(best_score)): {best_score.index(max(best_score))}")
print(f"{best_params[best_score.index(max(best_score))]}")
print(f"est_names[best_score.index(max(best_score))]: {est_names[best_score.index(max(best_score))]}")
final_pipe = best_estimator[best_score.index(max(best_score))]
final_pipe.named_steps['est']


# 上記 final_pipe で X_test から predict

# In[ ]:


# その前に joblib でディスクに保存しておく
from sklearn.externals import joblib
for i, pipe in enumerate(best_estimator):
#     print(f"{i}: {est_names[i]}\n{pipe}\n")
    joblib.dump(pipe, est_names[i] + '.pkl')


# In[ ]:


# ディスクから以下のように呼び出してもオーケー
best_name = est_names[best_score.index(max(best_score))]
# joblib_load_best = joblib.load('../input/' + best_name + '.pkl')
joblib_load_best = joblib.load(best_name + '.pkl')

joblib_load_best.named_steps['est']


# predict

# In[ ]:


print(f"X_test.shape: {X_test.shape}")
final_pred = final_pipe.predict(X_test)
print(f"final_pred.shape: {final_pred.shape}")
print(f"test_df['passengerid'].shape: {test_df['passengerid'].shape}")


# In[ ]:


submission_final = pd.DataFrame({
    'PassengerId': test_df['passengerid'], 
    'Survived': final_pred
    })
submission_final.to_csv('submission_final_0010_020.csv', index=False)
submission_final


# GradientBoosting.predict()

# In[ ]:


print(f"{best_estimator[9].named_steps['est']}")

pd.DataFrame({
    'PassengerId': test_df['passengerid'], 
    'Survived': best_estimator[9].predict(X_test)
}).to_csv('submission_final_0010_020_GB.csv', index=False)


# In[ ]:


n_to_get = 4
fig, axes = plt.subplots(1, n_to_get, figsize=(16,3))

e_list = np.arange(len(best_estimator))[-n_to_get:]
print(f"{e_list}")

for h, i in enumerate(e_list):
    ax = axes.ravel()[h]
    df_imp = pd.DataFrame(
        best_estimator[i].named_steps['est'].feature_importances_,
        index = X_train.columns ,
        columns=['f_imp'])
    df_imp.iloc[::-1, :].plot.barh(ax=ax); ax.set_title(est_names[i])
#     df_imp.plot.barh(ax=ax); ax.set_title(est_names[i])
    ax.set_yticklabels([c.upper()[-8:] for c in X_train.columns[::-1]])
#     print(f"{i}: {est_names[i]}\n{df_imp.iloc[::-1, :]}\n")
fig.tight_layout()


# ### Partial Dependence Plots

# In[ ]:


pdp_pipe = best_estimator[9] # GradientBoostingClassifier
df_imp = pd.DataFrame(pdp_pipe.named_steps['est'].feature_importances_, index=X_fin_train.columns, columns=['importance'])
fig, axes = plt.subplots(figsize=(8, 2))
df_imp.plot.bar(ax=axes); axes.set_xticklabels([i.upper()[:6] for i in df_imp.index])
fig.tight_layout()


# In[ ]:


from sklearn.ensemble.partial_dependence import plot_partial_dependence
df_sort = df_imp.reset_index().sort_values(by=['importance'], ascending=False)
fig, axes = plt.subplots(figsize=(12,6))
plot_partial_dependence(pdp_pipe.named_steps['est'], X_fin_train, features=df_sort.index, feature_names=df_sort['index'], ax=axes)
print(f"{df_sort}")
fig.tight_layout()


# In[ ]:


# pd.DataFrame(gs.cv_results_)


# In[ ]:




