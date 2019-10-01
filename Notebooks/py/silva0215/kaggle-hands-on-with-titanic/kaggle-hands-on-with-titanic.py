#!/usr/bin/env python
# coding: utf-8

# ### 構成
# 1. 課題の理解
# 2. データの理解
# 3. データ前処理
# 4. モデルの作成
# 5. 検証・評価
# 6. 結果提出

# ### 1. 課題の理解
# 目的はタイタニックの生存者を予測することです。タイタニック号には、十分な救命艇がなかったため、多くの命が失われたと言われており、2224人の乗客と乗組員の内、1502人が死亡しました。また、女性、子供、上位階級など、他の人より生き残る可能性が高いグループもあったようです。
# 
# 生き残れた最も重要な理由を考え、どの乗客が生き残ったかを予測します。

# ### 2.データの理解
# まずは、必要なPythonのライブラリをインポートします。

# In[ ]:


# 警告非表示
import warnings
warnings.filterwarnings('ignore')

# 数値計算のモジュールと行列データ処理のモジュール
import numpy as np
import pandas as pd

# モデルのアルゴリズム（ロジスティック回帰）
from sklearn.linear_model import LogisticRegression

# モデル作成で使用するヘルパー
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# 可視化のモジュール
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# 可視化の設定
get_ipython().magic(u'matplotlib inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# 次に可視化で使うヘルパー関数を定義します。

# In[ ]:


def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )


# 学習データとテストデータをロードします。

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

full = train.append( test , ignore_index = True )

# 学習データ
titanic = full[ :891 ]

del train , test

print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)


# データの基礎分析と可視化をしていきます。
# 
# まずは、どのようなデータかを確認します。各カラムの意味は配布資料に記載しています。

# In[ ]:


titanic.head()


# 欠損値を確認します。

# In[ ]:


titanic.isnull().sum()


# ”Age”、”Cabin”、”Embarked”に欠損値があるようですね。あとで補完します。
# 
# 次に、基本統計量を算出します。

# In[ ]:


titanic.describe()


# 算出結果の意味（上から）
# * 件数 (count)
# * 平均値 (mean)
# * 標準偏差 (std)
# * 最小値(min)
# * 第一四分位数 (25%)
# * 中央値 (50%)
# * 第三四分位数 (75%)
# * 最大値 (max)

# 特徴量の相関関係を表すヒートマップで、どの変数が重要かを探します。

# In[ ]:


plot_correlation_map( titanic )


# 例えば、年齢と生存率の関係がありそうだとなったら、その関係を詳しく見たりします。

# In[ ]:


plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )


# ### 3.データ準備
# カテゴリ変数は、数値変数に変換する必要があります。（Embarked、Pclass、Sex）
# 新しい数値変数のことをダミー変数と言います。

# In[ ]:


sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )


# In[ ]:


embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
embarked.head()


# In[ ]:


pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )
pclass.head()


# 次に、欠損値を入力します。今回は平均を欠損値として入力します。

# In[ ]:


imputed = pd.DataFrame()

imputed[ 'Age' ] = full.Age.fillna( full.Age.mean() )

imputed[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )

imputed.head()


# 次は、特徴量エンジニアリングです。予測精度を高めるために新たな特徴量を作成します。

# 乗客の名前からタイトル（Mr.やMs.など）を抽出します。

# In[ ]:


title = pd.DataFrame()
title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

# タイトルのマッピング
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )

title.head()


# 客室番号から客室カテゴリを抽出します。

# In[ ]:


cabin = pd.DataFrame()

cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )

cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )

cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )

cabin.head()


# チケット番号から、チケットクラスを抽出します。

# In[ ]:


# チケットクラスを抽出する関数
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()

ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )

ticket.shape
ticket.head()


# 次に、家族サイズとそのカテゴリをParchとSibSpを使って作成します。

# In[ ]:


family = pd.DataFrame()

family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1

family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

family.head()


# 最後にモデリングのためのデータセットを構築します。

# 以下の中から特徴量とする変数を選択します。
# * imputed
# * embarked
# * pclass
# * sex
# * family
# * cabin
# * ticket

# 今回は、"imputed"、"embarked"、"cabin"、"sex"を選択することにします。（"imputed"は"Age"と"Fare"）

# In[ ]:


full_X = pd.concat( [ imputed , embarked , cabin , sex ] , axis=1 )
full_X.head()


# データを学習データ、検証用データ、テストデータに分けます。

# In[ ]:


# 学習データ
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
# テストデータ
test_X = full_X[ 891: ]

# valid_Xとvalid_yは学習データからさらに検証用データに分割したもの
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )

print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)


# * train_X：学習データ
# * train_y：学習データの"Survived"
# * valid_X：検証用データ
# * valid_y：検証用データの"Survived"

# ### 4.モデルの作成
# モデルを選択し、学習データを使用して学習させ、テストデータを使用してモデルのパフォーマンスを評価します。
# 
# 今回はロジスティック回帰を選択します。

# In[ ]:


model = LogisticRegression()


# 学習データ（train_Xとtrain_y）を使ってモデルに学習させます。

# In[ ]:


model.fit( train_X , train_y )


# ### 5.検証・評価
# モデルのパフォーマンスを検証します。
# 
# 検証用のデータ（valid_Xとvalid_y）でモデルの精度を評価します。
# 
# 精度とは、「学習したモデルで生存かどうかを予測した値」と「実際の値」を比べた結果のことです。
# 
# 学習データを使用したときの精度と検証用データを使用したときの精度と比較します。
# 
# ここの差が大きい場合、過学習の可能性が高いです。

# In[ ]:


# scoreで精度を確認
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))


# また、交差検証と呼ばれる方法で、性能がよくなるモデルのパラメータを自動的に選択することや、再帰的特徴除去と呼ばれる方法で、性能がよくなる最適な数の特徴量を自動的に選択し学習させることもできます。

# In[ ]:


rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )
rfecv.fit( train_X , train_y )


# それでは、作ったモデルを使って実際に予測してみましょう。

# In[ ]:


test_Y = model.predict( test_X )
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test['Survived'] = test['Survived'].astype(int)
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )


# 予測結果ができたので、提出します。
