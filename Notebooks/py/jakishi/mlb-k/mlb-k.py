#!/usr/bin/env python
# coding: utf-8

# This Kernel is the report of assignment in August, 2018. Goals and TODOs are described as follows.  
# 
# ＝＝＝＝＝＝課題＝＝＝＝＝＝  
# Kaggle(カグル)のTitanicにチャレンジして頂きますが、  
# 機械学習を行う前にデータの特徴を調べ、アタリを付ける練習をして頂きます。
# 次回の勉強会で各自に簡単に発表して頂きます。
# 
# 
# ＜課題内容＞*(Assignment)*
# 
# (1) KaggleのTitanicチャレンジの内容を理解する*(Goal)*  
#      Overview, Data, Discussionタブの内容を読んで下さい  
# 　　https://www.kaggle.com/c/titanic
# 
# 
# (2) データの中身を見て、データの特徴を調べる*(TODOs) * 
# - 先ずは、学習用データ(train.csv)のみでよいです。
# - データ件数、カラム名、データ型、基礎集計、相関等を確認し、レポートに纏めて下さい。
# - 目的変数:"Survived"(0 or 1), 説明変数の候補：左記変数以外すべて  
#     - レポーティング形式は問いません。説明できれば何でもよいです。
# - 今回はExcelやパワポ等でも可です。    
#    - Rをお使いになる方は、Rソースコードからhtmlドキュメントを生成するサンプル(sample.Rmd)を置いておきます。  
#    - Python詳しくないですが、Pythonをお使いになる方はJupyter Notebookがよいかもしれません。  
# 
# 
# ＜タイタニックデータセット(下記3ファイル; 添付のRin/フォルダに格納)＞*(Data sets)*  
# 1. training set (train.csv)：学習用データ
# 2. test set (test.csv)：予測用データ
# 3. submissionサンプル (gender_submission.csv)：予測結果提出用の雛型ファイル 
# 
# 
# 以上、よろしくお願い致します。  
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 
# ### Personal takeaways:
# - Get used to use Kernel and Phython basic machine learning commands.
# - Understand how to read the statistic results. 

# In[ ]:


# This part of codes are originally provided by Kaggle Kernel for easy to start.
# ここのコードは、はじめやすくするために、元からKaggle Kernelによって提供されているものです。

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


#  > (2) データの中身を見て、データの特徴を調べる*(TODOs) * 
#  > - 先ずは、学習用データ(train.csv)のみでよいです。

# In[ ]:


# Reading data set of "train.csv" as `train`.
train = pd.read_csv("../input/train.csv")


# > - データ件数、カラム名、データ型、基礎集計、相関等を確認し、レポートに纏めて下さい。

# In[ ]:


# Have a look "train" data set.
train.head()

# 'SibSp': # of siblings / spouses aboard the Titanic 
# 'Parch' # of parents / children aboard the Titanic

# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)

# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.


# In[ ]:


# > - データ件数

# Length of 'train' data set.
len(train) # --> 891


# In[ ]:


# (Variation of outputting length of data set)
# Length of 'traing' data set and # of columns.
train.shape # --> (891, 12)


# In[ ]:


# > カラム名、データ型

# Columns vs. Data type
train.info()

# 'int64' : Integer
# 'float64' : Decimal (number)
# 'object' : String (in this data set such as "Braund, Mr. Owen Harris", "male" and so on.)


# In[ ]:


# > 基礎集計

# Basic statistics
train.describe()

# It gives the statistical results on ONLY numerical data type columns.
# 'count' : # of data
# 'mean' : average
# 'std' : standard deviation
#   "データや確率変数の散らばり具合（ばらつき）を表す数値のひとつ。" -- Wikipedia
#   "a measure that is used to quantify the amount of variation or dispersion of a set of data values." -- Wikipedia
# 'min', 'max' : Minimum/Maximum value of data set 
# '25%', '50%', '75%' : quartiles ('50%' is call in other say 'median'(中央値).)
#   "quartiles are the three cut points that will divide a dataset into four equal-sized groups." -- Wikipedia


# In[ ]:


# Grab the profile of data set with basic statistics.

# On 'Age' 
# mean : 29.699118
# std : 14.526497

age_mean_plus_std = 29.699118 + 14.526497
age_mean_minus_std = 29.699118 - 14.526497
age_mean_plus_2xstd = 29.699118 + 2*14.526497
age_mean_minus_2xstd = 29.699118 - 2*14.526497

print(age_mean_minus_std) #--> 15.172620999999998
print(age_mean_plus_std) #--> 44.225615
print(age_mean_minus_2xstd) #--> 0.6461239999999968
print(age_mean_plus_2xstd) #--> 58.752112

# The age range between 16 years old and 44 years old convers 68% of people on board.
# The age range between 1 year old and 58 years old convers 95% of people on board.


# In[ ]:


# On 'SibSp' # of siblings / spouses aboard the Titanic 
# mean : 0.523008
# std : 1.102743

sibsp_mean = 0.523008
sibsp_std = 1.102743
sibsp_mean_plus_std = sibsp_mean + sibsp_std
sibsp_mean_minus_std = sibsp_mean - sibsp_std
sibsp_mean_plus_2xstd = sibsp_mean + 2*sibsp_std
sibsp_mean_minus_2xstd = sibsp_mean - 2*sibsp_std

print(sibsp_mean_plus_std) #--> 1.6257510000000002
print(sibsp_mean_minus_std) #--> -0.579735
print(sibsp_mean_plus_2xstd) #--> 2.728494
print(sibsp_mean_minus_2xstd) #--> -1.6824780000000001

# 68% of people aboard Titanic have no SibSp or 1.
# 95% of people aboard Titanic have no SibSp, 1, 2.


# In[ ]:


# On 'Parch' # of parents / children aboard the Titanic
# mean : 0.381594
# std : 0.806057

parch_mean = 0.381594
parch_std = 0.806057
parch_mean_plus_std = parch_mean + parch_std
parch_mean_minus_std = parch_mean - parch_std
parch_mean_plus_2xstd = parch_mean + 2*parch_std
parch_mean_minus_2xstd = parch_mean - 2*parch_std

print(parch_mean_plus_std) #--> 1.187651
print(parch_mean_minus_std) #--> -0.42446300000000003
print(parch_mean_plus_2xstd) #--> 1.993708
print(parch_mean_minus_2xstd) #--> -1.23052

# 68% of people aboard Titanic have no Parch or 1.
# 95% of people aboard Titanic have no Parch, 1, (2?). -> Question: "1.993708" can be counted as "2"?


# In[ ]:


# On 'Fare'
# mean : 32.204208
# std : 49.693429

fare_mean = 32.204208
fare_std = 49.693429
fare_mean_plus_std = fare_mean + fare_std
fare_mean_minus_std = fare_mean - fare_std
fare_mean_plus_2xstd = fare_mean + 2*fare_std
fare_mean_minus_2xstd = fare_mean - 2*fare_std

print(fare_mean_plus_std) #--> 81.897637
print(fare_mean_minus_std) #--> -17.489221
print(fare_mean_plus_2xstd) #--> 131.591066
print(fare_mean_minus_2xstd) #--> -67.18265

# 68% of people aboard Titanic paid less than 81 [pounds].
# 95% of people aboard Titanic paid less than 131 [pounds].

# The third quartile of Fare (75%) is "31.000000".
# 75% of people aboard Titanic paid less than and equal to 31 [pounds].


# In[ ]:


# > 相関等

# As preparation of calculating correlation, we need to clean data set of 'train' in terms of follws.
# - There are some 'N/A'(not available). It can be calculated corrlation.
# - There are some 'object' data type columns such as "Sex" and "Embarked". Converting into categorical variables with Integer.

# See the data set condition of "N/A".

# Define the function kesson_table().
# Usage: kesson_table(dataset)
def kesson_table(df): 
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : '欠損数', 1 : '%'})
        return kesson_table_ren_columns


# In[ ]:


# Run kesson_table().
kesson_table(train)

# Attribution: # of N/A | %
# Age: 177 | 19.865320
# Cabin: 687 | 77.104377
# Embarked: 2 | 0.224467


# In[ ]:


# Filling "N/A" with "available" data.

# On "Age", filling with median.
train["Age"] = train["Age"].fillna(train["Age"].median())

# On "Embarked", filling with "Southampton".
# There is no good reason to fill with "Southampton". But it is just 0.2% part of dataset. The filling effect can be ignored.
train["Embarked"] = train["Embarked"].fillna("S")


# In[ ]:


# Check data condiion again.
kesson_table(train)

# Attribution: # of N/A | %
# Age: 0 | 0.0
# Cabin: 687 | 77.104377
# Embarked: 0 | 0.0


# In[ ]:


# Converting 'object' data type "Sex" and "Embarked" into categorical variables with Integer. 

# On "Sex", 'male' -> 0, 'female' -> 1
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# On "Embarked" 'S' -> 0, 'C'-> 1, 'Q' -> 2 
train["Embarked"][train["Embarked"] == "S" ] = 0
train["Embarked"][train["Embarked"] == "C" ] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# Let's ignore the warnings here　since they are no harm.
# 警告が出力されますが、問題はないのでここでは無視します。


# In[ ]:


# See 'train' data  set.
train.head(10)

# We see "Sex" and "Embarked" are converted into Integer.


# In[ ]:


# Feature selection: remove variables no longer containing relevant information
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_for_correlation = train.drop(drop_elements, axis = 1)

# See 'train_for_correlation' data set.
train_for_correlation.head(10)


# In[ ]:


# Calculating/visualising correlations of attributions.

# Importing libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Calculating
correlations = train_for_correlation.astype(float).corr()
display(correlations)

# Visualising
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(correlations,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


# Survied vs. Pclass : -0.338481 -> Negative correlations.
# Survied vs. Sex    : 0.543351  -> High positive correlations.
# Survied vs. Fare   : 0.257307  -> Positive correlations.

# Age vs. Fare       : -0.339898 -> Negative correlations. -> Children are accompanied with family. Only rich families can travel with children.
# SibSp vs. Parch    : 0.414838 -> Positive correlations. -> Wives are boarding with their husbands. Children are boarding with Parents.

# Fare vs. Pclass    : -0.549500 -> High negativve correlations. -> Good class tickets are expensive. 


# # Reference
# ### Python Pandas
#  - Link: [Python Pandasでのデータ操作の初歩まとめ − 前半：データ作成＆操作編](https://qiita.com/hik0107/items/d991cc44c2d1778bb82e)
# 
# ### Standard deviation
#  - Link: [標準偏差とは何か？その求め方や公式の意味・使い方をわかりやすく説明します](https://atarimae.biz/archives/5379#6070)  
#   It is really kind Japanese explanation about 'Standard deviation' to understand easily. Especially '68%' and '95%' concept helps to grab profile of data set.
#   
# ### Titanic
#  - Link: [データ分析能力を競うサイト「Kaggle」のブラウザからアクセスできる機械学習環境「Kernel」を使ってデータサイエンティスト入門](https://karaage.hatenadiary.jp/entry/2018/04/13/073000)
#  - Link: [Titanic Tutorial](https://www.kaggle.com/karaage0703/titanic-tutorial)  
#  It is the Kaggle's Kernel Notebook writen by the arther who owns the site above "データ分析能力を競うサイト「Kaggle」のブラウザからアクセスできる機械学習環境「Kernel」を使ってデータサイエンティスト入門".
#  - Link: [Introduction to Decision Trees (Titanic dataset)](https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset)  
#  It is really something about introductional explanation on machine learning with 'Titanic'. I used this as an example of a good report this time.
