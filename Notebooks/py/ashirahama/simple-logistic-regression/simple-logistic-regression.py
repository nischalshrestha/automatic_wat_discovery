#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv("../input/train.csv")
train.head()


# In[ ]:


test = pd.read_csv("../input/test.csv")
test.head()


# In[ ]:


# トレーニングデータのProfile Reportを作成
# (出力結果が膨大なのでコメントアウト。必要な時だけ実行)
# pandas_profiling.ProfileReport(train)


# In[ ]:


# テストデータのProfile Reportを作成
# (出力結果が膨大なのでコメントアウト。必要な時だけ実行)
# pandas_profiling.ProfileReport(test)


# # EDA

# In[ ]:


# 全データを一旦結合
# テストも含めた全データで特徴量を見ていく（例：欠損値を補完する際のmedian値など）
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.Survived.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['Survived'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# ## 欠損値

# In[ ]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:50]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})


# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# # Cabinの要素を細切れにする

# In[ ]:


all_data['Cabin'] = all_data['Cabin'].fillna("Undefined")


# In[ ]:


all_data.head()


# In[ ]:


all_data.dtypes


# # Cabinは複数持つ場合があるため分割する

# In[ ]:


def get_cabin_num(cabin):
    if cabin == "Undefined":
        return np.nan
    else:
        cabins = cabin.split(" ")
        count = len(cabins)
        return count
all_data['CabinCount'] = all_data['Cabin'].apply(get_cabin_num)


# In[ ]:


all_data.query("CabinCount > 1")


# In[ ]:


def split_cabin(cabin, num):
    if cabin == "Undefined":
        return "Undefined"
    else:
        cabins = cabin.split(" ")
        if len(cabins) >= num:
            return cabins[num - 1]
        else:
            return "Undefined"
all_data['Cabin1'] = all_data['Cabin'].apply(split_cabin, num=1)
all_data['Cabin2'] = all_data['Cabin'].apply(split_cabin, num=2)
all_data['Cabin3'] = all_data['Cabin'].apply(split_cabin, num=3)
all_data['Cabin4'] = all_data['Cabin'].apply(split_cabin, num=4)


# In[ ]:


all_data.query("CabinCount > 1").head()


# # CabinのTypeで生存率が変わるかチェック

# In[ ]:


def get_cabin_type(cabin):
    return cabin[0]
all_data['CabinType1'] = all_data['Cabin1'].apply(get_cabin_type)
all_data['CabinType2'] = all_data['Cabin2'].apply(get_cabin_type)
all_data['CabinType3'] = all_data['Cabin3'].apply(get_cabin_type)
all_data['CabinType4'] = all_data['Cabin4'].apply(get_cabin_type)


# In[ ]:


# CabinType別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('CabinType1').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('CabinType1',hue='Survived',data=train_ch)


# In[ ]:


# Type別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('CabinType2').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('CabinType2',hue='Survived',data=train_ch)


# In[ ]:


# Type別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('CabinType3').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('CabinType3',hue='Survived',data=train_ch)


# In[ ]:


# Type別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('CabinType4').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('CabinType4',hue='Survived',data=train_ch)


# CabinType1は使えそうだけど、CabinType2～4はいらなそう

# In[ ]:


print(train_ch.query('Survived == 1').size)
print(train_ch.query('Survived != 1').size)


# In[ ]:


def get_cabin_num(cabin):
    if cabin == "Undefined":
        return "Undefined"
    else:
        return "c_" + cabin[1:]
all_data['CabinNum1'] = all_data['Cabin1'].apply(get_cabin_num)
all_data['CabinNum2'] = all_data['Cabin2'].apply(get_cabin_num)
all_data['CabinNum3'] = all_data['Cabin3'].apply(get_cabin_num)
all_data['CabinNum4'] = all_data['Cabin4'].apply(get_cabin_num)


# In[ ]:


# print(all_data['CabinNum1'].unique())
# print(all_data['CabinNum2'].unique())
# print(all_data['CabinNum3'].unique())
# print(all_data['CabinNum4'].unique())


# In[ ]:


# CabinNum1別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('CabinNum1').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))


# In[ ]:


all_data = all_data.drop('Cabin', axis=1)
all_data = all_data.drop('CabinCount', axis=1)
all_data = all_data.drop('Cabin1', axis=1)
all_data = all_data.drop('Cabin2', axis=1)
all_data = all_data.drop('Cabin3', axis=1)
all_data = all_data.drop('Cabin4', axis=1)
# all_data = all_data.drop('CabinType1', axis=1)
all_data = all_data.drop('CabinType2', axis=1)
all_data = all_data.drop('CabinType3', axis=1)
all_data = all_data.drop('CabinType4', axis=1)
all_data = all_data.drop('CabinNum1', axis=1)
all_data = all_data.drop('CabinNum2', axis=1)
all_data = all_data.drop('CabinNum3', axis=1)
all_data = all_data.drop('CabinNum4', axis=1)


# In[ ]:


# AgeがNaNになってるやつの特徴を調べてみる
all_data.query('Age != Age').head()


# In[ ]:


def get_type_from_name(name):
    array_name = name.split(",")
    family_name = array_name[0]
    name_with_type = array_name[1].split(".")
    type = name_with_type[0].replace(" ","")
    return type
all_data['Type'] = all_data['Name'].apply(get_type_from_name)


# In[ ]:


all_data["Age"] = all_data.groupby("Type")["Age"].transform(
    lambda x: x.fillna(x.median()))


# In[ ]:


# EmbarkedがNaNになっているやつを調べてみる
all_data.query('Embarked != Embarked')


# Pclassが1なので、上流階級の人。Pclassが1の人々は恐らく同じようなところから乗ってるはずなので、Pclassが1の人の最頻値で埋めるようにする。

# In[ ]:


all_data["Embarked"] = all_data.groupby("Pclass")["Embarked"].transform(
    lambda x: x.fillna(x.mode()))


# # Fareの額を整理(どうも同一チケットを持っていいる人の金額がやたら高い。同じチケットを持っている人の数でFareを割ると良い感じになる)

# In[ ]:


same_ticket_count_gp = all_data.groupby('Ticket', as_index=False).size().reset_index()
same_ticket_count_gp = same_ticket_count_gp.rename(columns={0: 'TicketCount'})
same_ticket_count_gp.head()
all_data = pd.merge(all_data, same_ticket_count_gp, on='Ticket', how='left')
all_data['Fare'] = all_data['Fare'] / all_data['TicketCount']
all_data.head()


# In[ ]:


all_data = all_data.drop('TicketCount', axis=1)


# In[ ]:


# FareがNaNになっているやつを調べみる
all_data.query('Fare != Fare')


# Pclassが3の人なので、Pclassが3のFareの中央値を求めて埋める。

# In[ ]:


all_data["Fare"] = all_data.groupby("Pclass")["Fare"].transform(
    lambda x: x.fillna(x.median()))


# # Type別の傾向を見てみる

# In[ ]:


all_data['Type'].unique()


# In[ ]:


# Miss,Master,Mr,Mrs,Msくらいしかいらない。あとはOtherにする
all_data['Type'] = all_data['Type'].replace('Mlle', 'Miss')
all_data['Type'] = all_data['Type'].replace('Ms', 'Miss')
all_data['Type'] = all_data['Type'].replace('Mme', 'Mrs')
all_data['Type'] = [val if val in ['Mr', 'Mrs', 'Miss', 'Ms', 'Master'] else 'Others' for val in all_data['Type']]


# In[ ]:


# Type別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('Type').sum()
print(grouped_ch)
sns.countplot('Type',hue='Survived',data=train_ch)


# やっぱり当初の情報通り、Miss、Mrsが多い。ここでの特徴としてはMissとMrsではMissの方が多いということ。想定では既婚女性の方が多いと思ったんだけど恐らく子供も含まれるんだろうな。

# In[ ]:


# Master(赤ちゃん)
print("Pclass:1, Master count: " + str(train_ch.query('Type == "Master" and Pclass == 1').shape[0]))
print("Pclass:1, Master survived count: " + str(train_ch.query('Type == "Master" and Pclass == 1')['Survived'].sum()))
print("Pclass:2, Master count: " + str(train_ch.query('Type == "Master" and Pclass == 2').shape[0]))
print("Pclass:2, Master survived count: " + str(train_ch.query('Type == "Master" and Pclass == 2')['Survived'].sum()))
print("Pclass:3, Master count: " + str(train_ch.query('Type == "Master" and Pclass == 3').shape[0]))
print("Pclass:3, Master survived count: " + str(train_ch.query('Type == "Master" and Pclass == 3')['Survived'].sum()))


# 赤ちゃんは40人いたけど、20人くらいしか生き残らなかった。この辺は等級も関係してそう

# ## 女性は分かるけど男性の既婚、未婚が分からん。家族の数を出してみよう

# In[ ]:


all_data['FamilyNum'] = all_data['SibSp'] + all_data['Parch'] + 1


# In[ ]:


# FamilyNum別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('FamilyNum').sum()
print(grouped_ch)
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('FamilyNum',hue='Survived',data=train_ch)


# In[ ]:


all_data['Is_Alone'] = all_data['FamilyNum'] == 1


# In[ ]:


# Is_Alone別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('Is_Alone').sum()
print(grouped_ch)
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('Is_Alone',hue='Survived',data=train_ch)


# # Ticketは色々種類があるから分解

# In[ ]:


# 記載にぶれのある部分をなくす
# all_data['Ticket'] = all_data['Ticket'].str.replace("/","")
# all_data['Ticket'] = all_data['Ticket'].str.replace(".","")
def get_ticket_sign_1(ticket):
    array_name = ticket.split(" ")
    if len(array_name) > 1:
        sign = array_name[0]
        array_sign = sign.split("/")
        if len(array_sign) > 1:
            return array_sign[0]
        else:
            return array_sign[0]
    else:
        return "Undefined"
    
def get_ticket_sign_2(ticket):
    array_name = ticket.split(" ")
    if len(array_name) > 1:
        sign = array_name[0]
        array_sign = sign.split("/")
        if len(array_sign) > 1:
            return array_sign[1]
        else:
            return "Undefined"
    else:
        return "Undefined"

def get_ticket_no(ticket):
    array_name = ticket.split(" ")
    if len(array_name) > 2:
        return array_name[2]
    elif len(array_name) == 2:
        return array_name[1]
    else:
        return array_name[0]
    
all_data['Ticket_Sign1'] = all_data['Ticket'].apply(get_ticket_sign_1)
all_data['Ticket_Sign2'] = all_data['Ticket'].apply(get_ticket_sign_2)
all_data['Ticket_No'] = all_data['Ticket'].apply(get_ticket_no)


# In[ ]:


all_data.head()


# In[ ]:


print(all_data['Ticket_Sign1'].unique())
print(all_data['Ticket_Sign2'].unique())
print(all_data['Ticket_No'].unique())


# In[ ]:


all_data['Ticket_Sign1'] = all_data['Ticket_Sign1'].replace('STON', 'SOTON')
all_data['Ticket_Sign1'] = all_data['Ticket_Sign1'].str.replace('.', '')
print(all_data['Ticket_Sign1'].unique())


# In[ ]:


# all_data['Ticket_Sign1'] = [val if val in ['Undefined', 'PC', 'CA', 'SOTON', 'SC', 'FCC', 'SW', 'PP', 'A', 'C', 'P', 'SO', 'W', 'WE'] else 'Others' for val in all_data['Ticket_Sign1']]


# In[ ]:


# Ticket_Sign1別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('Ticket_Sign1').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('Ticket_Sign1',hue='Survived',data=train_ch)


# In[ ]:


# Ticket_Sign2別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('Ticket_Sign2').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('Ticket_Sign2',hue='Survived',data=train_ch)


# あんまり差分なさそう

# In[ ]:


# Ticket_No別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('Ticket_No').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
# sns.countplot('Ticket_No',hue='Survived',data=train_ch)


# Ticket Noにあまり差はなさそう

# In[ ]:


all_data = all_data.drop('Ticket_No', axis=1)
all_data = all_data.drop('Ticket_Sign2', axis=1)
all_data = all_data.drop("Ticket", axis=1)


# # データタイプを変換

# In[ ]:


all_data.dtypes


# In[ ]:


all_data['Pclass'] = ['c_' + str(x) for x in all_data.Pclass]


# # Fare, AgeをBinで表示（追加）

# In[ ]:


all_data['FareBin'] = pd.cut(all_data.Fare, 10, labels=False)
all_data['AgeBin'] = pd.cut(all_data.Age, 10, labels=False)


# In[ ]:


# FareBin別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('FareBin').sum()
print(grouped_ch)
sns.countplot('FareBin',hue='Survived',data=train_ch)


# In[ ]:


# AgeBin別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('AgeBin').sum()
print(grouped_ch)
sns.countplot('AgeBin',hue='Survived',data=train_ch)


# In[ ]:


all_data = all_data.drop(['Fare', 'Age'], axis=1)


# # SibSpとParch別の結果生存者数を確認

# In[ ]:


# SibSp別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
sns.countplot('AgeBin',hue='SibSp',data=train_ch)


# In[ ]:


# Parch別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('Parch',hue='Survived',data=train_ch)


# In[ ]:


all_data.head()


# # 特定項目をLabelEncoding

# In[ ]:


# 文字列をラベル化した数値に変換する為のライブラリをインポート
from sklearn.preprocessing import LabelEncoder
# データタイプがobjectの列の値をラベル化した数値に変換
lbl = LabelEncoder()
lbl.fit(list(all_data['CabinType1'].values))
all_data['CabinType1'] = lbl.transform(list(all_data['CabinType1'].values))
lbl.fit(list(all_data['Ticket_Sign1'].values))
all_data['Ticket_Sign1'] = lbl.transform(list(all_data['Ticket_Sign1'].values))


# # 相関が強そうなのを落とす

# In[ ]:


all_data.head()


# In[ ]:


all_data = all_data.drop('Sex', axis=1) # Typeと重複しそう


# # あんま関係なさそうなやつ（PassengerId,Name,Ticket）は消す

# In[ ]:


all_data_PassengerId = all_data['PassengerId']
all_data_Name = all_data['Name']
all_data = all_data.drop('PassengerId', axis=1)
all_data = all_data.drop('Name', axis=1)


# # 欠損値も埋まったので、object型をDummy Encodeする

# In[ ]:


all_data = pd.get_dummies(all_data)


# ## 相関関係のチェック

# In[ ]:


train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
# Find correlations with the target and sort
correlations = train_ch.corr()['Survived'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(30))


# # あんまり関係なさそうなやつと、余計なやつは削除する

# 検証結果を分析した結果、男性をUnSurvivedとする率が高すぎるため、あえてType_Mrを消す

# In[ ]:


all_data = all_data.drop('FamilyNum', axis=1)
all_data = all_data.drop('Type_Mr', axis=1)
# all_data = all_data.drop('Is_Alone', axis=1)
all_data = all_data.drop('Embarked_Q', axis=1)
all_data = all_data.drop('Type_Others', axis=1)


# In[ ]:


all_data.head()


# # 標準化（今回利用するモデルはLinear Regressionで予測するため）

# In[ ]:


all_data.dtypes


# In[ ]:


all_data.head()


# In[ ]:


# all_data['Fare'] = (all_data['Fare'] - all_data['Fare'].mean()) / all_data['Fare'].std()
# all_data['Age'] = (all_data['Age'] - all_data['Age'].mean()) / all_data['Age'].std()
# all_data['CabinNum'] = (all_data['CabinNum'] - all_data['CabinNum'].mean()) / all_data['CabinNum'].std()
# all_data['FamilyNum'] = (all_data['FamilyNum'] - all_data['FamilyNum'].mean()) / all_data['FamilyNum'].std()
# all_data['FareBin'] = (all_data['FareBin'] - all_data['FareBin'].mean()) / all_data['FareBin'].std()
# all_data['AgeBin'] = (all_data['AgeBin'] - all_data['AgeBin'].mean()) / all_data['AgeBin'].std()
# all_data['Cabin'] = (all_data['Cabin'] - all_data['Cabin'].mean()) / all_data['Cabin'].std()
# all_data['Ticket'] = (all_data['Ticket'] - all_data['Ticket'].mean()) / all_data['Ticket'].std()
all_data = (all_data - all_data.mean()) / all_data.std()


# # モデルの作成と予測

# In[ ]:


X_train = all_data.iloc[:train.shape[0],:]
X_train['PassengerId'] = all_data_PassengerId.iloc[:train.shape[0]]
X_test = all_data.iloc[train.shape[0]:,:]
X_test_PassengerId = all_data_PassengerId.iloc[train.shape[0]:]


# In[ ]:


from sklearn.model_selection import KFold, train_test_split
X = X_train
Y = y_train
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.3, random_state=0
)


# In[ ]:


X_train_Passenger_Id = X_train.PassengerId
X_train = X_train.drop('PassengerId', axis=1)
X_val_Passenger_Id = X_val.PassengerId
X_val = X_val.drop('PassengerId', axis=1)


# # SVC

# In[ ]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)


# In[ ]:


y_pred_val_1 = svc.predict(X_val)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_val, y_pred_val_1)
print(cm)

accuracy=accuracy_score(y_pred_val_1,y_val)
print(accuracy)


# # Logistic Regression

# In[ ]:


# モデル用のライブラリインポート
from sklearn.linear_model import LogisticRegression
# Cを1,10,100,1000,10000と変えたがあまり結果に影響はなかった
slr = LogisticRegression(C = 100)

# fit関数で学習開始
slr.fit(X_train,y_train)

# 偏回帰係数(回帰分析において得られる回帰方程式の各説明変数の係数)を出力
# 偏回帰係数はscikit-learnのcoefで取得
print('傾き：{0}'.format(slr.coef_[0]))

# y切片(直線とy軸との交点)を出力
# 余談：x切片もあり、それは直線とx軸との交点を指す
print('y切片: {0}'.format(slr.intercept_))


# In[ ]:


y_pred_val_2 = slr.predict(X_val)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_val, y_pred_val_2)
print(cm)

accuracy=accuracy_score(y_pred_val_2,y_val)
print(accuracy)


# In[ ]:


import copy
X_val_ch = copy.copy(X_val)
X_val_ch['PassengerId'] = X_val_Passenger_Id
X_val_ch['Survived_pred'] = y_pred_val
X_val_ch = pd.merge(X_val_ch, train, on='PassengerId')


# In[ ]:


X_val_ch.to_csv('validation_result.csv', index=False)


# In[ ]:


X_test.head()


# In[ ]:


X_train.head()


# In[ ]:


y_test_pred = slr.predict(X_test)


# In[ ]:


y_test_pred_2 = svc.predict(X_test)


# # 結果CSVの作成

# In[ ]:


submission = pd.DataFrame({
    "PassengerId": X_test_PassengerId,
    "Survived": y_test_pred_2
})
submission.to_csv('submission.csv', index=False)


# In[ ]:


X_train.head()

