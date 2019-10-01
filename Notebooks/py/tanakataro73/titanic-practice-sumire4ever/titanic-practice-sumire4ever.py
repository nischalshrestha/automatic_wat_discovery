#!/usr/bin/env python
# coding: utf-8

# # https://qiita.com/sumire4ever/items/8f55a11e826c1454611d の写経

# # モデル1(特徴量追加なし、randomforest)詳細

# In[ ]:


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


# >  Ageに値があるデータ、ないデータに分割して
# + Ageに値があるtrainデータで学習し、Ageに値があるtestデータを(特徴量にAgeを含めて)予測
# + Ageに値がないtrainデータで学習し、Ageに値がないtestデータを(特徴量にAgeを含めず)予測  

# In[ ]:


import csv as csv
import math
from numpy import *
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier ,GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model


# + Ageに値が入っているものはAgeも使って予測し、そうでないものはAgeを抜いて予測する
# + Embarkedを説明変数に加える。ただし、nullになっているレコードはdropする
# + Ageに値が入っているもの... Pclass,SibSp,Parch,Age,Gender,EmbarkedIntからSurvivedを予測する
# + Ageに値が入っていないもの... Pclass,SibSp,Parch,Gender,EmbarkedIntからSurvivedを予測する    

# In[ ]:


#header=0は？
train_df = pd.read_csv("../input/train.csv", header=0)
train_df = train_df[train_df["Embarked"].notnull()]


# In[ ]:


train_df


# In[ ]:


train_df["Gender"] = train_df["Sex"].map({"female": 0, "male": 1}).astype(int)
train_df["EmbarkedInt"] = train_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)
age1_df = train_df[train_df.Age >= 0]   #Ageあり
age0_df = train_df[train_df["Age"].isnull()]    #Ageなし


# In[ ]:


age1_features = age1_df[["Pclass","Age","SibSp","Parch","Gender","EmbarkedInt"]]         #特徴量のデータ 
age1_labels   = age1_df["Survived"]          #特徴量に対する正解データ
age0_features = age0_df[["Pclass","SibSp","Parch","Gender","EmbarkedInt"]]         #特徴量のデータ 
age0_labels   = age0_df["Survived"]          #特徴量に対する正解データ


# In[ ]:


# Load test data, Convert "Sex" to be a dummy variable
test_df = pd.read_csv("../input/test.csv", header=0)
test_df = test_df[test_df["Embarked"].notnull()]
test_df["Gender"] = test_df["Sex"].map({"female": 0, "male": 1}).astype(int)
test_df["EmbarkedInt"] = test_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)
age1_t_df = test_df[test_df.Age >= 0]   #Ageあり
age0_t_df = test_df[test_df["Age"].isnull()]    #Ageなし


# In[ ]:


# Copy test data's "PassengerId" column, and remove un-used columns
ids_age1 = age1_t_df["PassengerId"].values
ids_age0 = age0_t_df["PassengerId"].values


# In[ ]:


#train-data
age1_df = age1_df.drop(["Name", "Ticket", "Sex", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
age0_df = age0_df.drop(["Name", "Ticket", "Age", "Sex", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)


# In[ ]:


#必要なもののみ残してある
age1_df


# In[ ]:


#test-data
age1_t_df = age1_t_df.drop(["Name", "Ticket", "Sex", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
age0_t_df = age0_t_df.drop(["Name", "Ticket", "Age", "Sex", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)


# In[ ]:


train_data_age1 = age1_df.values
test_data_age1 = age1_t_df.values


# In[ ]:


train_data_age1


# In[ ]:


#5つぐらいのモデルを適当に試して一番CVスコアが良いモデルで予測
#age1は最もスコアの良かったGradientBoostingClassifierを使って予測
model_age1 = GradientBoostingClassifier(n_estimators=100)
#1列目以降が特徴量、0列目がターゲット変数
output_age1 = model_age1.fit(train_data_age1[0::, 1::], train_data_age1[0::, 0]).predict(test_data_age1).astype(int)


# In[ ]:


#age0は最もスコアの良かったAdaBoostClassifierを使って予測
train_data_age0 = age0_df.values
test_data_age0 = age0_t_df.values
model_age0 = AdaBoostClassifier(n_estimators=50)
output_age0 = model_age0.fit(train_data_age0[0::, 1::], train_data_age0[0::, 0]).predict(test_data_age0).astype(int)


# In[ ]:


# export result to be "titanic_submit.csv"
submit_file = open("titanic_submit_r5.csv", "w", newline="")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids_age1, output_age1))
file_object.writerows(zip(ids_age0, output_age0))
submit_file.close()


# # モデル2(ルールベース)詳細

# In[ ]:


train = train[train["Embarked"].notnull()]
test['Fare'].fillna(test['Fare'].median(), inplace = True)


# In[ ]:


#辞書ではなくリストで集計方法を渡すと、すべてのカラムの集計値が出る
print(train.groupby(["Pclass"]).agg(["count","mean"]))


# In[ ]:


#survived列に対してのみ集計値を取得
print(train.groupby(["Pclass"]).agg(["count","mean"])["Survived"])
print(train.groupby(["Sex"]).agg(["count","mean"])["Survived"])


# > 全部は載せませんが、特徴量の中ではSexとPclassが比較的生存と相関が強そう。
# > ということで、この2属性で再度groupbyを行い、値の組み合わせごとに生存率を確認。

# In[ ]:


#すべての列に対してcountされる
print(train.groupby(["Sex","Pclass"],as_index=False).count())


# In[ ]:


# (全行、sex,pclass,survived列)を取得
print(train.groupby(["Sex","Pclass"],as_index=False).count().loc[:, ["Sex","Pclass","Survived"]])


# In[ ]:


#Survivedをcountに変更、ならPassengerIdをcountにしても同じことではある
print(train.groupby(["Sex","Pclass"],as_index=False).count().loc[:, ["Sex","Pclass","Survived"]].rename(columns={"Survived":"count"}))


# In[ ]:


#groupbyにリストを渡すと組み合わせて分類してくれる
count = train.groupby(["Sex","Pclass"],as_index=False).count().loc[:, ["Sex","Pclass","Survived"]].rename(columns={"Survived":"count"})
#meanで生きてた人の割合になる
mean = train.groupby(["Sex","Pclass"],as_index=False).mean().loc[:, ["Sex","Pclass","Survived"]].rename(columns={"Survived":"ratio"})
rule = pd.merge(count, mean, on=["Sex","Pclass"])
print(rule)


# > Pclass = 1 or 2の女性の生存率と、Pclass = 2 or 3の男性の死亡率が極めて高いことがわかります。
# > 今回はAccuracy 0.8を目安としているので、(trainとtestのデータの傾向が同じであれば)testデータに対して  
# >   
# > + Pclass = 1 or 2の女性 => 生存  
# > + Pclass = 2 or 3の男性 => 死亡  
# >  
# > と一律で予測しても、少なくともこの条件を満たす人についてはAccuracy 0.8を超えられそうです。
# 
# 生存率＝0.96と生存率＝0.13だから

# > それ以外(Pclass = 3の女性、Pclass = 1の男性)については複雑な条件で生死が分かれていそうなので、この条件に該当するtestデータのみrandomforestで予測していきます。

# In[ ]:


passenger_id = list()
survived = list()


# In[ ]:


# 女性でpclassが1か2       →生存 
# 男性でpclassが2か３の人→死亡
for i in range(len(test)):
    data = test.iloc[i, :]
    if data["Sex"]=="female" and data["Pclass"]<=2:
        passenger_id.append(data["PassengerId"])
        survived.append(1)
    elif data["Sex"]=="male" and data["Pclass"]>=2:
        passenger_id.append(data["PassengerId"])
        survived.append(0)


# In[ ]:


#列方向で結合
output_df = pd.concat([pd.Series(passenger_id),pd.Series(survived)],axis=1)
output_df.columns = ["PassengerId", "Survived"]


# > 上記データに対するrandomforestでの予測ですが、新たに下記のような特徴量を生成しています。
# > - FamilySize: 同乗している家族人数(SibSpとParchの値に本人分の1を足したもの)
# > - IsAlone: 上記FamilySizeが1か否かのフラグ
# > - Title_Code: 敬称(Mr., Mrs.等)のコード
# > - Cabin_Flag: Cabinカラムに値が入っているか否かのフラグ
# > - FareBin_Code: FareをBinで適当に分割(今回は4分割)した後の各FareのBin位置
# > - Sex_Code: 性別のコード
# > - Embarked_Code: Embarkedのコード
# > - AgeBin_Code: AgeをBinで適当に分割(今回は4分割)した後の各AgeのBin位置(※Ageに値が入っているデータのみ)

# In[ ]:


#学習データとテストデータを先に結合しておくことで同時にcleaningが可能
data_cleaner = [train, test]
print(data_cleaner)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()


# In[ ]:


for dataset in data_cleaner:
    #特徴量:同乗家族人数(自身も含む)を生成し追加
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1    
    #特徴量:単独乗船フラグを生成し追加
    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
    #特徴量:敬称を生成し追加
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    #下記Cabin_Flagを追加
    #Cabinに値が入っているなら1,そうでないなら0
    dataset["Cabin_Flag"] = dataset["Cabin"].notnull().replace({True:1, False:0})

    #特徴量:Fareのbinをqcutで指定し追加(qcut:境界を自動的に設定し分類)
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

    dataset["Sex_Code"] = label.fit_transform(dataset['Sex'])
    dataset["Embarked_Code"] = label.fit_transform(dataset['Embarked'])


# In[ ]:


dataset['FareBin_Code']


# In[ ]:


#4分位数の境界値が入る
dataset['FareBin']


# In[ ]:


#特殊な敬称をクリーニング
stat_min = 10
title_names_train = (train['Title'].value_counts() < stat_min)


# In[ ]:


title_names_train


# In[ ]:


title_names_test = (test['Title'].value_counts() < stat_min)
#敬称の使用数が9以下のものはMiscと分類
train['Title'] = train['Title'].apply(lambda x: 'Misc' if title_names_train.loc[x] == True else x)
test['Title'] = test['Title'].apply(lambda x: 'Misc' if title_names_test.loc[x] == True else x)


# In[ ]:


train_age1 = train[train["Age"].notnull()]
test_age1 = test[test["Age"].notnull()]
data_cleaner_age1 = [train_age1, test_age1]


# In[ ]:


data_cleaner_age1


# In[ ]:


for dataset in data_cleaner_age1:
    #量ではなく値で分類。qcutは全体に対する量で等分割するが、cutは絶対値の等分割となる
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])


# ### 結合は http://pppurple.hatenablog.com/entry/2016/06/27/022310 がわかりやすい

# In[ ]:


train_age1 = train_age1.loc[:, ["PassengerId", "AgeBin_Code"]]
test_age1 = test_age1.loc[:, ["PassengerId", "AgeBin_Code"]]
#PassengerIdをマージ列名として、左側のテーブルに存在する（年齢がnullでない者）のみ、AgeBin_Code列をマージ（左外部結合）
train = pd.merge(train, train_age1, on="PassengerId", how="left")
test = pd.merge(test, test_age1, on="PassengerId", how="left")


# # データ整形完了、GridSearchして最適なパラメータで予測。

# In[ ]:


#分類がめんどくさい女性かつpclass3,男性かつpclass1のデータのみ取得
train_rf = train[((train["Sex"]=="female")&(train["Pclass"]==3))|((train["Sex"]=="male")&(train["Pclass"]==1))]
test_rf = test[((test["Sex"]=="female")&(test["Pclass"]==3))|((test["Sex"]=="male")&(test["Pclass"]==1))]
#print(test_rf.groupby(["Sex","Pclass"],as_index=False).mean())
train_age1_rf = train_rf[train_rf["Age"].notnull()] #Ageあり
train_age0_rf = train_rf[train_rf["Age"].isnull()]  #Ageなし
test_age1_rf = test_rf[test_rf["Age"].notnull()]    #Ageあり
test_age0_rf = test_rf[test_rf["Age"].isnull()] #Ageなし
ids_age1 = list(test_age1_rf["PassengerId"])
ids_age0 = list(test_age0_rf["PassengerId"])


# In[ ]:


#train-dataをnumpy arrayとして用意
train_data_age1 = train_age1_rf.loc[:, ['Survived', 'Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Title_Code', 
                                      'Cabin_Flag', 'FareBin_Code', 'Sex_Code', 'Embarked_Code', 'AgeBin_Code']].values


# In[ ]:


#test-data
test_data_age1 = test_age1_rf.loc[:, ['Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Title_Code', 
                                      'Cabin_Flag', 'FareBin_Code', 'Sex_Code', 'Embarked_Code', 'AgeBin_Code']].values


# In[ ]:


train_data_age0 = train_age0_rf.loc[:, ['Survived', 'Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Title_Code', 
                                      'Cabin_Flag', 'FareBin_Code', 'Sex_Code', 'Embarked_Code']].values


# In[ ]:


test_data_age0 = test_age0_rf.loc[:, ['Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Title_Code', 
                                      'Cabin_Flag', 'FareBin_Code', 'Sex_Code', 'Embarked_Code']].values


# In[ ]:


xs_1 = train_data_age1[0::, 1::]
y_1 = train_data_age1[0::, 0]

xs_0 = train_data_age0[0::, 1::]
y_0 = train_data_age0[0::, 0]


# In[ ]:


xs_test1 = test_data_age1
xs_test0 = test_data_age0


# In[ ]:


parameters = {'max_depth': [2,4,6,8,10], 'n_estimators': [50,100,200]}


# In[ ]:


from xgboost import XGBClassifier
from sklearn import grid_search
clf = grid_search.GridSearchCV(XGBClassifier(), parameters)
clf.fit(xs_1, y_1)


# In[ ]:





# In[ ]:


print(clf.best_score_)


# In[ ]:


clf.best_params_


# In[ ]:


#引数で渡す時の**がポイントです。このようにすることで、辞書形式で関数に引数を渡すことができます。
clf_final = XGBClassifier(**clf.best_params_)
clf_final.fit(xs_1, y_1)
Y_pred1 = clf.predict(xs_test1).astype(int)


# In[ ]:


clf = grid_search.GridSearchCV(XGBClassifier(), parameters)
clf.fit(xs_0, y_0)


# In[ ]:


print(clf.best_score_)


# In[ ]:


clf_final = XGBClassifier(**clf.best_params_)
clf_final.fit(xs_0, y_0)
Y_pred0 = clf.predict(xs_test0).astype(int)


# In[ ]:


ids = pd.Series(ids_age1 + ids_age0)
pred = pd.Series(list(Y_pred1)+list(Y_pred0))
output_df2 = pd.concat([pd.Series(ids),pd.Series(pred)],axis=1)
output_df2.columns = ["PassengerId", "Survived"]


# In[ ]:


#output_df は面倒くさくない人たちのレコード（女性でpclass1,2あるいは男性でpclass3の人）
print(pd.concat([output_df,output_df2],axis=0))


# In[ ]:


print(pd.concat([output_df,output_df2],axis=0).sort_values(by="PassengerId"))


# In[ ]:


#drop=tureでないと旧インデックスがデータ列に残存してしまう
print(pd.concat([output_df,output_df2],axis=0).sort_values(by="PassengerId").reset_index(drop=True))


# In[ ]:


final_output = pd.concat([output_df,output_df2],axis=0).sort_values(by="PassengerId").reset_index(drop=True)


# In[ ]:


final_output.to_csv("predict_hybrid_r1.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:












































