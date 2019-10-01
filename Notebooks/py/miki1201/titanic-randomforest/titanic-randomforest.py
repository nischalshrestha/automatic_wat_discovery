#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Data Exploration

# In[8]:


train= pd.read_csv("../input/train.csv")
test= pd.read_csv("../input/test.csv")
train.head(3)


# It seems like Sex & Embarked are categorical value so transform them into dummy variables.

# In[10]:


train= train.replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
test= test.replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)


# Missing values

# In[ ]:


df_train.isnull().sum()


# Cabin has too many missing values and Age also has some null values so once removing those missing values to see the correlation between survived and other parameters.

# In[11]:


train["Age"].fillna(train.Age.mean(), inplace=True) 
train["Embarked"].fillna(train.Embarked.mean(), inplace=True)


# ここまでは比較的に簡単な処理を行ってきました。今処理できていないのはName、TicketとCabinでしょう。それぞれNameは全くみんなバラバラでどう処理の仕方がわからない。Ticketも同様。Cabinは欠損値が多すぎる。これらの骨が折れる奴らを処理しなければなりません。私には全く思いつきません。では、どうするのか？Kaggleにはカーネルがありますので参考にしながら進めて行きましょう。ここまでも参考にしてきましたが… はじめにNameを処理して行きます。Nameを眺めてなんとなく共通しているものに気がつくでしょうか？Mr、Mrs、Missなどの敬称があると思います。これで分類してみましょう。Salutationは挨拶という意味らしいです。そして今は女性に対して使うMissとMrsがMsに統合されているそうです。

# In[12]:


combine1 = [train]

for train in combine1: 
        train['Salutation'] = train.Name.str.extract(' ([A-Za-z]+).', expand=False) 
for train in combine1: 
        train['Salutation'] = train['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        train['Salutation'] = train['Salutation'].replace('Mlle', 'Miss')
        train['Salutation'] = train['Salutation'].replace('Ms', 'Miss')
        train['Salutation'] = train['Salutation'].replace('Mme', 'Mrs')
        del train['Name']
Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 
for train in combine1: 
        train['Salutation'] = train['Salutation'].map(Salutation_mapping) 
        train['Salutation'] = train['Salutation'].fillna(0)


# 
# 次に、Ticketの処理です。今度はTicketの先頭の文字で分けていきます。また、文字列の長さでも分けていきます。そして、その後にそれらの文字を数字に直します。

# In[13]:


for train in combine1: 
        train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
        train['Ticket_Lett'] = train['Ticket_Lett'].apply(lambda x: str(x)) 
        train['Ticket_Lett'] = np.where((train['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), train['Ticket_Lett'], np.where((train['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 
        train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x)) 
        del train['Ticket'] 
train['Ticket_Lett']=train['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3)


# In[14]:


#同様にCabinも先頭の文字で分類
for train in combine1: 
    train['Cabin_Lett'] = train['Cabin'].apply(lambda x: str(x)[0]) 
    train['Cabin_Lett'] = train['Cabin_Lett'].apply(lambda x: str(x)) 
    train['Cabin_Lett'] = np.where((train['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),train['Cabin_Lett'], np.where((train['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))
del train['Cabin'] 
train['Cabin_Lett']=train['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1)


# ここで一度trainを見てみましょう。 一応全て数字になりました。ここからは、さらに生存予測の精度を高めるために新たな変数を追加していきます。 FamilySizeとIsAloneです。なぜなら一緒に乗船している人数によって生存に大きく差が出るからです。ここまででまだ使われていないものはPclass、SibspとParchです。Pclassは何等級のところに乗っていたかを表すものなのでこのままでいいです。Sibspは乗っていた夫婦と兄弟の人数を表したものです。Parchは乗っていた親と子供の人数を表したものです。よってSibsp+Parch+1がFamilySizeとなります。また、FamilySizeが1だとIsAlone一人で乗っているかどうかが1となります。

# In[15]:


train.head(10)


# In[16]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
for train in combine1:
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1


# ここまででtrainの処理は終わりました。次に、trainのデータを機械学習にかけるために加工します。はじめにtrainの値だけを取り出し、次にそれを正解データと学習用のデータに分けます。

# In[ ]:





# In[17]:


train_data = train.values
xs = train_data[:, 2:] # Pclass以降の変数
y  = train_data[:, 1]  # 正解データ


# 次に、testの処理をしていきたいと思います。ほぼtrainと一緒のことをします。ですが気をつけなければいけないことがあります。testのデータをみてみましょう。

# In[18]:


test.info()


# わかりますでしょうか？ 姑息なことにFareが一つ欠損しております。これを埋めなかったが為に何度エラーが起きたことでしょう。気をつけてください。こういうところでもチュートリアルなのかもしれません。

# In[19]:


test["Age"].fillna(train.Age.mean(), inplace=True)
test["Fare"].fillna(train.Fare.mean(), inplace=True)

combine = [test]
for test in combine:
    test['Salutation'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for test in combine:
    test['Salutation'] = test['Salutation'].replace(['Lady', 'Countess','Capt', 'Col',         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    test['Salutation'] = test['Salutation'].replace('Mlle', 'Miss')
    test['Salutation'] = test['Salutation'].replace('Ms', 'Miss')
    test['Salutation'] = test['Salutation'].replace('Mme', 'Mrs')
    del test['Name']
Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for test in combine:
    test['Salutation'] = test['Salutation'].map(Salutation_mapping)
    test['Salutation'] = test['Salutation'].fillna(0)

for test in combine:
        test['Ticket_Lett'] = test['Ticket'].apply(lambda x: str(x)[0])
        test['Ticket_Lett'] = test['Ticket_Lett'].apply(lambda x: str(x))
        test['Ticket_Lett'] = np.where((test['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), test['Ticket_Lett'],
                                   np.where((test['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            '0', '0'))
        test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))
        del test['Ticket']
test['Ticket_Lett']=test['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3) 

for test in combine:
        test['Cabin_Lett'] = test['Cabin'].apply(lambda x: str(x)[0])
        test['Cabin_Lett'] = test['Cabin_Lett'].apply(lambda x: str(x))
        test['Cabin_Lett'] = np.where((test['Cabin_Lett']).isin(['T', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']),test['Cabin_Lett'],
                                   np.where((test['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            '0','0'))        
        del test['Cabin']
test['Cabin_Lett']=test['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1).replace("G",1) 

test["FamilySize"] = train["SibSp"] + train["Parch"] + 1

for test in combine:
    test['IsAlone'] = 0
    test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1
    
test_data = test.values
xs_test = test_data[:, 1:]


# さて、ここまでで面倒だった前処理が終わりました。もちろんもっといい処理の方法があったと思いますが、初心者ではこんなもんです。ここからがメインの学習タイムです。様々な学習方法を試した結果RandomForestClassifierを使うのが一番結果が私的には良かったです。ではやってみましょう。

# In[20]:


'''
from sklearn.ensemble import RandomForestClassifier

random_forest=RandomForestClassifier()
random_forest.fit(xs, y)
Y_pred = random_forest.predict(xs_test)

import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), Y_pred.astype(int)):
        writer.writerow([pid, survived])
'''


# 「あれ？毎回結果変わってないか？」そうです。RandomForestClassifierには自由に変えられるパラメータがたくさんあります。これをいじっていくことでどんどん結果が変わります。最初の壁は0.8を超えられるかです。 その後は1つ結果を更新するのもしんどくなっていきます。私は、0.82297で止まりました。これ以上はもっとデータ処理を変えるか、もっといいパラメータを見つけるかしかないと思います。パラメータの探索方法をお教えしましょう。

# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
'''

parameters = {
        'n_estimators'      : [10,25,50,75,100],
        'random_state'      : [0],
        'n_jobs'            : [4],
        'min_samples_split' : [5,10, 15, 20,25, 30],
        'max_depth'         : [5, 10, 15,20,25,30]
}

clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
clf.fit(xs, y)
 
print(clf.best_estimator_)
'''


# 今回いじるのはn_estimators、min_samples_splitとmax_depthです。ちなみに上のコードを走らせると時間がかなりかかります。random_stateは初期状態を固定するためのものなのでとりあえず0にしました。n_jobsは計算に使うCPUの数なので適切のものにしてください。 パラメータをいじりながら最適なものを探し、それを提出し、結果を確認する。これを何度も繰り返していいものを見つけてください。今現在私の最高を出したパラメータを下に示します。ただ、これが再現性があるのかどうかは知りません。

# In[38]:


random_forest=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=4,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
random_forest.fit(xs, y)
Y_pred = random_forest.predict(xs_test)

import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), Y_pred.astype(int)):
        writer.writerow([pid, survived])


# データ分析
# ここからはデータの分析をして行きたいと思います。タイタニックでは、約1500人が犠牲となり、生存者は約700人であるそうです。なぜこれほどまで死者が出たのか？その原因の一つには救命ボートが船に乗っていた人の半分を載せられるぐらいしかなかったことでしょう。
# それでは実際にデータをみていきましょう。はじめに、性別によってどれほど生存に違いが出たかをみてみましょう。下の図をみてください。0が男性、1が女性です。明らかに女性の生存率が高いですね。そして男性の人数がかなり多いですね。これは推測ですが、乗組員に男性が多かったからではないでしょうか。

# In[25]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
g = sns.factorplot(x="Sex", y="Survived",  data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[26]:


sns.countplot(x='Sex', data = train)


# 
# 次に乗っていた等級による生存率をみてみましょう。明らかに等級がいい順に生存率が高いです。ただ、ここで気がついたことがあります。今回のデータは全てこのPclassの分類があるため乗組員はデータに含まれていないのではないでしょうか。先ほどの推測は間違いでした。

# In[27]:


g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# 次は等級と性別を複合した時の生存率をみてみましょう。1等と2等の女性の生存率が高く、2等と3等の男性の生存率が低いです。

# In[28]:


g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# 
# 次に敬称による生存率をみてみましょう。ここで0と5は珍しいもので数が少ないので無視します。注目するべきは他のものです。1のMrは大人の男性に使われるものです。だから生存率が少ないのですね。次に、2と3はMissとMrsなので女性に使われるものなので生存率が高いです。4はMasterです。誰に使われるものかと言いますと、青年や若い男性です。 以上からこの敬称は結局性別の言い換えに近いものなのではないでしょうか？一つ違うのは男性を年齢で分けているMasterがあるところではないでしょうか。

# In[31]:


g = sns.factorplot(x="Salutation", y="Survived",  data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# ここでデータの生存率に対する相関をみます。先ほどみてきた通り性別(Sex,Salutation)による相関、属している社会階級つまりお金をどれだけ持っているか(Pclass,Fare)に対する相関が高いです。 Cabin_LettとTicket_Lettの相関も高いです、これも社会的地位ではないでしょうか？高いFareであればいいTicketを取れ、乗るCabinの生存率も上がるはずです。他にはIsAloneも相関が高いです。次からは一緒に乗船した人数による生存率をみてみましょう。

# In[32]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
del train['PassengerId']
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# はじめに、何人ぐらいで乗っていたのか、また、どのくらい生存したのかをみていきましょう。圧倒的に一人で乗っていた人が多かったです。そして、1人か5人以上で乗っていると生存率が悪かったこともわかります。

# In[33]:


sns.countplot(x='FamilySize', data = train, hue = 'Survived')


# ではなぜ大家族か一人で乗ると死亡率が高かったのか？大家族であると当然お金もたくさんかかるので3等に乗った人が多かったようです。一人の方々も3等が多かった。家族が多いとそもそも非難するのも難しいし、3等なので救出の優先度も低かったのでしょう。

# In[34]:


sns.countplot(x='FamilySize', data = train,hue = 'Pclass')


# 次に、乗船場所による生存率の違いを見てみます。タイタニックの航路はイギリスのサウサンプトン→フランスのシェルブール→アイルランドのクイーンズタウンの順番でした。 下の図を見るとシェルブールから乗った人の生存率が高かったことがわかります。その理由は一つ下の図を見るとわかります。1等に乗った人の割合が高かったからでしょう。しかし、そのように考えるとクイーンズタウンから乗った人は3等ばかりなのに生存率が少し高いです。その理由のうち少しはさらに下の図の男女比だと思いますが、正確な理由はわかりません。

# In[36]:


t=pd.read_csv("../input/train.csv").replace("S",0).replace("C",1).replace("Q",2)
train['Embarked']= t['Embarked']
g = sns.factorplot(x="Embarked", y="Survived",  data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[37]:


sns.countplot(x='Embarked', data = train,hue = 'Pclass')


# In[39]:


sns.countplot(x='Embarked', data = train,hue = 'Sex')


# 最後は年齢による違いをみて行きたいと思います。下の図をみてみると10代後半から30代ほどまでは死亡率が高く、子供の死亡率は低いです。どうやらこのころは15歳より上だとほとんど成人とみなされていたようです。また、老人の死亡率も高いです。

# In[40]:


plt.figure()
sns.FacetGrid(data=t, hue="Survived", aspect=4).map(sns.kdeplot, "Age", shade=True)
plt.ylabel('Passenger Density')
plt.title('KDE of Age against Survival')
plt.legend()


# In[41]:


for t in combine1: 
    t.loc[ t['Age'] <= 15, 'Age']                                                = 0
    t.loc[(t['Age'] > 15) & (t['Age'] <= 25), 'Age'] = 1
    t.loc[(t['Age'] > 25) & (t['Age'] <= 48), 'Age'] = 2
    t.loc[(t['Age'] > 48) & (t['Age'] <= 64), 'Age'] = 3
    t.loc[ t['Age'] > 64, 'Age'] =4
g = sns.factorplot(x="Age", y="Survived",  data=t,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[42]:


sns.countplot(x='Age', data = t,hue = 'Sex')


# In[43]:


sns.countplot(x='Age', data = t,hue = 'Survived')


# 5.まとめ
# 以上からどのような人が生き残りやすいのかというと15歳以下の子供あるいは女性であればかなり助かります。また、それで1等に乗っていて3人か4人家族だとほぼ生き残れるでしょう。逆に、私のような男でお金のない学生で一人旅が好きな奴は死にます。私は乗らなくてよかったです。タイタニックのおかげで救命ボートがたくさん積まれるようになり、無線も常備されたりなどして今の航海の安全に繋がっているそうです。

# In[ ]:




