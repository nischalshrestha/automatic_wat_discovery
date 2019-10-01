#!/usr/bin/env python
# coding: utf-8

# In[144]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#import pydotplus
from scipy import stats
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from graphviz import Digraph

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[145]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train["istrain"] = 1
target = train["Survived"]
train.drop("Survived", axis=1,inplace=True)
test["istrain"] = 0
df = pd.concat([train, test], axis=0)


# # EDA & 前処理

# ## overview & fillna

# In[146]:


df.head()


# In[147]:


df.info()


# In[148]:


# 欠損値の総数を見る
df.isnull().sum(axis=0)


# In[149]:


#df.drop(["Cabin", "Name", "Ticket"], axis=1, inplace=True)


# In[150]:


df["Embarked"].value_counts().plot(kind="pie")
df["Embarked"].fillna("S", inplace=True)


# In[151]:


sns.kdeplot(df["Fare"])
plt.axvline(df.Fare.mean(), c="r")
plt.axvline(df.Fare.median(), c="y")
df["Fare"].fillna(df.Fare.median(), inplace=True)


# In[152]:


plt.figure(figsize=(6, 3))
#sns.kdeplot(df["Age"])
df["Age"].plot(kind="hist", bins=np.arange(0, 80, 5))
plt.axvline(df.Fare.mean(), c="r")
plt.axvline(df.Fare.median(), c="y")
#df["Age"].fillna(df.Fare.median(), inplace=True)


# ## add feautures

# ### Cabin

# In[153]:


tmp_train = pd.concat([df.query("istrain == 1"), target], axis=1)
tmp_train["cabin_found"] = ~tmp_train.Cabin.isnull()
tmp_train.groupby("cabin_found")["Survived"].mean()


# In[154]:


tmp_train["Cabin"].fillna("U", inplace=True)
tmp_train["Cabin"].replace("T", "U", inplace=True)
tmp_train["Cabin_I"] = tmp_train["Cabin"].apply(lambda x: x[0])
tmp_train.groupby("Cabin_I")["Survived"].mean()


# In[155]:


tmp_train["Cabin_I"].value_counts()


# In[156]:


x = np.linspace(0, 1, 1000)
for c in sorted(tmp_train["Cabin_I"].unique()):
    tmp = tmp_train.query("Cabin_I == '{}'".format(c))
    N = tmp.shape[0]
    a = (tmp["Survived"] == 1).sum()
    y = stats.beta(a+0.01, N-a+0.01).pdf(x)
    plt.plot(x, y, label=c)
plt.legend()


# ### Ticket

# In[157]:


tmp_train["Ticket_sp"] = tmp_train["Ticket"].apply(lambda x: len(x.split()))
tmp_train["Ticket_num"] = tmp_train["Ticket"].apply(lambda x: len(x.split()[-1]))
tmp_train.groupby(["Ticket_num", "Pclass"]).count()["PassengerId"]


# In[158]:


idx = tmp_train["Ticket"].apply(lambda x: len(x.split())) == 1
idx2 = tmp_train[idx]["Ticket"].apply(lambda x: x.isdigit())
tmp_train[idx]["Ticket"][~idx2]


# In[159]:


# さすがに使えそうにないので削除
#df.drop("Ticket", inplace=True, axis=1)


# ### Name

# In[160]:


from collections import defaultdict
ddic = defaultdict(int)
for name in df.Name:
    for n in name.replace(",", "").split():
        ddic[n] += 1


# In[161]:


for i in pd.Series(ddic).sort_values(ascending=False).index:
    if i[-1] == ".":
        print(i, end=": ")
        print(df.Name.str.contains(i.replace(".", "\.")).sum())


# In[162]:


tmp_train[tmp_train["Name"].str.contains("Mrs\.")]["Survived"].mean()


# In[163]:


df.head()


# In[164]:


df["Cabin_I"] = df["Cabin"].fillna("U").replace("T", "U").apply(lambda x: x[0])
df["Alone"] = (df.Parch + df.SibSp == 0).astype(np.uint8)
df["Family_size"] = (df.Parch + df.SibSp + 1).astype(np.uint8)

for title in ["Mr.", "Miss.", "Master.", "Mrs."]:
    df["is_" + title[:-1]] = df.Name.str.contains(title.replace(".", "\.")).astype(np.uint8)
df = pd.concat([df, pd.get_dummies(df[["Sex", "Embarked", "Cabin_I"]], drop_first=True)], axis=1)
df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked", "Cabin_I"], axis=1, inplace=True)


# ### ischild (予測)

# In[165]:


idx  = df.Age.isna()
c_data = df[~idx]
c_pass = c_data["PassengerId"]


# In[166]:


D = c_data.Age.astype(np.uint8).value_counts()
x = D.keys()
y = D.values
plt.bar(x, y)
plt.axvline(15, c="r", alpha=0.5)


# In[167]:


c_target = (c_data.Age < 16).astype(np.uint8)
c_data.drop(["PassengerId", "istrain", "Age"], axis=1, inplace=True)


# In[168]:


c_target.value_counts()


# In[169]:


rfc = RandomForestClassifier()
parameters = {"criterion": ["gini", "entropy"],
              "max_depth": range(1, 8),
              "n_estimators": [3, 5, 10, 20, 40, 80, 160, 320]}
clf = GridSearchCV(estimator=rfc,
                   param_grid=parameters,
                   cv=10,
                   n_jobs=-1,
                   verbose=5,
                   scoring="f1")


# In[170]:


clf.fit(c_data, c_target)


# In[171]:


fig, axs = plt.subplots(1, 2, figsize=(12, 4))
res = pd.DataFrame(clf.cv_results_)

for i, criterion in enumerate(res["param_criterion"].unique()):
    for depth in sorted(res["param_max_depth"].unique()):
        res_c = res.query("param_criterion == '{}' & param_max_depth == {}"
                              .format(criterion, depth))
        x = res_c["param_n_estimators"].astype(np.float32)
        y_mean = res_c["mean_test_score"]
        y_std = res_c["std_test_score"]
        axs[i].errorbar(np.log(x), y_mean, yerr=y_std, label=depth)
    axs[i].set_xlabel("log(estimators)")
    axs[i].set_ylabel("f1 score")
    axs[i].set_title("criterion = {}".format(criterion))
    axs[i].legend()
plt.savefig("c_random_forest.png")


# In[172]:


c_rfc = RandomForestClassifier(max_depth=5,
                               n_estimators=20,
                               criterion="entropy",
                               n_jobs=-1)
c_rfc.fit(c_data, c_target)


# In[174]:


df["ischild"] = (df.Age < 16).astype(np.uint8)
df.loc[idx, "ischild"] = c_rfc.predict(df.loc[idx, :].drop(["PassengerId", "istrain", "ischild", "Age"], axis=1))


# In[175]:


### Age(fillna)


# In[176]:


df.query("ischild == 1").Age.dropna().astype(np.uint8).value_counts().sort_index().plot(kind="bar", color="b")


# In[183]:


na = df.loc[(df.ischild == 1)&(df.Age.isna()), "Age"].shape[0]
df.loc[(df.ischild == 1)&(df.Age.isna()), "Age"] = np.random.randint(0, 16, na)


# In[184]:


df.query("ischild == 0").Age.dropna().astype(np.uint8).value_counts().sort_index().plot(kind="bar", color="b")


# In[185]:


# ヒストグラムから乱数生成
idx = df.query("ischild == 0").Age.dropna().astype(np.uint8).value_counts().sort_index().index
val = df.query("ischild == 0").Age.dropna().astype(np.uint8).value_counts().sort_index().values
cum = np.add.accumulate(val) / np.sum(val)
na = na = df.loc[(df.ischild == 0)&(df.Age.isna()), "Age"].shape[0]
df.loc[(df.ischild == 0)&(df.Age.isna()), "Age"] = idx[np.searchsorted(cum, np.random.random(na))]


# In[186]:


df.query("ischild == 0").Age.dropna().astype(np.uint8).value_counts().sort_index().plot(kind="bar", color="b")


# In[187]:


df.isnull().sum()


# In[188]:


train = df.query("istrain == 1")
test = df.query("istrain == 0")
train.drop(["istrain", "PassengerId"], axis=1, inplace=True)
test.drop(["istrain", "PassengerId"], axis=1, inplace=True)


# In[189]:


X_train, X_test, y_train, y_test = train_test_split(train, target, train_size=0.8)


# # 決定木

# In[190]:


parameters = {"criterion": ["gini", "entropy"],
              "max_depth": range(1, 11)}
dtc = DecisionTreeClassifier()
clf = GridSearchCV(estimator=dtc,
                   param_grid=parameters,
                   cv=10,
                   n_jobs=-1,
                   verbose=1,
                   scoring="f1")
clf.fit(train, target)


# In[191]:


res = pd.DataFrame(clf.cv_results_)
x = res.query("param_criterion == 'gini'")["param_max_depth"]
for criterion in res["param_criterion"].unique():
    res_c = res.query("param_criterion == '{}'".format(criterion))
    x = res_c["param_max_depth"]
    y_mean = res_c["mean_test_score"]
    y_std = res_c["std_test_score"]
    plt.errorbar(x, y_mean, yerr=y_std, label=criterion)
plt.legend()
plt.ylabel("f1score")
plt.xlabel("max_depth")
plt.savefig("decision_tree.png")


# In[193]:


clf = DecisionTreeClassifier(criterion="gini",
                             max_depth=3)

clf.fit(train, target)


# In[16]:


#os.mkdir("../output")
#export_graphviz(clf, out_file="../output/tree.dot", feature_names=train.columns)


# In[17]:


#graph = pydotplus.graph_from_dot_file("../output/tree.dot")


# # random forest

# In[207]:


rfc = RandomForestClassifier()
parameters = {"criterion": ["gini", "entropy"],
              "max_depth": range(1, 8),
              "n_estimators": [3, 5, 10, 20, 40, 80, 160, 320]}
clf = GridSearchCV(estimator=rfc,
                   param_grid=parameters,
                   cv=10,
                   n_jobs=-1,
                   verbose=2,
                   scoring="f1")


# In[208]:


clf.fit(train, target)


# In[211]:


pd.DataFrame(clf.cv_results_)


# In[231]:


fig, axs = plt.subplots(1, 2, figsize=(12, 4))
res = pd.DataFrame(clf.cv_results_)

for i, criterion in enumerate(res["param_criterion"].unique()):
    for depth in sorted(res["param_max_depth"].unique()):
        res_c = res.query("param_criterion == '{}' & param_max_depth == {}"
                              .format(criterion, depth))
        x = res_c["param_n_estimators"].astype(np.float32)
        y_mean = res_c["mean_test_score"]
        y_std = res_c["std_test_score"]
        axs[i].errorbar(np.log(x), y_mean, yerr=y_std, label=depth)
    axs[i].set_xlabel("log(estimators)")
    axs[i].set_ylabel("f1 score")
    axs[i].set_title("criterion = {}".format(criterion))
    axs[i].legend()
plt.savefig("random_forest.png")


# In[232]:


rfc = RandomForestClassifier(n_estimators=20,
                             criterion="entropy",
                             max_depth=5)
rfc.fit(train, target)
pred = rfc.predict(test)


# In[233]:


Passenger_id = pd.read_csv("../input/test.csv")["PassengerId"]
res = pd.DataFrame([Passenger_id.values, pred]).T
res.columns = ["PassengerId", "Survived"]
res.to_csv("2nd_submit.csv", index=False)


# In[ ]:




