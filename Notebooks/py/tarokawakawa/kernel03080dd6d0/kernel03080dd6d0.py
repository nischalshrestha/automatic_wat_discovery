#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df= pd.read_csv("../input/train.csv").replace("male",0).replace("female",1)


# In[ ]:


df["Age"].fillna(df.Age.median(),inplace=True)


# In[ ]:


split_data = []
for survived in [0,1]:
    split_data.append(df[df.Survived==survived])

temp = [i["Pclass"].dropna() for i in split_data] # dropna()

plt.hist(temp, histtype="barstacked", bins=3)


# In[ ]:


df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df2 = df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

print(df2.dtypes)


# In[ ]:


train_data = df2.values
print(train_data)
xs = train_data[:, 2:] # Pclass以降の変数 [:,]カンマは二次元配列の際に使う
                       # 2: ＝開始位置を指定し、最後までってこと
print(xs)
y  = train_data[:, 1]  # 正解データ n行の1列目だけほしい
print(y)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)

# 学習
forest = forest.fit(xs, y)

test_df= pd.read_csv("../input/test.csv").replace("male",0).replace("female",1)
# 欠損値の補完
test_df["Age"].fillna(df.Age.median(), inplace=True)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

test_df2 = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1) # 列指定で消す（axis=1)


# In[ ]:



test_data = test_df2.values
xs_test = test_data[:, 1:] # n行1列から
output = forest.predict(xs_test)

print(len(test_data[:,0]), len(output))
zip_data = zip(test_data[:,0].astype(int), output.astype(int)) # zip 関数はindexを与えてくれる
predict_data = list(zip_data) # astype()キャスト


# In[ ]:


import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])


# In[ ]:




