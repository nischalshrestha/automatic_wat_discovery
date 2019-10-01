#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


##### https://qiita.com/suzumi/items/8ce18bc90c942663d1e6
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df= pd.read_csv("../input/train.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)


# In[ ]:


df


# In[ ]:


df["Age"].fillna(df.Age.mean(), inplace=True)


# In[ ]:


df2 = df.drop(["Name", "Ticket","Cabin"], axis=1)
df2 = df2.drop(df2.columns[np.isnan(df2).any()], axis=1)


# In[ ]:


df2


# In[ ]:


train_data = df2.values
xs = train_data[:, 2:] # Pclass以降の変数
y  = train_data[:, 1]  # 正解データ


# In[ ]:


forest = RandomForestClassifier(n_estimators = 100)

# 学習
forest = forest.fit(xs, y)


# In[ ]:


print forest.feature_importances_


# In[ ]:


test_df= pd.read_csv("../input/test.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
# 欠損値の補完
test_df["Age"].fillna(df.Age.mean(), inplace=True)
test_df2 = test_df.drop(["Name", "Ticket","Cabin"], axis=1)
#NaN行削除
test_df2 = test_df2.drop(test_df2.columns[np.isnan(test_df2).any()], axis=1)


# In[ ]:


test_data = test_df2.values
xs_test = test_data[:, 1:]
output = forest.predict(xs_test)

print(len(test_data[:,0]), len(output))
zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)


# In[ ]:


import csv
with open("../input/predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])


# In[ ]:




