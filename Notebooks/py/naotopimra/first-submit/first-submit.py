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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


df= pd.read_csv("../input/train.csv").replace("male",0).replace("female",1)


# In[ ]:


df["Age"].fillna(df.Age.median(), inplace=True)


# In[ ]:


split_data = []


# In[ ]:


for survived in [0,1]:
    split_data.append(df[df.Survived==survived])


# In[ ]:


print(split_data)


# In[ ]:


temp = [i["Pclass"].dropna() for i in split_data]


# In[ ]:


print(temp)


# In[ ]:


plt.hist(temp, histtype="barstacked", bins=3)


# In[ ]:


temp = [i["Age"].dropna() for i in split_data]


# In[ ]:


plt.hist(temp, histtype="barstacked", bins=3)


# In[ ]:


plt.hist(temp, histtype="barstacked", bins=16)


# In[ ]:


df["FamilySize"] = df["SibSp"] + df["Parch"] + 1


# In[ ]:


df2 = df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)


# In[ ]:


df2.dtypes


# In[ ]:


train_data=df2.values


# In[ ]:


xs = train_data[:,2:]


# In[ ]:


print(xs)


# In[ ]:


y=train_data[:,1]


# In[ ]:


forest = RandomForestClassifier(n_estimators=100)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


forest = RandomForestClassifier(n_estimators=100)


# In[ ]:


forest = forest.fit(xs, y)


# In[ ]:


test_df = pd.read_csv("../input/test.csv").replace("male", 0).replace("female",1)


# In[ ]:


test_df["Age"].fillna(df.Age.median(), inplace=True)


# In[ ]:


test_df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
test_df2 = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)


# In[ ]:


print(test_df2)


# In[ ]:


test_data=test_df2.values


# In[ ]:


print(test_data)


# In[ ]:


xs_test = test_data[:, 1:]


# In[ ]:


output = forest.predict(xs_test)


# In[ ]:


print(len(test_data[:,0]), len(output))


# In[ ]:


zip_data = zip(test_data[:,0].astype(int), output.astype(int))


# In[ ]:


predict_data = list(zip_data)


# In[ ]:


print(predict_data)


# In[ ]:


import csv


# In[ ]:


with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])


# In[ ]:


pwd


# In[ ]:


ls


# In[ ]:




