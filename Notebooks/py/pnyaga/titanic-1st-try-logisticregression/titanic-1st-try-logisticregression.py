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

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# ------------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import sklearn as skl


# In[ ]:


def drawLine(x_plot, y_plot, x_train = None, model = None, title = ""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_plot, y_plot, c='g', marker='o')
    
    score = 0
    
    if (model is not None and x_train is not None):
        ax.plot(x_plot, model.predict(x_train), color='orange', linewidth=1, alpha=0.7)
        score = model.score(x_train, y_plot)
        
    title += " R2: " + str(score)
    ax.set_title(title)

    plt.show()


# In[ ]:


df  = pd.read_csv('../input/train.csv', sep=',')
#df  = pd.read_csv('train.csv', sep=',')
# print(df)


# In[ ]:


df.head(5)


# In[ ]:


df.columns


# In[ ]:


'''
VariableDefinitionKey 
survival Survival 0 = No, 1 = Yes 
pclass Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd 
sex Sex 
Age Age in years 
sibsp # of siblings / spouses aboard the Titanic 
parch # of parents / children aboard the Titanic 
ticket Ticket number 
fare Passenger fare 
cabin Cabin number 
embarked Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
'''


# In[ ]:


df.dtypes


# In[ ]:


df.describe()


# In[ ]:


df.Sex.head(5)


# In[ ]:


ordered_sex = ['female', 'male']
# df.Sex = df.Sex.astype("category",
#      ordered=True,
#      categories=ordered_sex).cat.codes

df.Sex = df.Sex.astype(pd.api.types.CategoricalDtype(categories = ordered_sex, ordered = True)).cat.codes


# In[ ]:


df.Sex.head(5)


# In[ ]:


del df['Name'], df['Embarked'], df['Ticket'], df['Cabin']


# In[ ]:


df.dtypes


# In[ ]:


df.describe()


# In[ ]:


dfc = df.iloc[:,1:]
dfc_res = dfc.corr()
print(dfc_res)


# In[ ]:


plt.imshow(dfc_res, cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(dfc_res.columns))]
plt.xticks(tick_marks, dfc_res.columns, rotation='vertical')
plt.yticks(tick_marks, dfc_res.columns)
plt.show()


# In[ ]:


pd.plotting.andrews_curves(dfc,"Survived")


# In[ ]:


pd.plotting.parallel_coordinates(dfc,"Survived")


# In[ ]:


x_train = df.iloc[:,2:]
x_train.head(3)


# In[ ]:


x_train.isnull().values.any()


# In[ ]:


x_train.plot.hist(alpha=0.75)


# In[ ]:


#print(x_train["Pclass"].unique())
#print(x_train["Sex"].unique())
print(x_train["Age"].unique())
#print(x_train["SibSp"].unique())
#print(x_train["Parch"].unique())
#print(x_train["Fare"].unique())


# In[ ]:


x_train["Age"] = x_train["Age"].fillna(x_train["Age"].mean())


# In[ ]:


x_train.describe()


# In[ ]:


import sklearn.preprocessing
min_max_scaler = skl.preprocessing.MinMaxScaler()
x_train_sc = min_max_scaler.fit_transform(x_train)
x_train = pd.DataFrame(data = x_train_sc[0:,0:],    # values
             #index = x_train_sc[0:,0],    # 1st column as index
             columns = x_train.columns) # 1st row as the column names


# In[ ]:


x_train.values.shape


# In[ ]:


x_train["Age"].plot.hist(alpha=0.5)


# In[ ]:


x_train["Pclass"].plot.hist(alpha=0.5)
x_train["Sex"].plot.hist(alpha=0.5)


# In[ ]:


x_train["SibSp"].plot.hist(alpha=0.5)


# In[ ]:


x_train["Parch"].plot.hist(alpha=0.5)


# In[ ]:


x_train["Fare"].plot.hist(alpha=0.5)


# In[ ]:


y_train = df.iloc[:,[1]]
y_train.head(3)


# In[ ]:


y_train.isnull().values.any()


# In[ ]:


y_train["Survived"].unique() 


# In[ ]:


y_train.plot.hist(alpha=0.75)


# In[ ]:


from sklearn import linear_model
model = linear_model.LogisticRegression()


# In[ ]:


model.fit(x_train.values, y_train.values.ravel())


# In[ ]:


score = model.score (x_train, y_train)
print(score)


# In[ ]:


df  = pd.read_csv('../input/test.csv', sep=',')
#df  = pd.read_csv('test.csv', sep=',')


# In[ ]:


del df['Name'], df['Embarked'], df['Ticket'], df['Cabin']


# In[ ]:


df.head(5)


# In[ ]:


ordered_sex = ['female', 'male']
df.Sex = df.Sex.astype(pd.api.types.CategoricalDtype(categories = ordered_sex, ordered = True)).cat.codes


# In[ ]:


df.isnull().values.any()


# In[ ]:


#print(df["Pclass"].unique())
#print(df["Sex"].unique())
#print(df["Age"].unique())
#print(df["SibSp"].unique())
#print(df["Parch"].unique())
print(df["Fare"].unique())


# In[ ]:


df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df.head(5)


# In[ ]:


x_test = df.iloc[:,1:]
x_test.head(5)


# In[ ]:


min_max_scaler = skl.preprocessing.MinMaxScaler()
x_test_sc = min_max_scaler.fit_transform(x_test)
x_test = pd.DataFrame(data = x_test_sc[0:,0:],    # values
             #index = x_train_sc[0:,0],    # 1st column as index
             columns = x_test.columns) # 1st row as the column names


# In[ ]:


x_test.head(5)


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


print(y_pred)


# In[ ]:


type(y_pred)


# In[ ]:


y = pd.DataFrame(data = y_pred[0:],    # values
             #index = x_train_sc[0:,0],    # 1st column as index
             columns = ["Survived"]) # 1st row as the column names


# In[ ]:


y.head(5)


# In[ ]:


output  = pd.DataFrame.join(df[["PassengerId"]], y[["Survived"]])


# In[ ]:


output.head(5)


# In[ ]:


output.to_csv("output.csv",index=False)

