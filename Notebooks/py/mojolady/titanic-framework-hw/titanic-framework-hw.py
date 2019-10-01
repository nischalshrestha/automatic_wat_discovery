#!/usr/bin/env python
# coding: utf-8

# # Titanic
# https://www.kaggle.com/c/titanic

# 加入numpy, pandas, matplot等库

# In[ ]:


import numpy as np 
import pandas as pd 


# 读入数据

# In[ ]:


path = "../input"
train = pd.read_csv(path + "/train.csv")
test = pd.read_csv(path + "/test.csv")


# In[ ]:


print(train.columns)


# 0, PassengerId：乘客的数字id
# 
# 1, Survived：幸存(1)、死亡(0)
# 
# 2, Pclass：乘客船层—1st = Upper，2nd = Middle， 3rd = Lower
# 
# 3, Name：名字。
# 
# 4, Sex：性别
# 
# 5, Age：年龄
# 
# 6, SibSp：兄弟姐妹和配偶的数量。
# 
# 7, Parch：父母和孩子的数量。
# 
# 8, Ticket：船票号码。
# 
# 9, Fare：船票价钱。
# 
# 10, Cabin：船舱。
# 
# 11, Embarked：从哪个地方登上泰坦尼克号。 C = Cherbourg, Q = Queenstown, S = Southampton

# # 查看读入数据

# In[ ]:


train.head()


# In[ ]:


print(train.dtypes)


# In[ ]:


print(train.describe())


# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察年龄

# In[ ]:


import matplotlib.pyplot as plt
survived_male = train.Survived[train.Sex == 'male'].value_counts()
survived_female = train.Survived[train.Sex == 'female'].value_counts()

df = pd.DataFrame({'male': survived_male, 'female': survived_female})
df.plot(kind = 'bar', stacked = True)
plt.title("survived by sex")
plt.xlabel("survived")
plt.ylabel("count")
plt.show()


# 看看年龄

# In[ ]:


train['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution")
plt.show()

# survived == 0
train[train.Survived == 0]['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution of people who didn't survived")
plt.show()

# survived == 1
train[train.Survived == 1]['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution of people who survived")
plt.show()

print(train[train.Survived == 1]['Age'].head())


# 看看船票价钱

# In[ ]:


train.Fare.hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare distribution")
plt.show()

train[train.Survived == 0].Fare.hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare distribution for survivers")
plt.show()

train[train.Survived == 1].Fare.hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare distribution for non-survivers")
plt.show()


# 观察乘客舱层

# In[ ]:


train.Pclass.hist()
plt.show()
print(train.Pclass.isnull().values.any())

sur_cls1 = train.Survived[train.Pclass == 1].value_counts()
sur_cls2 = train.Survived[train.Pclass == 2].value_counts()
sur_cls3 = train.Survived[train.Pclass == 3].value_counts()

df = pd.DataFrame({"p1": sur_cls1, "p2": sur_cls2, "p3":sur_cls3})
print(df)
df.plot(kind = "bar", stacked = True)
plt.title("survived by pclass")
plt.xlabel("pclass")
plt.ylabel("count")
plt.show()


# 观察登船地点

# In[ ]:


sur_cls1 = train.Survived[train.Embarked == 'S'].value_counts()
sur_cls2 = train.Survived[train.Embarked == 'C'].value_counts()
sur_cls3 = train.Survived[train.Embarked == 'Q'].value_counts()

df = pd.DataFrame({"S": sur_cls1, "C": sur_cls2, "Q": sur_cls3})
df.plot(kind = "bar", stacked = True)
plt.title("survived by embarking location")
plt.xlabel("embarked")
plt.ylabel("count")
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


train_select = train[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
test_select = test[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
label = train[['Survived']]
print(label.head())
print(train_select.shape)


# 处理空数据

# In[ ]:


def fill_NaN(data):
    data_copy = data.copy(deep= True)
    data_copy.loc[:,"Age"] = data_copy["Age"].fillna(data_copy["Age"].median())
    data_copy.loc[:,"Fare"] = data_copy["Fare"].fillna(data_copy["Fare"].median())
    data_copy.loc[:,"Pclass"] = data_copy["Pclass"].fillna(data_copy["Pclass"].median())
    data_copy.loc[:,"Sex"] = data_copy["Sex"].fillna("female")
    data_copy.loc[:,"Embarked"] = data_copy["Embarked"].fillna("S")
    return data_copy

train_no_nan = fill_NaN(train_select)
test_no_nan = fill_NaN(test_select)
print(train_no_nan.isnull().values.any())
print(train.isnull().values.any())
print(train_no_nan.shape, test_no_nan.shape)


# 处理Sex 

# In[ ]:


print(train_no_nan["Sex"].isnull().values.any())
print(train_no_nan.Sex.value_counts())
def transfer_sex(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy.Sex == 'female', 'Sex']= 0
    data_copy.loc[data_copy.Sex == 'male', 'Sex']= 1
    return data_copy

train_after_sex = transfer_sex(train_no_nan)
test_after_sex = transfer_sex(test_no_nan)
print(train_after_sex.shape, test_after_sex.shape)
print(train_after_sex.Sex.value_counts())


# 处理Embarked

# In[ ]:


def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy.Embarked == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy.Embarked == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy.Embarked == 'Q', 'Embarked'] = 2
    return data_copy

train_after_embarked = transfer_embark(train_after_sex)
test_after_embarked = transfer_embark(test_after_sex)
print(train_after_embarked.Embarked.value_counts())


# 利用KNN训练数据

# In[ ]:


train_now = train_after_embarked
test_now = test_after_embarked
from sklearn.model_selection import train_test_split


# In[ ]:


train_data,test_data,train_labels, test_labels = train_test_split(train_now, label, random_state=0, test_size = 0.2)


# In[ ]:


print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)


# In[ ]:


# 预测
print(label)
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
k_range = range(1,51)
k_scores = []

for k in k_range:
    clf = KNN(n_neighbors = k)
    clf.fit(train_data,train_labels)
    print("K=", k)
    predictions = clf.predict(test_data)
    score = accuracy_score(test_labels,predictions)
    print(score)
    k_scores.append(score)


# In[ ]:


# 检测模型precision， recall 等各项指标
plt.plot(k_range,k_scores)
plt.xlabel("K for KNN")
plt.ylabel("Accuracy on validation set")
plt.show()
print(np.array(k_scores).argsort())


# In[ ]:


clf = KNN(n_neighbors = 33)
clf.fit(train_now, label)
result=clf.predict(test_now)


# In[ ]:


# 预测
print(result)


# In[ ]:


df = pd.DataFrame({'PassengerId': test['PassengerId'], "Survived": result})


# 打印输出

# In[ ]:


print(df.shape)
print(df.head())


# In[ ]:


df.to_csv('submission.csv', header=True, index = False)

