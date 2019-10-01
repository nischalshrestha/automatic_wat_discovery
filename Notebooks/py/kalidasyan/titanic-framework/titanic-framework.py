#!/usr/bin/env python
# coding: utf-8

# # Titanic
# https://www.kaggle.com/c/titanic

# 加入numpy, pandas, matplot等库

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# 读入数据

# In[ ]:


data_dir = '../input'
dataset = pd.read_csv(data_dir + '/train.csv')
testset = pd.read_csv(data_dir + '/test.csv')


# In[ ]:


print(dataset.columns)


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


dataset.head()


# In[ ]:


dataset.dtypes


# In[ ]:


dataset.describe()


# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据有字符串

# # 仔细观察数据

# 观察年龄

# In[ ]:


Survived_male = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_female = dataset.Survived[dataset.Sex == 'female'].value_counts()

df = pd.DataFrame({'male':Survived_male, 'female':Survived_female})
df.plot(kind='bar')
plt.title("survived by sex")
plt.xlabel('survived')
plt.ylabel('count')
plt.show()


# 看看年龄

# In[ ]:


dataset['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title('Age distribution')
plt.show()

#查看存活的年龄分布
dataset[dataset.Survived == 1]['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution of people who survived")
plt.show()

#查看没存活的年龄分布
dataset[dataset.Survived == 0]['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution of people who didn't survive")
plt.show()


# 看看船票价钱

# In[ ]:


dataset['Fare'].hist()
plt.xlabel("Fare")
plt.ylabel("Number")
plt.title("Fare distribution")
plt.show()

dataset[dataset.Survived == 1]['Fare'].hist()
plt.xlabel("Fare")
plt.ylabel("Number")
plt.title("Fare distribution of people who survived")
plt.show()

dataset[dataset.Survived == 0]['Fare'].hist()
plt.xlabel("Fare")
plt.ylabel("Number")
plt.title("Fare distribution of people who didn't survive")
plt.show()


# 观察乘客舱层

# In[ ]:


dataset['Pclass'].hist()
plt.xlabel("Pclass")
plt.ylabel("Number")
plt.show()
print(dataset['Pclass'].isnull().values.any())

Survived_p1 = dataset[dataset['Pclass'] == 1]['Survived'].value_counts()
Survived_p2 = dataset[dataset['Pclass'] == 2]['Survived'].value_counts()
Survived_p3 = dataset[dataset['Pclass'] == 3]['Survived'].value_counts()

df = pd.DataFrame({'p1':Survived_p1, 'p2':Survived_p2, 'p3':Survived_p3})
df.plot(kind='bar', stacked=True)
plt.title("survived by pclass")
plt.xlabel("pclass") 
plt.ylabel("count")
plt.show()


# 观察登船地点

# In[ ]:


Survived_S = dataset.Survived[dataset.Embarked == 'S'].value_counts()
Survived_C = dataset.Survived[dataset.Embarked == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset.Embarked == 'Q'].value_counts()

df = pd.DataFrame({'S':Survived_S, 'C':Survived_C, 'Q':Survived_Q})
df.plot(kind = 'bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("Embarked")
plt.ylabel("count")
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


label = dataset.loc[:, 'Survived']
data = dataset.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdat = testset.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(data.shape)
print(data.head())
print(testdat.shape)


# 处理空数据

# In[ ]:


def fill_NAN(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:, 'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:, 'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:, 'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:, 'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

data_no_nan = fill_NAN(data)
testdat_no_nan = fill_NAN(testdat)

print(data.isnull().values.any())
print(data_no_nan.isnull().values.any())
print(testdat.isnull().values.any())
print(testdat_no_nan.isnull().values.any())

print(data_no_nan)


# 处理Sex 

# In[ ]:


print(data_no_nan['Sex'].isnull().values.any())

def transfer_sex(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

data_after_sex = transfer_sex(data_no_nan)
testdat_after_sex = transfer_sex(testdat_no_nan)
print(testdat_after_sex)


# 处理Embarked

# In[ ]:


def transfer_embarked(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_after_embarked = transfer_embarked(data_after_sex)
testdat_after_embarked = transfer_embarked(testdat_after_sex)
print(testdat_after_embarked)


# 利用KNN训练数据

# In[ ]:


data_now = data_after_embarked
testdat_now = testdat_after_embarked
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(data_now, label, test_size = 0.2, random_state = 0)


# In[ ]:


print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

k_range = range(1, 51)
k_scores = []

for K in k_range:
    clf = KNeighborsClassifier(n_neighbors = K)
    clf.fit(train_data, train_labels)
    print('K=', K)
    predictions = clf.predict(test_data)
    score = accuracy_score(test_labels, predictions)
    print(score)
    # 检测模型precision， recall 等各项指标
    print(classification_report(test_labels, predictions))
    k_scores.append(score)


# In[ ]:


plt.plot(k_range, k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())


# In[ ]:


# 预测
clf = KNeighborsClassifier(n_neighbors = 33)
clf.fit(data_now, label)
result = clf.predict(testdat_now)


# In[ ]:


print(result)


# 打印输出

# In[ ]:


df = pd.DataFrame({"PassengerId": testset['PassengerId'], "Survived": result})
df.to_csv("submission.csv", header=True, index=False)


# In[ ]:




