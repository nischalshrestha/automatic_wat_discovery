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


dataset = pd.read_csv('../input/train.csv')


# In[ ]:


testset = pd.read_csv('../input/test.csv')


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


dataset.columns


# In[ ]:


dataset.head()


# In[ ]:


print(dataset.dtypes)


# In[ ]:


print(dataset.describe())


# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察性别

# In[ ]:


Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()
dat = pd.DataFrame({'male': Survived_m, 'female': Survived_f})
dat.plot(kind = 'bar', stacked = True)
plt.title('Survival by sex')
plt.xlabel('Survival')
plt.ylabel('Count')
plt.show()


# 看看年龄

# In[ ]:


dataset['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Number')
plt.title('Age Distribution')
plt.show()

dataset[dataset.Survived == 0]['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age Distribution of people who did not survive')
plt.show()

dataset[dataset.Survived == 1]['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age Distribution of people who did survive')
plt.show()


# 看看船票价钱

# In[ ]:


dataset['Fare'].hist()
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare Distribution')
plt.show()

dataset[dataset.Survived == 0]['Fare'].hist()
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare Distribution of people who did not survive')
plt.show()

dataset[dataset.Survived == 1]['Fare'].hist()
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare Distribution of people who did survive')
plt.show()


# 观察乘客舱层

# In[ ]:


dataset['Pclass'].hist()
plt.show()

Survived_p1 = dataset.Survived[dataset['Pclass'] == 1].value_counts()
Survived_p2 = dataset.Survived[dataset['Pclass'] == 2].value_counts()
Survived_p3 = dataset.Survived[dataset['Pclass'] == 3].value_counts()

df = pd.DataFrame({'p1': Survived_p1, 'p2': Survived_p2, 'p3': Survived_p3})
df.plot(kind = 'bar', stacked = True)
plt.title('Survival by pclass')
plt.xlabel('Survival')
plt.ylabel('count')
plt.show()


# 观察登船地点

# In[ ]:


Survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()

df = pd.DataFrame({'S': Survived_S, 'C': Survived_C, 'Q': Survived_Q})
df.plot(kind = 'bar', stacked = True)
plt.title('Survival by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


label = dataset.loc[:, 'Survived']
data = dataset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdat = testset.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]


# 处理空数据

# In[ ]:


def fill_NAN(data):
    data_copy = data.copy(deep = True)
    data_copy['Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy['Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy['Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy['Sex'] = data_copy['Sex'].fillna('female')
    data_copy['Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

data_no_nan = fill_NAN(data)
testdat_no_nan = fill_NAN(testdat)
print(data_no_nan.isnull().values.any())


# 处理Sex 

# In[ ]:


def transfer_sex(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy
data_after_sex = transfer_sex(data_no_nan)
testdat_after_sex = transfer_sex(testdat_no_nan)


# 处理Embarked

# In[ ]:


def transfer_embark(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0 # loc is to access a gropu of rows
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_after_embarked = transfer_embark(data_after_sex)
testdat_after_embarked = transfer_embark(testdat_after_sex)


# 利用KNN训练数据

# In[ ]:


data_now = data_after_embarked
testdat_now = testdat_after_embarked
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data_now, label, random_state = 0, test_size = 0.2)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_scores = []
k_range = range(1, 51)
for k in k_range:
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(train_data, train_labels)
    print('k=', k)
    predictions = clf.predict(test_data)
    score = accuracy_score(test_labels, predictions)
    print(score)
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


# In[ ]:


# 检测模型precision， recall 等各项指标
result = clf.predict(testdat_now)


# In[ ]:





# In[ ]:


# 预测


# In[ ]:





# 打印输出

# In[ ]:


df = pd.DataFrame({'PassengerId': testset['PassengerId'], 'Survived': result})
df.to_csv('submission.csv', header = True)


# In[ ]:




