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


data_dir = '../input/'
train_set = pd.read_csv(data_dir + 'train.csv')
test_set = pd.read_csv(data_dir + 'test.csv')


# In[ ]:


train_set.columns


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


train_set.head()


# In[ ]:


train_set.describe()


# In[ ]:


train_set.shape


# In[ ]:


train_set.dtypes


# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察年龄

# In[ ]:


Survived_m = train_set.Survived[train_set.Sex == 'male'].value_counts()
Survived_f = train_set.Survived[train_set.Sex == 'female'].value_counts()

df = pd.DataFrame({'male': Survived_m, 'female': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title('Survival rate by gender')
plt.xlabel('survived')
plt.ylabel('counts')
plt.show()


# 看看年龄

# In[ ]:


train_set.Age.hist()
train_set.Age[train_set.Survived == 1].hist()
plt.show()


# 看看船票价钱

# In[ ]:


train_set.Fare.hist()
train_set.Fare[train_set.Survived == 1].hist()
plt.show()


# 观察乘客舱层

# In[ ]:


train_set.Pclass.hist()

Survived_P1 = train_set.Survived[train_set.Pclass == 1].value_counts()
Survived_P2 = train_set.Survived[train_set.Pclass == 2].value_counts()
Survived_P3 = train_set.Survived[train_set.Pclass == 3].value_counts()

df1 = pd.DataFrame({'P1': Survived_P1, 'P2': Survived_P2, 'P3': Survived_P3})
df1.plot(kind='bar', stacked=True)


# 观察登船地点

# In[ ]:


Survived_S = train_set.Survived[train_set['Embarked'] == 'S'].value_counts()
Survived_C = train_set.Survived[train_set['Embarked'] == 'C'].value_counts()
Survived_Q = train_set.Survived[train_set['Embarked'] == 'Q'].value_counts()

df2 = pd.DataFrame({'S':Survived_S, 'C':Survived_C, 'Q':Survived_Q})
df2.plot(kind='bar', stacked=True)
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


label = train_set.Survived
train_data = train_set.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
test_data = test_set.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
print(label.shape, train_data.shape)


# 处理空数据

# In[ ]:


def fillna(data):
    data_copy = data.copy(deep=True)
    data_copy.Pclass = data_copy.Pclass.fillna(data.Pclass.mean())
    data_copy.Sex = data_copy.Sex.fillna('female')
    data_copy.Age = data_copy.Age.fillna(data.Age.mean())
    data_copy.Fare = data_copy.Fare.fillna(data.Fare.mean())
    data_copy.Embarked = data_copy.Embarked.fillna('S')
    return data_copy

train_no_nan = fillna(train_data)
test_no_nan = fillna(test_data)

print(train_data.isnull().values.any())
print(train_no_nan.isnull().values.any())

print(train_no_nan)


# 处理Sex 

# In[ ]:


def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy.Sex == 'female', 'Sex'] = 0
    data_copy.loc[data_copy.Sex == 'male', 'Sex'] = 1
    return data_copy

train_after_sex = transfer_sex(train_no_nan)
test_after_sex = transfer_sex(test_no_nan)


# 处理Embarked

# In[ ]:


def transfer_embarked(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

train_after_embarked = transfer_embarked(train_after_sex)
test_after_embarked = transfer_embarked(test_after_sex)


# 利用KNN训练数据

# In[ ]:


from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(train_after_embarked, label, random_state=0, test_size=0.2)


# In[ ]:


# 预测
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = range(1,50)
k_scores = []

for K in k_range:
    clf = KNeighborsClassifier(n_neighbors=K)
    clf.fit(train_data, train_label)
    predictions = clf.predict(test_data)
    score = accuracy_score(test_label, predictions)
    print('K = '+str(K)+', score = '+str(score))
    k_scores.append(score)


# In[ ]:


# 检测模型precision， recall 等各项指标
from sklearn.metrics import classification_report,confusion_matrix
clf = KNeighborsClassifier(n_neighbors=33)
clf.fit(train_data, train_label)
predictions = clf.predict(test_data)
print(classification_report(test_label, predictions))
print(confusion_matrix(test_label, predictions))


# In[ ]:


# 预测
clf = KNeighborsClassifier(n_neighbors=33)
clf.fit(train_after_embarked, label)
result = clf.predict(test_after_embarked)
print(result)


# 打印输出

# In[ ]:


df = pd.DataFrame({"PassengerId": test_set['PassengerId'], "Survived": result})
df.to_csv('submission.csv', header=True, index=False)



# In[ ]:




