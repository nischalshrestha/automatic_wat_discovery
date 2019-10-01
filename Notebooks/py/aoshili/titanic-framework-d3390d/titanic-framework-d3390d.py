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
testset = pd.read_csv('../input/test.csv')


# In[ ]:


dataset.columns


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


print(dataset.dtypes)


# In[ ]:


print(dataset.describe())


# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察年龄

# In[ ]:


Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()

print('Survived male value counts', Survived_m)
print('Survived female value counts', Survived_f)

df = pd.DataFrame({'male': Survived_m, 'female': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title('survived by sex')
plt.xlabel('survived')
plt.ylabel('count')
plt.show()


# 看看年龄

# In[ ]:


dataset['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution')
plt.show()

dataset[dataset.Survived == 0]['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution of dead')
plt.show()

dataset[dataset.Survived == 1]['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution of Survived')
plt.show()


# 看看船票价钱

# In[ ]:


dataset['Fare'].hist()
plt.xlabel('Fare')
plt.ylabel('Number')
plt.title('Fare distribution')
plt.show()

dataset[dataset.Survived == 0]['Fare'].hist()
plt.xlabel('Fare')
plt.ylabel('Number')
plt.title('Fare distribution of dead')
plt.show()


dataset[dataset.Survived == 1]['Fare'].hist()
plt.xlabel('Fare')
plt.ylabel('Number')
plt.title('Fare distribution of survived')
plt.show()


# 观察乘客舱层

# In[ ]:


dataset['Pclass'].hist()
plt.show()
print(dataset['Pclass'].isnull().values.any())

Survived_p1 = dataset.Survived[dataset['Pclass'] == 1].value_counts()
Survived_p2 = dataset.Survived[dataset['Pclass'] == 2].value_counts()
Survived_p3 = dataset.Survived[dataset['Pclass'] == 3].value_counts()

df = pd.DataFrame({'p1': Survived_p1, 'p2': Survived_p2, 'p3': Survived_p3})
print(df)
df.plot(kind='bar', stacked=True)
plt.title('survived by pclass')
plt.xlabel('pclass')
plt.ylabel('count')
plt.show()


# 观察登船地点

# In[ ]:


dataset['Embarked'].isnull().values.any()
Survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()

print(Survived_S)
df = pd.DataFrame({'S': Survived_S, 'C': Survived_C, 'Q': Survived_Q})
df.plot(kind='bar', stacked=True)
plt.title('Survived by embark')
plt.xlabel('Embarked')
plt.ylabel('count')
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


label = dataset.loc[:, 'Survived']
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
data = dataset.loc[:, features]
testdata = testset.loc[:, features]

print(data.shape)
print(data)


# 处理空数据

# In[ ]:


def fill_NAN(data):
    data_copy = data.copy(deep=True)
    for feature in features:
        target = None
        if feature == 'Sex':
            target = 'female'
        elif feature == 'Embarked':
            target = 'S'
        else:
            target = data_copy[feature].median()
        data_copy.loc[:, feature] = target = data_copy[feature].fillna(target)
    return data_copy

data_no_nan = fill_NAN(data)
testdata_no_nan = fill_NAN(testdata)

print(testdata.isnull().values.any())
print(testdata_no_nan.isnull().values.any())
print(data.isnull().values.any())
print(data_no_nan.isnull().values.any())

print(data_no_nan)


# 处理Sex 

# In[ ]:


def transfer_sex(data):
    data_sex_int = data.copy(deep=True)
    sex_to_int = {'male': 1, 'female': 0}
    data_sex_int['Sex'] = data_sex_int['Sex'].map(sex_to_int)
    return data_sex_int
data_sex_int = transfer_sex(data_no_nan)
testdata_sex_int = transfer_sex(testdata_no_nan)
print(testdata_sex_int)


# 处理Embarked

# In[ ]:


from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
def transfer_embarked(data):
    data_embark_int = data.copy(deep=True)
    y = class_le.fit_transform(data_embark_int['Embarked'].values)
    data_embark_int['Embarked'] = y
    return data_embark_int
data_embark_int = transfer_embarked(data_sex_int)
testdata_embark_int = transfer_embarked(testdata_sex_int)
print(testdata_embark_int)


# 利用KNN训练数据

# In[ ]:


data_now = data_embark_int
testdata_now = testdata_embark_int
from sklearn.model_selection import train_test_split
train_X, vali_X, train_y, vali_y = train_test_split(data_now, label, random_state=0, test_size=0.2)


# In[ ]:


print(train_X.shape, vali_X.shape, train_y.shape, vali_y.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = range(1, 51)
k_to_score = {}
for K in k_range:
    clf = KNeighborsClassifier(n_neighbors=K)
    clf.fit(train_X, train_y)
    print('K=', K)
    predictions = clf.predict(vali_X)
    score = accuracy_score(vali_y, predictions)
    print(score)
    k_to_score[K] = score


# In[ ]:


# 预测
best_k = 1
max_score = 0.0
for k, score in k_to_score.items():
    if score > max_score:
        print(score)
        max_score = score
        best_k = k
print('best k', best_k)
print('score', k_to_score[best_k])


# In[ ]:


# 检测模型precision， recall 等各项指标
from sklearn.metrics import precision_score, recall_score
clf = KNeighborsClassifier(n_neighbors=best_k)
clf.fit(train_X, train_y)
predictions = clf.predict(vali_X)
print('Precision', precision_score(vali_y, predictions))
print('Recall', recall_score(vali_y, predictions))


# In[ ]:


# 预测
result = clf.predict(testdata_now)


# In[ ]:


print(result)


# 打印输出

# In[ ]:


df = pd.DataFrame({'PassengerId': testset['PassengerId'], 'Survived': result})
df.to_csv('submission.csv', header=True, index=False)


# In[ ]:




