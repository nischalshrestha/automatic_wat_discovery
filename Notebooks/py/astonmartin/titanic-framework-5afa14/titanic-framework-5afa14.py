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


dataset=pd.read_csv('../input/train.csv')
testset=pd.read_csv('../input/test.csv')


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


dataset.dtypes


# In[ ]:


dataset.describe()


# In[ ]:


dataset.info


# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察年龄

# In[ ]:


s_m=dataset.Survived[dataset.Sex == 'male'].value_counts()
s_f=dataset.Survived[dataset.Sex == 'female'].value_counts()

df=pd.DataFrame({'male':s_m, 'female':s_f})
df.plot(kind='bar', stacked=True)
plt.title("Survived by sex")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()


# 看看年龄

# In[ ]:


dataset['Age'].hist()
plt.xlabel("Age")
plt.ylabel("Number")
plt.title("Age distribution")
plt.show()

dataset[dataset.Survived==0]['Age'].hist()
plt.xlabel("Age")
plt.ylabel("Number")
plt.title("Age distribution of non-survivor")
plt.show()

dataset[dataset.Survived==1]['Age'].hist()
plt.xlabel("Age")
plt.ylabel("Number")
plt.title("Age distribution of survivor")
plt.show()


# 看看船票价钱

# In[ ]:


dataset['Fare'].hist()
plt.xlabel("Fare")
plt.ylabel("Number")
plt.title("Fare distribution")
plt.show()

dataset[dataset.Survived==0]['Fare'].hist()
plt.xlabel("Fare")
plt.ylabel("Number")
plt.title("Fare distribution of non-survivor")
plt.show()

dataset[dataset.Survived==1]['Fare'].hist()
plt.xlabel("Fare")
plt.ylabel("Number")
plt.title("Fare distribution of survivor")
plt.show()


# 观察乘客舱层

# In[ ]:


dataset['Pclass'].hist()
plt.show()
print(dataset['Pclass'].isnull().values.any())

s_p1=dataset.Survived[dataset.Pclass==1].value_counts()
s_p2=dataset.Survived[dataset.Pclass==2].value_counts()
s_p3=dataset.Survived[dataset.Pclass==3].value_counts()

df=pd.DataFrame({'p1':s_p1, 'p2':s_p2, 'p3':s_p3})
print(df)
df.plot(kind='bar', stacked=True)
plt.title("Survived by Pclass")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()


# 观察登船地点

# In[ ]:


s_s=dataset.Survived[dataset.Embarked=='S'].value_counts()
s_c=dataset.Survived[dataset.Embarked=='C'].value_counts()
s_q=dataset.Survived[dataset.Embarked=='Q'].value_counts()

print(s_s)
df=pd.DataFrame({'S':s_s, 'C':s_c, 'Q':s_q})
df.plot(kind='bar', stacked=True)
plt.title("Survived by Embarked")
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


label=dataset.loc[:, 'Survived']
data=dataset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdata=testset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(data.shape)
data.head()


# 处理空数据

# In[ ]:


def fill_NAN(data):
    data_copy=data.copy(deep=True)
    data_copy.loc[:, 'Age']=data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:, 'Fare']=data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:, 'Pclass']=data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:, 'Sex']=data_copy['Sex'].fillna('female')
    data_copy.loc[:, 'Embarked']=data_copy['Embarked'].fillna('S')
    return data_copy

data_no_nan=fill_NAN(data)
testdata_no_nan=fill_NAN(testdata)

print(testdata.isnull().values.any())
print(testdata_no_nan.isnull().values.any())
print(data.isnull().values.any())
print(data_no_nan.isnull().values.any())

data_no_nan.head()


# 处理Sex 

# In[ ]:


print(data_no_nan['Sex'].isnull().values.any())

def transfer_sex(data):
    data_copy=data.copy(deep=True)
    data_copy.loc[data_copy.Sex=='female', 'Sex'] =0
    data_copy.loc[data_copy.Sex=='male', 'Sex']=1
    return data_copy

data_after_sex=transfer_sex(data_no_nan)
test_data_after_sex=transfer_sex(testdata_no_nan)
test_data_after_sex.head()


# 处理Embarked

# In[ ]:


def transfer_embark(data):
    data_copy=data.copy(deep=True)
    data_copy.loc[data_copy.Embarked=='S', 'Embarked'] = 0
    data_copy.loc[data_copy.Embarked=='C', 'Embarked'] = 1
    data_copy.loc[data_copy.Embarked=='Q', 'Embarked'] = 2
    return data_copy

data_after_embark=transfer_embark(data_after_sex)
test_data_after_embark=transfer_embark(test_data_after_sex)
test_data_after_embark.head()


# 利用KNN训练数据

# In[ ]:


data_now=data_after_embark
testdata_now=test_data_after_embark
from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(data_now, label, test_size=0.2, random_state=0)


# In[ ]:


print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range=range(1, 51)
k_scores=[]
for k in k_range:
    clf=KNeighborsClassifier(n_neighbors=k)
    clf.fit(train_data, train_label)
    print('K=', k)
    pred=clf.predict(test_data)
    score=accuracy_score(test_label, pred)
    print(score)
    k_scores.append(score)


# In[ ]:


# 预测
plt.plot(k_range, k_scores)
plt.show()
print(np.array(k_scores).argsort())


# In[ ]:


# 检测模型precision， recall 等各项指标
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


# In[ ]:


k=33
clf=KNeighborsClassifier(n_neighbors=k)
clf.fit(train_data, train_label)
pred=clf.predict(test_data)
#average_precision = average_precision_score(k_scores)
precision, recall, _ = precision_recall_curve(pred, test_label)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 0.8])
plt.xlim([0.0, 1.0])


# In[ ]:


# 预测


# In[ ]:


clf=KNeighborsClassifier(n_neighbors=33)
clf.fit(data_now,label)
result=clf.predict(testdata_now)


# 打印输出

# In[ ]:


df = pd.DataFrame({"PassengerId": testset['PassengerId'],"Survived": result})
df.to_csv('submission.csv',header=True, index=False)

