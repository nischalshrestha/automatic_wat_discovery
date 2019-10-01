#!/usr/bin/env python
# coding: utf-8

# # HomeWork1: KNN- Titanic-0915
# https://www.kaggle.com/c/titanic

# 加入numpy, pandas, matplot等库

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# *** 1. Load data and take a look

# In[ ]:


train= pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.shape, test.shape
#train.info()
train.columns


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


train.dtypes


# In[ ]:


train.describe()


# In[ ]:


# Check if there is missing data in datasset
train.isnull().values.any(), test.isnull().values.any()


# In[ ]:


train.head()


# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 1) Sex

# In[ ]:


male_s = train.Survived[train['Sex'] == 'male'].value_counts()
female_s = train.Survived[train['Sex'] == 'female'].value_counts()

df_sex = pd.DataFrame({'m': male_s, 'f': female_s})
df_sex.plot(kind ='bar', stacked = True)


# 2) Age

# In[ ]:


plt.subplot(131)
train['Age'].hist(density = True)
plt.subplot(132)
train.Age[train['Survived'] == 0].hist(density = True)
plt.subplot(133)
train.Age[train['Survived'] == 1].hist(density = True)


# 
# 3) Fare

# In[ ]:


plt.subplot(131)
plt.ylim(0,0.03)
train['Fare'].hist(density = True)
plt.ylim(0,0.03)
plt.subplot(132)
train.Fare[train['Survived'] == 0].hist(density = True)
plt.subplot(133)
plt.ylim(0,0.03)
train.Fare[train['Survived'] == 1].hist(density = True)


# 4) Pclass

# In[ ]:


P1_s = train.Survived[train['Pclass'] == 1].value_counts()
P2_s = train.Survived[train['Pclass'] == 2].value_counts()
P3_s = train.Survived[train['Pclass'] == 3].value_counts()

df_pclass = pd.DataFrame({'1': P1_s, '2': P2_s, '3': P3_s})
df_pclass.plot(kind ='bar', stacked = True)


# 5) Embarked Place

# In[ ]:


S_s = train.Survived[train['Embarked'] == 'S'].value_counts()
C_s = train.Survived[train['Embarked'] == 'C'].value_counts()
Q_s = train.Survived[train['Embarked'] == 'Q'].value_counts()

df_embarked = pd.DataFrame({'S': S_s, 'C': C_s, 'Q': Q_s})
df_embarked.plot(kind ='bar', stacked = True)


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


train_select = train.loc[:,['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
#train_select.head()
test_select=test.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
train_select.head()


# 处理空数据

# In[ ]:


def fill_NA(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[:, 'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

train_filled = fill_NA(train_select)
test_filled = fill_NA(test_select)
train_filled.head()


# 处理Sex 

# In[ ]:


def sex2num(data):
    data_copy = data.copy(deep = True)
    #data_copy.Sex[data_copy['Sex'] == 'female'] = 0
    #data_copy.Sex[data_copy['Sex'] == 'male'] = 1
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    
    return data_copy
    
train_sex2num = sex2num(train_filled)
test_sex2num = sex2num(test_filled)

train_sex2num.head()
    


# 处理Embarked

# In[ ]:


def embark2num(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy
    
train_embark2num = embark2num(train_sex2num)
test_embark2num = embark2num(test_sex2num)

train_embark2num.columns
    


# 利用KNN训练数据

# In[ ]:


train_data = train_embark2num
X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
#y = train_data[['Survived']]
y = train_data.loc[:,'Survived']
test_data = test_embark2num


# In[ ]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=0,test_size=0.2)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = range(1, 51)
k_scores = []
for K in k_range:
    clf=KNeighborsClassifier(n_neighbors = K)
    clf.fit(Xtrain,ytrain)
    print('K=', K)
    predictions=clf.predict(Xtest)
    score = accuracy_score(ytest,predictions)
    print(score)
    k_scores.append(score)
    


# In[ ]:


# 预测
plt.plot(k_range, k_scores)    
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())


# In[ ]:





# In[ ]:


# 预测
clf=KNeighborsClassifier(n_neighbors=33)
clf.fit(X,y)
y_pred = clf.predict(X)


# In[ ]:


# 检测模型precision， recall 等各项指标
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))


# In[ ]:


result=clf.predict(test_data)


# 打印输出

# In[ ]:


df = pd.DataFrame({"PassengerId": test['PassengerId'],"Survived": result})
df.to_csv('submission.csv',header=True, index=False)


# In[ ]:




