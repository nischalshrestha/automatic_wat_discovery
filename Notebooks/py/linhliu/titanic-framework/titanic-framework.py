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


train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


test_data.columns


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


train_data.head()


# In[ ]:


print(train_data.dtypes)


# In[ ]:


print(train_data.describe())


# In[ ]:


print(train_data.info())


# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察年龄

# In[ ]:


import matplotlib.pyplot as plt
survived_f = train_data.Survived[train_data.Sex == 'female'].value_counts()
survived_m = train_data.Survived[train_data.Sex == 'male'].value_counts()

df = pd.DataFrame({'female' : survived_f,'male':survived_m})
df.plot(kind = 'bar', stacked = True)
plt.title('Survived by Sex')
plt.xlabel('Survived')
plt.ylabel('count')
plt.show()


# 看看年龄

# In[ ]:


train_data['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution')
plt.show()

train_data[train_data.Survived ==0]['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution')
plt.show()


# 看看船票价钱

# In[ ]:


train_data['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare distribution')
plt.show() 

train_data[train_data.Survived==0]['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare distribution of people who did not survive')
plt.show()


# 观察乘客舱层

# In[ ]:


train_data['Pclass'].hist()  
plt.show()  
#print(dataset['Pclass'].isnull().values.any())

Survived_p1 = train_data.Survived[train_data['Pclass'] == 1].value_counts()
Survived_p2 = train_data.Survived[train_data['Pclass'] == 2].value_counts()
Survived_p3 = train_data.Survived[train_data['Pclass'] == 3].value_counts()

df=pd.DataFrame({'p1':Survived_p1, 'p2':Survived_p2, 'p3':Survived_p3})
print(df)
df.plot(kind='bar', stacked=True)
plt.title("survived by pclass")
plt.xlabel("pclass") 
plt.ylabel("count")
plt.show()


# 观察登船地点

# In[ ]:


Survived_S = train_data.Survived[train_data['Embarked'] == 'S'].value_counts()
Survived_C = train_data.Survived[train_data['Embarked'] == 'C'].value_counts()
Survived_Q = train_data.Survived[train_data['Embarked'] == 'Q'].value_counts()

print(Survived_S)
df = pd.DataFrame({'S':Survived_S, 'C':Survived_C, 'Q':Survived_Q})
df.plot(kind='bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("Embarked") 
plt.ylabel("count")
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


label = train_data.loc[:,'Survived']
data = train_data.loc[:,['Pclass','Sex','Age','Fare','Embarked']]
test = test_data.loc[:,['Pclass','Sex','Age','Fare','Embarked']]

print(data.shape)


# 处理空数据

# In[ ]:


def fill_NAN(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy
data_NoNaN = fill_NAN(data)
test_NoNaN = fill_NAN(test)
print(data_NoNaN.isnull().values.any())
print(test_NoNaN.isnull().values.any())


# 处理Sex 

# In[ ]:


def Sex_Trans(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Sex'] == 'female','Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male','Sex'] = 1
    return data_copy
data_after_sex = Sex_Trans( data_NoNaN )
test_after_sex = Sex_Trans( test_NoNaN )


# 处理Embarked

# In[ ]:


def Embark_Trans(data):
    data_copy = data.copy(deep= True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy
data_after_embarked = Embark_Trans(data_after_sex)
test_after_embarked = Embark_Trans(test_after_sex)


# 利用KNN训练数据

# In[ ]:


data_now = data_after_embarked
test_now = test_after_embarked

from sklearn.model_selection import train_test_split
train_data,test_test,train_labels,test_labels=train_test_split(data_now,label,random_state=0,test_size=0.2)


# In[ ]:


print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = range(1,51)
k_scores = []
for k in k_range:
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(train_data,train_labels)
    print('k is' , k)
    predictions = clf.predict(test_test)
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

clf=KNeighborsClassifier(n_neighbors=33)
clf.fit(data_now,label)
result=clf.predict(test_now)


# In[ ]:


# 检测模型precision， recall 等各项指标


# In[ ]:


# cross validation 找到最好的k值
from sklearn.model_selection import cross_val_score


# In[ ]:


k_range = range(1,51)
k_scores_cv = []
for k in range(1,51):    
    clf = KNeighborsClassifier(n_neighbors = k)
    clf_cv_score = cross_val_score(clf,train_data,train_labels,cv = 10,scoring='accuracy').mean()
    k_scores_cv.append(clf_cv_score)
    print(clf_cv_score)


# In[ ]:


plt.plot(k_range, k_scores_cv)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set - cv')
plt.show()
print(np.array(k_scores_cv).argsort())


# In[ ]:


# 预测
clf=KNeighborsClassifier(n_neighbors=21)
clf.fit(data_now,label)
result=clf.predict(test_now)


# In[ ]:





# 打印输出

# In[ ]:




