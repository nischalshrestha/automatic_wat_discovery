#!/usr/bin/env python
# coding: utf-8

# # Titanic
# https://www.kaggle.com/c/titanic

# 加入numpy, pandas, matplot等库

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # 画图常用库


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


dataset.shape, testset.shape


# In[ ]:


dataset.head(10)


# In[ ]:


dataset.dtypes


# In[ ]:


dataset.describe()


# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察性别

# In[ ]:


Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()
print(Survived_m)
print(Survived_f)


# In[ ]:


df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
df
#?plt.bar


# In[ ]:


df.plot(kind='bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("survived") 
plt.ylabel("count")
plt.show()


# 看看年龄

# In[ ]:


dataset['Age'].hist() 
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution')
plt.show() 

dataset[dataset.Survived==0]['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution of people who did not survive')
plt.show()

dataset[dataset.Survived==1]['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution of people who survived')
plt.show()


# In[ ]:


Survived_Age = dataset.groupby(['Survived', 'Age']).count()
Survived_Age.add_suffix('_Count').reset_index()


# 看看船票价钱

# In[ ]:





# 观察乘客舱层

# In[ ]:





# 观察登船地点

# In[ ]:





# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:





# 处理空数据

# In[ ]:





# 处理Sex 

# In[ ]:





# 处理Embarked

# In[ ]:





# 利用KNN训练数据

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# 预测


# In[ ]:


# 检测模型precision， recall 等各项指标


# In[ ]:


# cross validation 找到最好的k值


# In[ ]:





# In[ ]:


# 预测


# In[ ]:





# 打印输出

# In[ ]:




