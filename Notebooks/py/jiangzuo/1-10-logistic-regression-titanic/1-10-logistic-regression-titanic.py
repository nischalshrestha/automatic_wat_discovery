#!/usr/bin/env python
# coding: utf-8

# 预测Titanic乘客逃生

# In[1]:


import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库


# 从../input/train.csv读入数据

# In[2]:


data = pd.read_csv('../input/train.csv')


# 删除不需要的列以及含有NA值的行

# In[3]:


#embarked Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()


# In[4]:


len(data.index)


# 观察逃生人数与未逃生人数

# In[5]:


data.Survived.value_counts().plot(kind='bar')
plt.xlabel('Survived')
plt.show()


# 观察女性逃生人数

# In[6]:


female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='barh', color='blue', label='Female')
plt.show()


# 观察男性逃生人数

# In[7]:


male = data.Survived[data.Sex == 'male'].value_counts().sort_index()
male.plot(kind='barh',label='Male', color='red')
plt.show()


# 观察非低等舱逃生情况

# In[8]:


highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()


# 观察低等舱逃生情况

# In[9]:


lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar',label='Highclass', color='Blue', alpha=0.6)
plt.show()


# dmatrices将数据中的离散变量变成哑变量，并指明用Pclass, Sex, Embarked来预测Survived

# In[10]:


y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type='dataframe')
y = np.ravel(y)


# In[11]:


model = LogisticRegression()


# In[12]:


model.fit(X, y)


# 输出训练准确率

# In[13]:


model.score(X, y)


# 输出空模型的正确率：空模型预测所有人都未逃生

# In[14]:


1 - y.mean()


# 观察模型系数，即每种因素对于预测逃生的重要性

# In[15]:


pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))


# 对测试数据生成预测

# In[30]:


test_data = pd.read_csv('../input/test.csv')


# In[31]:


test_data['Survived'] = 1
test_data.loc[np.isnan(test_data.Age), 'Age'] = np.mean(data['Age'])


# In[32]:


ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = test_data, return_type='dataframe')


# In[33]:


pred = model.predict(Xtest).astype(int)
solution = pd.DataFrame(list(zip(test_data['PassengerId'], pred)), columns=['PassengerID', 'Survived'])


# In[34]:


solution.to_csv('./my_prediction.csv', index = False)


# In[ ]:




