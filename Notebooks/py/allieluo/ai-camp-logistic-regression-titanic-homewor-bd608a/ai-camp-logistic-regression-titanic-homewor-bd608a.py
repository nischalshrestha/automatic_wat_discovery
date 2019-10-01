#!/usr/bin/env python
# coding: utf-8

# 预测Titanic乘客逃生

# In[ ]:


import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from patsy import dmatrix # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库


# > 从../input/train.csv读入数据

# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# 删除不需要的列以及含有NA值的行

# In[ ]:


data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()


# In[ ]:


data.dtypes


# 观察逃生人数与未逃生人数

# In[ ]:


surd = data.Survived.value_counts().plot(kind='bar')
surd.plot(kind='bar', label='Survived')
#plt.xlabel('Survived')
plt.show()


# 观察女性逃生人数

# In[ ]:


female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='bar', color='m', label='Female')
plt.show()


# 观察男性逃生人数

# In[ ]:


male = data.Survived[data.Sex == 'male'].value_counts().sort_index()
male.plot(kind='bar', color='b', label='male')
plt.show()


# 观察非低等舱逃生情况

# In[ ]:


highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()


# 观察低等舱逃生情况

# In[ ]:


lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar',label='Lowclass', color='y', alpha=0.6)
plt.show()


# dmatrices将数据中的离散变量变成哑变量，并指明用Pclass, Sex, Embarked来预测Survived

# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type='dataframe')
y = np.ravel(y)


# In[ ]:


X.shape


# In[ ]:


y.size


# In[ ]:


X.head()


# In[ ]:


y[:10]


# In[ ]:


model = LogisticRegression(C = 1e5)


# In[ ]:


model.fit(X, y)


# In[ ]:


answer = y
pred = model.predict(X)


# 输出训练准确率

# In[ ]:


model.score(X, y)


# 输出空模型的正确率：空模型预测所有人都未逃生

# In[ ]:


1 - y.mean()


# 观察模型系数，即每种因素对于预测逃生的重要性

# In[ ]:


pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))


# 对测试数据../input/test.csv生成预测，将结果写入./my_prediction.csv

# In[ ]:


testdata = pd.read_csv('../input/test.csv')


# In[ ]:


testdata.head()


# In[ ]:


testdata = testdata.drop(['Ticket', 'Cabin'], axis = 1)
testdata = testdata.dropna()


# In[ ]:


testdata.head()


# In[ ]:


testdata.shape


# In[ ]:


testdata['Survived'] = 1
testdata.loc[np.isnan(testdata.Age), 'Age'] = np.mean(data['Age'])


# In[ ]:


ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = testdata, return_type='dataframe')


# In[ ]:


Xtest.head()


# In[ ]:


pred = model.predict(Xtest).astype(int)
solution = pd.DataFrame(list(zip(testdata['PassengerId'], pred)), columns=['PassengerID', 'Survived'])


# In[ ]:


solution.to_csv('./my_prediction.csv', index = False)


# In[ ]:





# In[ ]:




