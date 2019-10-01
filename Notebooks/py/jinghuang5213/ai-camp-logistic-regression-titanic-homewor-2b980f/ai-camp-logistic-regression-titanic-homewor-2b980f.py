#!/usr/bin/env python
# coding: utf-8

# 预测Titanic乘客逃生

# In[ ]:


import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库


# > 从../input/train.csv读入数据

# In[ ]:


data=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# 删除不需要的列以及含有NA值的行

# In[ ]:


data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()


# In[ ]:


len(data.index)


# 观察逃生人数与未逃生人数

# In[ ]:


data.Survived.value_counts().plot(kind='bar')


# 观察女性逃生人数

# In[ ]:


female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='bar')


# 观察男性逃生人数

# In[ ]:


female = data.Survived[data.Sex == 'male'].value_counts()
female.plot(kind='bar') 


# 观察非低等舱逃生情况

# In[ ]:


highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar')
plt.show()


# 观察低等舱逃生情况

# In[ ]:


highclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
highclass.plot(kind='bar')
plt.show()


# dmatrices将数据中的离散变量变成哑变量，并指明用Pclass, Sex, Embarked来预测Survived

# In[ ]:


y, X = dmatrices('Survived~C(Pclass) + C(Sex) + Age + C(Embarked)', data=data, return_type='dataframe')
y = np.ravel(y)


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(X, y)


# 输出训练准确率

# In[ ]:


model.score(X,y)


# 输出空模型的正确率：空模型预测所有人都未逃生

# In[ ]:


1-y.mean()


# 观察模型系数，即每种因素对于预测逃生的重要性

# In[ ]:


pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))


# 对测试数据../input/test.csv生成预测，将结果写入./my_prediction.csv

# In[ ]:


test['Survived'] = 0
test.loc[np.isnan(test.Age), 'Age'] = np.mean(data['Age'])
ytest, Xtest=dmatrices('Survived~C(Pclass)+C(Sex)+Age+C(Embarked)', data=test,return_type='dataframe')
pred=model.predict(Xtest).astype(int)
sol=pd.DataFrame(list(zip(test['PassengerId'],pred)), columns=['PassengerId', 'Survived'])
sol.to_csv('./my_prediction.csv', index = False)

