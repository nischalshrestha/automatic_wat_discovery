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


data = pd.read_csv("../input/train.csv")


# 删除不需要的列以及含有NA值的行

# In[ ]:


data = data.drop(['Ticket','Cabin'], axis = 1)
data = data.dropna()


# In[ ]:


len(data.index)


# 观察逃生人数与未逃生人数

# In[ ]:


data.Survived.value_counts().plot(kind='bar')
plt.xlabel('Survived')
plt.show()


# 观察女性逃生人数

# In[ ]:


female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='barh', color='blue', label='Female')
plt.show()


# 观察男性逃生人数

# In[ ]:


male = data.Survived[data.Sex == 'male'].value_counts().sort_index()
male.plot(kind='barh',label='Male', color='red')
plt.show()


# 观察非低等舱逃生情况

# In[ ]:


highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()


# 观察低等舱逃生情况

# In[ ]:


lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar',label='Highclass', color='Blue', alpha=0.6)
plt.show()


# dmatrices将数据中的离散变量变成哑变量，并指明用Pclass, Sex, Embarked来预测Survived

# In[ ]:


y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type='dataframe')
y = np.ravel(y)


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(X, y)


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


test_data = pd.read_csv('../input/test.csv')
#test_data.head(): there is no label column “Survived”, for dmatrices purpose, you need to manually add dummy column
test_data['Survived'] = 1
test_data.head()
#test_data.isnull().sum(), will list NAN value counts for each column

#fill nan
# https://www.kaggle.com/jungan/ai-camp-logistic-regression-homework-025178?scriptVersionId=5790279
#data_copy=data.copy(deep=True)
#data_copy.loc[:, 'Age'] = data_copy.Age.fillna(data_copy.Age.median())

test_data.loc[np.isnan(test_data.Age), 'Age'] = np.mean(data['Age'])
# double check
#test_data.isnull().sum()
ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = test_data, return_type='dataframe')

pred = model.predict(Xtest).astype(int)
# columns=['PassengerID', 'Survived'] just for adding new titles for columns 
solution = pd.DataFrame(list(zip(test_data['PassengerId'], pred)), columns=['PassengerID', 'Survived'])
#solution.head()
solution.to_csv('./my_prediction.csv', index = False)


# In[ ]:




